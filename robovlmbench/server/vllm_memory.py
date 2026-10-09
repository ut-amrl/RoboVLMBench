"""Memory for a vLLM server, from what the model needs rather than a fixed share of the device: the KV cache of
exactly one request at the longest length it will be sent (requests go one at a time), on top of the weights
and the working memory vLLM measures at startup. Standard library only: the edge package (robovlmbench/edge)
ships a copy next to its client.

vLLM sizes the KV cache from --kv-cache-memory-bytes when it is given, and reserves nothing else up front. It
also checks at startup that free memory is at least --gpu-memory-utilization of the total (default 0.9, more
than a device shared with other programs may have free), so that is set to what the model needs too.

The KV cache a model needs depends on its layers (hybrid models also keep a fixed state per request) and on
vLLM's page layout, so it is vLLM's own figure: the first start of a model at a given length (a probe) asks for
a small KV cache and is stopped once vLLM has sized it. vLLM then either stopped, saying how much is needed, or
reports how many requests of that length the small cache holds. vLLM also holds back one block of the KV
cache (MBs to tens of MBs), which its message shows too (cache given minus cache available). A start that still
finds the cache short says by how much (kv_shortfall), and is run again with that added. The figure is saved
(MemoryNeeds) for the next starts, with the server's other memory (weights and working memory) once a start
has measured it.
"""

from __future__ import annotations  # the edge package's copy runs on Python 3.8+

import fcntl
import json
import os
import re
import subprocess
import threading
from pathlib import Path

GiB = 1 << 30
MiB = 1 << 20
PROBE_KV_BYTES = (64 * MiB, 512 * MiB, 4 * GiB)  # tried in turn: the first one larger than a block decides
ROUNDING = int(0.02 * GiB)  # vLLM reports these figures rounded to 0.01 GiB
PROBE_DONE = ("GiB KV cache is needed", "No available memory for the cache blocks", "Maximum concurrency for")
_NEEDED = re.compile(r"\(([\d.]+) GiB KV cache is needed, which is larger than the available KV cache memory "
                     r"\(([\d.]+) GiB\)")
_CONCURRENCY = re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x")


def server_args(kv_cache_bytes: int, other_bytes: int, total_bytes: int, reserve: float = 1.0) -> list[str]:
    """vLLM arguments for one request at a time: the server's need is the KV cache of `kv_cache_bytes` plus
    `other_bytes`, the rest of what it takes (weights and working memory, as measured on an earlier start; the
    weights alone before there is one). `reserve` times the need is reserved: what's over the need goes to the
    KV cache, so vLLM holds all of it. The utilization share is vLLM's startup check that it is free."""
    reserved = int(reserve * (other_bytes + kv_cache_bytes))
    share = min(reserved / total_bytes, 1.0)
    return ["--max-num-seqs", "1", "--kv-cache-memory-bytes", str(reserved - other_bytes),
            "--gpu-memory-utilization", f"{share:.4f}"]


def probe_args(kv_bytes: int, weights_bytes: int, total_bytes: int) -> list[str]:
    """vLLM arguments for a probe start with a KV cache of `kv_bytes`; the share (vLLM's startup check of free
    memory) is what the probe takes: the weights and that KV cache."""
    return ["--max-num-seqs", "1", "--kv-cache-memory-bytes", str(kv_bytes),
            "--gpu-memory-utilization", f"{min((weights_bytes + kv_bytes) / total_bytes, 1.0):.4f}"]


def needed_kv_bytes(log_text: str, probe_bytes: int) -> int | None:
    """The KV cache one request at the server's length needs, from a probe's log (see the module doc); None
    if the probe's cache was smaller than a block. vLLM's figures are rounded; rounded up here."""
    if match := _NEEDED.search(log_text):
        needed, available = (float(g) * GiB for g in match.groups())
        return int(needed + (probe_bytes - available)) + ROUNDING  # plus the block held back
    if match := _CONCURRENCY.search(log_text):  # a start that will check the block itself (kv_shortfall)
        return int(probe_bytes / float(match.group(1))) + ROUNDING
    if "No available memory for the cache blocks" in log_text:
        return None
    raise RuntimeError("vLLM's probe start ended without sizing its KV cache; end of its log:\n"
                       + "\n".join(log_text.splitlines()[-20:]))


def kv_shortfall(log_text: str) -> int | None:
    """How much more KV cache a start that stopped for lack of it needed; None if it didn't stop for that."""
    if match := _NEEDED.search(log_text):
        needed, available = (float(g) * GiB for g in match.groups())
        return int(needed - available) + ROUNDING
    return None


def startup_memory(log_text: str) -> dict:
    """What vLLM reports about its memory at startup: weights, KV cache (bytes and tokens), CUDA graphs."""
    def number(pattern):
        found = re.findall(pattern, log_text)
        return float(found[-1].replace(",", "")) if found else None

    return {"weights_gib": number(r"Model loading took ([\d.]+) GiB"),
            "kv_cache_gib": number(r"reserved ([\d.]+) GiB memory for KV Cache"),
            "kv_cache_tokens": number(r"GPU KV cache size: ([\d,]+) tokens"),
            "cuda_graphs_gib": number(r"Graph capturing finished in \d+ secs, took ([-\d.]+) GiB")}


_WRITING = threading.Lock()  # MemoryNeeds.put from several threads (models measured at once)


class MemoryNeeds:
    """Saved memory needs of vLLM servers: one JSON file, keyed by model, revision, vLLM version and length;
    per key the KV cache's bytes and, once measured, the server's other bytes (weights, working memory). Safe
    to share between processes and threads (models measured or timed at once): each update is made under a
    lock file and written whole, by renaming."""

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def key(repo: str, revision: str, vllm_version: str, max_model_len: int) -> str:
        return f"{repo}@{revision} | vllm {vllm_version} | max_model_len {max_model_len} | max_num_seqs 1"

    def get(self, key: str) -> dict:
        return self._load().get(key, {})

    def put(self, key: str, **values: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITING, open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            needs = self._load()
            needs[key] = {**needs.get(key, {}), **values}
            temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}")
            temporary.write_text(json.dumps(needs, indent=1, sort_keys=True))
            os.replace(temporary, self.path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}


def device_memory(gpu: str = "0") -> tuple[int, int, str]:
    """(total, free, source) in bytes: the GPU's own memory as nvidia-smi reports it, or, where the GPU uses
    the system's memory (e.g. Jetson) and nvidia-smi has no figure, the system's (/proc/meminfo)."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu.split(",")[0], "--query-gpu=memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        total, free = (int(x) * (1 << 20) for x in out.strip().split(","))
        return total, free, "nvidia-smi"
    except (OSError, ValueError, subprocess.SubprocessError):
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        kib = {k: int(v.split()[0]) * 1024 for k, v in info.items() if k in ("MemTotal", "MemAvailable")}
        return kib["MemTotal"], kib["MemAvailable"], "/proc/meminfo"


def group_pids(pgid: int) -> set[int]:
    """The processes of a process group (a server started in its own session, with its workers)."""
    pids = set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid:  # after the name: state, ppid, pgrp
            pids.add(int(stat.parent.name))
    return pids


def process_gpu_bytes(pids: set[int]) -> int | None:
    """GPU memory the processes hold, as nvidia-smi reports it; None where it reports none (e.g. Jetson)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
        held = [int(mib) * (1 << 20) for pid, mib in rows if int(pid) in pids]
        return sum(held) if held else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
