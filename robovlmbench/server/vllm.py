"""vLLM: serve a model with `vllm serve` from the project env, with the memory it needs (vllm_memory.py)."""

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Callable

from robovlmbench import REPO_ROOT
from robovlmbench.config import ModelConfig
from robovlmbench.server.vllm_memory import (PROBE_DONE, PROBE_KV_BYTES, MemoryNeeds, device_memory, group_pids,
                                             kv_shortfall, needed_kv_bytes, probe_args, process_gpu_bytes, server_args,
                                             startup_memory)


def command(config: ModelConfig, weights: Path, port: int) -> list[str]:
    vllm = Path(sys.executable).parent / "vllm"  # the env's own vllm, whatever PATH says
    return [
        str(vllm), "serve", str(weights),
        "--served-model-name", config.name,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(config.max_seq_len),
        *config.engine_args("vllm"),
    ]


def version() -> str:
    return f"vllm {importlib.metadata.version('vllm')}"


def memory_args(config: ModelConfig, weights: Path, gpus: str, probe: Callable[[list[str]], str]) -> tuple[list[str], str]:
    """The memory arguments for serving `config` (see vllm_memory.py), and the key its needs are saved under.
    `probe(args)` runs the server with `args` added until it has sized its KV cache (or stopped), stops it and
    returns its log: the first start of a model at its length, to learn the KV cache it needs."""
    needs = _needs()
    key = needs.key(config.repo, config.revision, importlib.metadata.version("vllm"), config.max_seq_len)
    total, _, _ = device_memory(gpus)
    weights_bytes = sum(f.stat().st_size for f in weights.glob("*.safetensors"))
    if "kv_cache_bytes" not in needs.get(key):
        for probe_bytes in PROBE_KV_BYTES:
            if kv := needed_kv_bytes(probe(probe_args(probe_bytes, weights_bytes, total)), probe_bytes):
                break
        else:
            raise RuntimeError(f"vLLM found no KV cache block in {PROBE_KV_BYTES[-1]} bytes")
        needs.put(key, kv_cache_bytes=kv)
    saved = needs.get(key)
    other = saved.get("other_bytes") or weights_bytes
    return server_args(saved["kv_cache_bytes"], other, total), key


def add_shortfall(key: str, log_text: str) -> bool:
    """After a start that stopped: if it stopped for lack of KV cache, save the larger need; whether it did."""
    if (short := kv_shortfall(log_text)) is None:
        return False
    needs = _needs()
    needs.put(key, kv_cache_bytes=needs.get(key)["kv_cache_bytes"] + short)
    return True


def record_memory(key: str, pgid: int, log_text: str) -> dict:
    """The running server's memory: what vLLM reports, and what its processes hold (saved for the next start's
    free-memory check)."""
    memory = startup_memory(log_text)
    held = process_gpu_bytes(group_pids(pgid))
    if held:
        needs = _needs()
        needs.put(key, other_bytes=held - needs.get(key)["kv_cache_bytes"])
        memory["server_gib"] = round(held / (1 << 30), 2)
    return memory


def _needs() -> MemoryNeeds:
    return MemoryNeeds(REPO_ROOT / "cache" / os.environ.get("RVB_GPU", "gpu") / "vllm_memory_needs.json")
