#!/usr/bin/env python3
"""Time a model on this subset: send its requests (requests/<model>.jsonl) one at a time to an OpenAI-style
server with streaming, and record per request the time to the first token, the total time and the token
counts, and the memory the server takes. Python 3.8+, standard library only.

    python3 run_bench.py --model qwen3.8-27b-nvfp4_nothink
    python3 run_bench.py --check            # only check that every request and image file is in place
    python3 run_bench.py --check-system     # only check the device (check_system.py; reads only)

The server: if one answers at --base-url serving the model (--served-model-name, default --model, or the
model's repo), it is used and left running. Otherwise the system's `vllm` (on PATH) is started on that port with
the memory the model needs (vllm_memory.py), and stopped at the end; its log is results/<model>/server.log.
Results: results/<model>/requests.jsonl (one line per request; re-running sends only the missing ones),
results/<model>/memory.json and results/<model>/summary.json (also printed).
"""

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # write nothing outside results/
sys.path.insert(0, str(HERE))
import vllm_memory  # noqa: E402  (shipped next to this file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="a file name of requests/ without .jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model-name")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    parser.add_argument("--no-fresh-images", action="store_true",
                        help="don't give images a new id per request (vLLM's per-image uuid, which keeps it from "
                             "reusing an earlier request's work on the same image; leave it on for vLLM)")
    parser.add_argument("--check", action="store_true", help="only check the package's files")
    parser.add_argument("--check-system", action="store_true",
                        help="only check the device: vLLM, GPU, memory, disk (reads only; prints what to fix)")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    if args.check:
        return check(manifest)
    if args.check_system:
        import check_system
        sys.exit(0 if check_system.check(manifest, HERE, args.base_url) else 1)
    if not args.model:
        parser.error("--model is needed (one of: " + ", ".join(manifest["models"]) + ")")
    run(args, manifest)


def check(manifest: dict) -> None:
    ids = [item["id"] for item in manifest["items"]]
    for model in manifest["models"]:
        requests = load_requests(model)
        assert [r["id"] for r in requests] == ids, f"{model}: requests don't match the manifest"
        for r in requests:
            for path in image_paths(r["messages"]):
                assert (HERE / path).is_file(), f"{model}: missing {path}"
    for name in ("vllm_memory.py", "check_system.py"):
        assert (HERE / name).is_file(), f"missing {name}"
    print(f"OK: {len(ids)} requests for each of {len(manifest['models'])} models, all image files present")


def load_requests(model: str) -> list:
    return [json.loads(line) for line in (HERE / "requests" / f"{model}.jsonl").read_text().splitlines()]


def image_paths(messages: list) -> list:
    return [part["image_url"]["url"] for m in messages if isinstance(m["content"], list)
            for part in m["content"] if part.get("type") == "image_url"]


def run(args, manifest: dict) -> None:
    requests = load_requests(args.model)
    out_dir = HERE / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "requests.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r["error"]:
                done[r["id"]] = r
    todo = [r for r in requests if r["id"] not in done]
    print(f"{args.model}: {len(requests)} requests, {len(done)} already timed, {len(todo)} to send")
    if todo:
        with Server(args, manifest["models"][args.model], out_dir) as server:
            warm_up = send(args, server.name, requests[0])  # not saved: a new server's first request can be slower
            if warm_up["error"]:
                sys.exit(f"the server didn't answer: {warm_up['error']}")
            started = time.monotonic()
            with MemoryWatch(server) as watch, open(path, "a") as f:
                for k, request in enumerate(todo, 1):
                    result = send(args, server.name, request)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    if result["error"]:
                        print(f"  {request['id']}: {result['error']}")
                    else:
                        done[result["id"]] = result
                    print(f"  {k}/{len(todo)} {request['id']}: {result['latency_s']:.2f}s, "
                          f"{result['completion_tokens']} tokens out ({time.monotonic() - started:.0f}s so far)")
            report = watch.report()
            (out_dir / "memory.json").write_text(json.dumps(report, indent=1))
            server.record_peak(report)
    summary = summarize(manifest, args.model, [done[r["id"]] for r in requests if r["id"] in done], len(requests))
    if (out_dir / "memory.json").exists():
        summary["memory"] = json.loads((out_dir / "memory.json").read_text())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


class Server:
    """The model's server: one already running at --base-url, or the system's vLLM started (and stopped) here."""

    def __init__(self, args, info: dict, out_dir: Path):
        self.args, self.info, self.out_dir = args, info, out_dir
        self.url = args.base_url.rstrip("/")
        self.name = args.served_model_name or args.model
        self.process = None
        self.gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        self.memory_before = vllm_memory.device_memory(self.gpu)  # (total, free, source)
        self.startup = {}

    def __enter__(self):
        served = self._served_models()
        if served is not None:
            match = [m for m in served if m in (self.name, self.info["repo"])]
            if not match:
                sys.exit(f"a server at {self.url} serves {served}, not {self.name}: stop it, or pass another "
                         "--base-url (a free port, where this script starts one)")
            self.name = match[0]
            print(f"using the server running at {self.url} ({self.name}); it is left running")
            return self
        try:
            self._start()
        except BaseException:  # a server started here is stopped, whatever stopped the start
            if self.process is not None:
                _stop(self.process)
            raise
        return self

    def __exit__(self, *exc):
        if self.process is not None:
            print("stopping the server started here")
            _stop(self.process)

    def _served_models(self):
        try:
            with urllib.request.urlopen(f"{self.url}/models", timeout=10) as r:
                return [m["id"] for m in json.load(r)["data"]]
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _start(self) -> None:
        vllm = shutil.which("vllm")
        if vllm is None:
            sys.exit("no server answers at " + self.url + " and there is no `vllm` on PATH. Install vLLM "
                     f"({self.info['vllm_version_of_the_full_runs']} for the full runs' version) or start an "
                     "OpenAI-compatible server for " + self.info["repo"] + ", then run this again.")
        port = urllib.parse.urlparse(self.url).port or 8000
        version = subprocess.run([vllm, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        command = [vllm, "serve", self.info["repo"], "--revision", self.info["revision"], "--served-model-name",
                   self.name, "--host", "127.0.0.1", "--port", str(port), "--max-model-len",
                   str(self.info["max_model_len"]), *self.info["vllm_args"]]
        env = {**os.environ, "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        # The memory the model needs, as this device measured it on an earlier start; until then as measured on
        # the full runs' hardware (manifest). memory_reserve times that is reserved (vllm_memory.server_args).
        needs = vllm_memory.MemoryNeeds(HERE / "results" / "vllm_memory_needs.json")
        key = needs.key(self.info["repo"], self.info["revision"], version, self.info["max_model_len"])
        total, free, _ = self.memory_before
        log = self.out_dir / "server.log"
        for attempt in range(3):  # again with more KV cache if a start finds it short
            need = {**self.info["memory_measured"], **needs.get(key)}
            memory = vllm_memory.server_args(need["kv_cache_bytes"], need["other_bytes"], total,
                                             self.info["memory_reserve"])
            print(f"starting {self.info['repo']} with vLLM {version}: {' '.join(memory)} (log: {log}; the first "
                  "start downloads the weights)")
            self.process = _launch(command + memory, env, log)
            if _wait_until_ready(self.process, self.url, log):
                break
            _stop(self.process)
            self.process = None
            short = vllm_memory.kv_shortfall(log.read_text(errors="replace"))
            if short is None or attempt == 2:
                sys.exit(f"the server didn't start; end of {log}:\n" + "\n".join(log.read_text(errors="replace").splitlines()[-20:]))
            needs.put(key, kv_cache_bytes=need["kv_cache_bytes"] + short)
        self.startup = vllm_memory.startup_memory(log.read_text(errors="replace"))
        # what the server holds: its processes' GPU memory, or where that isn't reported (memory shared with the
        # system), the drop in free memory since before it started
        held = (vllm_memory.process_gpu_bytes(vllm_memory.group_pids(self.process.pid))
                or free - vllm_memory.device_memory(self.gpu)[1])
        self.kv_given = int(memory[memory.index("--kv-cache-memory-bytes") + 1])
        self.needs, self.key = needs, key
        needs.put(key, kv_cache_bytes=need["kv_cache_bytes"], other_bytes=held - self.kv_given)

    def record_peak(self, report: dict) -> None:
        """Save the most memory the server held during the timing as its need (working memory grows with the
        requests), for the next start's reservation."""
        if self.process is not None and report.get("server_peak_gib"):
            peak = int(report["server_peak_gib"] * (1 << 30))
            self.needs.put(self.key, other_bytes=max(self.needs.get(self.key).get("other_bytes", 0), peak - self.kv_given))


class MemoryWatch:
    """The device's free memory during the speed test, sampled twice a second; with what it was before the
    server started (if started here) and the server's own memory, a report of the server's footprint."""

    def __init__(self, server: Server, interval: float = 0.5):
        self.server, self.interval = server, interval
        self.lowest_free = None
        self._stop = threading.Event()

    def __enter__(self):
        self.loaded = vllm_memory.device_memory(self.server.gpu)
        self.lowest_free = self.loaded[1]
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.lowest_free = min(self.lowest_free, vllm_memory.device_memory(self.server.gpu)[1])

    def report(self) -> dict:
        gib = 1 << 30
        total, free_before, source = self.server.memory_before
        started_here = self.server.process is not None
        report = {"source": source, "device_total_gib": round(total / gib, 2),
                  "device_free_before_gib": round(free_before / gib, 2),
                  "device_free_loaded_gib": round(self.loaded[1] / gib, 2),
                  "device_free_lowest_gib": round(self.lowest_free / gib, 2),
                  "server_started_here": started_here}
        if started_here:  # what the server took: free memory before it minus during the test
            report["server_loaded_gib"] = round((free_before - self.loaded[1]) / gib, 2)
            report["server_peak_gib"] = round((free_before - self.lowest_free) / gib, 2)
            held = vllm_memory.process_gpu_bytes(vllm_memory.group_pids(self.server.process.pid))
            if held:
                report["server_processes_gib"] = round(held / gib, 2)
            report["vllm"] = self.server.startup
        return report


def _launch(command: list, env: dict, log: Path) -> subprocess.Popen:
    with open(log, "w") as f:
        f.write(" ".join(command) + "\n\n")
        f.flush()
        return subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def _stop(process: subprocess.Popen) -> None:
    """Stop a server started here, with every process it started."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _wait_until_ready(process: subprocess.Popen, url: str, log: Path, idle_s: float = 900) -> bool:
    """Whether the server answers; False if it exits, or its log stops growing for `idle_s` (a first start
    downloading the weights takes as long as the download)."""
    health = url[:-3] + "/health" if url.endswith("/v1") else url + "/health"
    size, changed = -1, time.monotonic()
    while process.poll() is None:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if log.stat().st_size != size:
            size, changed = log.stat().st_size, time.monotonic()
        elif time.monotonic() - changed > idle_s:
            return False
        time.sleep(5)
    return False


def send(args, served: str, request: dict) -> dict:
    """One streamed request: time to the first token (answer or reasoning), total time, token counts."""
    body = {"model": served, "messages": materialize(request["messages"], not args.no_fresh_images),
            **request["sampling"], **request["extra_body"], "stream": True, "stream_options": {"include_usage": True}}
    http = urllib.request.Request(f"{args.base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"})
    result = {"id": request["id"], "benchmark": request["benchmark"], "stratum": request["stratum"],
              "ttft_s": None, "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
              "finish_reason": None, "error": None}
    start = time.monotonic()
    try:
        with urllib.request.urlopen(http, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if text and result["ttft_s"] is None:
                        result["ttft_s"] = time.monotonic() - start
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
                if chunk.get("usage"):
                    result["prompt_tokens"] = chunk["usage"].get("prompt_tokens")
                    result["completion_tokens"] = chunk["usage"].get("completion_tokens")
    except (urllib.error.URLError, OSError, ValueError) as e:
        detail = e.read().decode(errors="replace")[:500] if isinstance(e, urllib.error.HTTPError) else ""
        result["error"] = f"{type(e).__name__}: {e} {detail}".strip()
    result["latency_s"] = time.monotonic() - start
    return result


def materialize(messages: list, fresh_images: bool) -> list:
    """The messages with each image file inlined as a data URL."""
    messages = json.loads(json.dumps(messages))
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for part in message["content"]:
            if part.get("type") == "image_url":
                path = HERE / part["image_url"]["url"]
                mime = "image/png" if path.suffix == ".png" else "image/jpeg"
                part["image_url"]["url"] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
                if fresh_images:
                    part["uuid"] = uuid.uuid4().hex
    return messages


def summarize(manifest: dict, model: str, results: list, total: int) -> dict:
    """What was measured on the subset, and from it the estimated mean time per request of each benchmark's
    full run on this device: output tokens after the first at the decode speed measured here, plus the full
    run's time to the first token (its latency minus its other output tokens at the full run's decode speed)
    times this device's slowdown on that time, measured here per benchmark."""
    items = {item["id"]: item for item in manifest["items"]}
    info = manifest["models"][model]
    full_decode = info["full_run_decode_s_per_token"]

    def full(r):
        return items[r["id"]]["full_run"][model]

    def first_token_s(record):
        return max(record["latency_s"] - max(record["completion_tokens"] - 1, 0) * full_decode, 0.005)

    timed = [r for r in results if r["ttft_s"] is not None and r["completion_tokens"]]
    decode_time = sum(r["latency_s"] - r["ttft_s"] for r in timed if r["completion_tokens"] > 1)
    decode_tok_s = sum(r["completion_tokens"] - 1 for r in timed if r["completion_tokens"] > 1) / decode_time \
        if decode_time > 0 else None
    benches = {}
    for r in timed:
        benches.setdefault(r["benchmark"], []).append(r)

    per_benchmark = {}
    for bench, rows in sorted(benches.items()):
        weight = {r["id"]: items[r["id"]]["weight"] for r in rows}
        total_weight = sum(weight.values())

        def mean(value):
            return round(sum(weight[r["id"]] * value(r) for r in rows) / total_weight, 4)

        slowdown = sum(weight[r["id"]] * r["ttft_s"] for r in rows) / sum(weight[r["id"]] * first_token_s(full(r)) for r in rows)
        run = info["full_run"][bench]
        estimate = ((slowdown * run["first_token_s"] + run["output_tokens_after_first"] / decode_tok_s) / run["requests"]
                    if decode_tok_s else None)
        full_mean = run["latency_s"] / run["requests"]
        per_benchmark[bench] = {
            "subset": {"requests": len(rows), "latency_s": mean(lambda r: r["latency_s"]),
                       "ttft_s": mean(lambda r: r["ttft_s"]), "prompt_tokens": mean(lambda r: r["prompt_tokens"] or 0),
                       "completion_tokens": mean(lambda r: r["completion_tokens"]),
                       "full_run_completion_tokens": mean(lambda r: full(r)["completion_tokens"])},
            "first_token_slowdown_vs_full_run": round(slowdown, 3),
            "full_benchmark_estimate": {"requests": run["requests"], "mean_s": round(estimate, 4) if estimate else None,
                                        "total_h": round(estimate * run["requests"] / 3600, 2) if estimate else None,
                                        "full_run_mean_s": round(full_mean, 4),
                                        "slowdown_vs_full_run": round(estimate / full_mean, 2) if estimate else None},
        }
    return {"model": model, "requests_timed": len(results), "requests_in_subset": total,
            "cut_off": sum(r["finish_reason"] == "length" for r in results),
            "subset_total_s": round(sum(r["latency_s"] for r in results), 1),
            "decode_tok_s": round(decode_tok_s, 1) if decode_tok_s else None,
            "full_run_hardware": manifest["source"]["hardware"],
            "per_benchmark": per_benchmark}


if __name__ == "__main__":
    main()
