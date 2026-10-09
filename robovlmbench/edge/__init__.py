"""A small subset of the benchmarks for timing models on an edge device (e.g. NVIDIA Jetson AGX Thor), picked
from the full runs in outputs/ and packed with everything needed to send it: scripts/make_edge_subset.py.

What is timed is inference only: each request is sent as the full run sent it (same prompt, images, sampling
settings and seed), one at a time. The subset has to fit a time budget for the slowest model, so it is small;
it is picked to cover the full runs' workload:

- Every capability of every benchmark is a *stratum* with its own requests: RoboSpatial's 3 question types,
  InstructPart's 2 variants, RoboBench's dimensions (image and video apart), EPIC-Bench's 3 categories and
  OVDEval's 8 subsets (one and several targets apart: models answer with a box per target, and OVDEval
  questions also list descriptions of things that aren't there), and CaP-X's code-writing calls (first call of
  single- and multi-turn episodes, and multi-turn follow-ups; replayed from the runs' call logs, so no
  simulator is needed).
- Within a stratum, the requests picked are those whose mean matches the stratum's mean in the full runs, for
  every model: prompt tokens, output tokens, latency and score. Matching each model's output tokens keeps
  requests that make models answer at length; matching each model's score keeps requests where models
  disagree, in proportion. Each picked request stands for an equal share of its stratum.
- Requests a model answered until its max_tokens are left out of the subset (one can take minutes on an
  edge device), but not of the full-run totals the device's results are scaled to.

The device's results estimate each model's full benchmark on the device, per benchmark (run_bench.py): output
tokens after the first at the device's decode speed (measured on the subset), plus each full-run request's time
to its first token (its latency minus its other output tokens at the full run's decode speed) times the
device's slowdown on that time (measured on the subset's requests of that benchmark, by streaming). validate() checks this
on a simulated device, leaving out of the pick the model being estimated.

The budget is checked with a time estimate per request on the device: per model, latency = overhead + prompt
tokens / prefill speed + output tokens / decode speed, fitted on its full runs, each term slowed down by
DEVICE_SLOWDOWN.
"""

import base64
import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import math
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robovlmbench import REPO_ROOT
from robovlmbench.benchmarks import capx, epic_bench, instructpart, ovdeval, robobench, robospatial
from robovlmbench.config import MODEL_CONFIG_DIR, load_model_config
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.models.client import Client
from robovlmbench.runner import OUTPUTS_DIR

MODE = "nothink"
BENCHMARKS = {"robospatial": robospatial, "instructpart": instructpart, "robobench": robobench,
              "epic_bench": epic_bench, "ovdeval": ovdeval}
PER_STRATUM = {"robospatial": 3, "instructpart": 3, "robobench": 2, "epic_bench": 2, "ovdeval": 1, "capx": 1}
METRICS = ("prompt_tokens", "completion_tokens", "latency_s", "score")  # matched per stratum and model
# Jetson AGX Thor vs RTX PRO 6000 Blackwell (where the full runs ran): decode is memory-bound (273 vs 1792 GB/s),
# prefill compute-bound (FP4: ~1 vs ~4 PFLOPS dense), the per-request overhead CPU-bound (Arm cores)
DEVICE_SLOWDOWN = {"overhead": 3.0, "prefill": 3.9, "decode": 6.6}
PACKAGE_FILES = [Path(__file__).parent / "run_bench.py", Path(__file__).parent / "check_system.py",
                 Path(__file__).parent / "README.md",
                 Path(__file__).parent.parent / "server" / "vllm_memory.py"]  # copied into the package
CONTEXT = 8192 * 2  # tokens each request may use, prompt and answer: its max_tokens is CONTEXT minus its prompt
LENGTH_SLACK = 256  # the server's max_model_len over CONTEXT, should the device's vLLM count image tokens differently
MAX_MODEL_LEN = CONTEXT + LENGTH_SLACK
RESERVE = 1.5  # times its measured memory need, what a server started on the device reserves


@dataclass
class Request:
    """One request of a full run: a benchmark sample, or a CaP-X call."""
    id: str
    benchmark: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cut_off: bool  # answered until max_tokens
    score: float = 0.0
    error: bool = False


@dataclass
class Item:
    """A request of the subset's population, with every model's full-run record."""
    id: str
    benchmark: str
    stratum: str
    records: dict[str, Request]  # model -> its request
    source: object = None        # benchmark: the Sample; capx: model -> the logged call
    stands_for: int = 0          # requests of its stratum it stands for
    weight: float = 0.0          # their share of its benchmark's requests


@dataclass
class Subset:
    models: list[str]
    items: list[Item]
    runs: dict[str, dict[str, list[Request]]]  # model -> benchmark -> every request of the full run
    fits: dict[str, list[float]]               # model -> (overhead s, s per prompt token, s per output token)
    validation: dict = field(default_factory=dict)


def models() -> list[str]:
    """The models with a full no-think run of every benchmark."""
    names = []
    for path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        name = f"{path.stem}_{MODE}"
        if all((OUTPUTS_DIR / name / "vllm" / b / "scores.json").exists() for b in [*BENCHMARKS, "capx"]):
            names.append(name)
    return names


def load(names: list[str]) -> tuple[list[Item], dict[str, dict[str, list[Request]]]]:
    """The subset's population (requests every model answered without error or cut-off) and the full runs."""
    runs = {m: {} for m in names}
    items = []
    for bench, module in BENCHMARKS.items():
        samples = {s.id: s for s in module.load_samples()}
        for m in names:
            runs[m][bench] = _benchmark_run(m, bench)
        by_id = {m: {r.id: r for r in runs[m][bench]} for m in names}
        strata = _strata(bench, samples, names[0])
        for sid, sample in samples.items():
            rid = f"{bench}:{sid}"
            records = {m: by_id[m].get(rid) for m in names}
            if all(r and not r.error and not r.cut_off for r in records.values()):
                items.append(Item(rid, bench, strata[sid], records, sample))
    calls = {}
    for m in names:
        runs[m]["capx"], calls[m] = _capx_run(m)
    for rid in sorted(set.intersection(*(set(c) for c in calls.values()))):
        records = {m: calls[m][rid][0] for m in names}
        if any(r.error or r.cut_off for r in records.values()):
            continue
        episode, _, call = rid.removeprefix("capx:").rpartition("/")
        config = episode.rpartition("/")[0]
        suite = "libero" if config.startswith("libero") else "robosuite"
        stratum = f"{suite}/{config.rsplit('/', 1)[-1]}/{'first call' if call == 'call1' else 'follow-up call'}"
        items.append(Item(rid, "capx", stratum, records, {m: calls[m][rid][1] for m in names}))
    return items, runs


def _benchmark_run(model: str, bench: str) -> list[Request]:
    out = OUTPUTS_DIR / model / "vllm" / bench
    scores = {s["id"]: s for s in json.loads((out / "scores.json").read_text())["per_sample"]}
    run = []
    for line in (out / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        run.append(Request(f"{bench}:{r['id']}", bench, r["prompt_tokens"] or 0, r["completion_tokens"] or 0,
                           r["latency_s"], r["finish_reason"] == "length", _score(bench, scores.get(r["id"], {})),
                           bool(r["error"])))
    return run


def _score(bench: str, s: dict) -> float:
    """The sample's result as a number in [0, 1]."""
    if bench == "epic_bench":
        return float(s.get("score", 0.0))
    if bench == "ovdeval":  # share of the targets found
        return s.get("found", 0) / s["n_gt"] if s.get("n_gt") else float(not s.get("boxes"))
    return float(bool(s.get("correct")))


def _strata(bench: str, samples: dict, model: str) -> dict[str, str]:
    if bench == "robospatial":
        return {i: s.category for i, s in samples.items()}
    if bench == "instructpart":
        return {i: s.variant for i, s in samples.items()}
    if bench == "robobench":
        return {i: f"{s.dimension}/{s.input_type}" for i, s in samples.items()}

    def targets(n):
        return "several targets" if n > 1 else "one target"

    if bench == "epic_bench":
        def count(s):
            return json.loads(s.gt_path.read_text())["text_annotation"].get("count_target_instance", [1])[0] or 1
        return {i: f"{s.category}/{targets(count(s))}" if s.category == "TargetLocalization" else s.category
                for i, s in samples.items()}
    if bench == "ovdeval":
        scores = json.loads((OUTPUTS_DIR / model / "vllm" / bench / "scores.json").read_text())["per_sample"]
        n_gt = {s["id"]: s.get("n_gt", 0) for s in scores}
        return {i: f"{s.subset}/{targets(n_gt.get(i, 0))}" for i, s in samples.items()}
    raise ValueError(bench)


def _capx_run(model: str) -> tuple[list[Request], dict[str, tuple[Request, dict]]]:
    """Every CaP-X call of the full run; and the subset's candidates: per episode its first call, and for
    multi-turn episodes its second (a follow-up, with the code's output in the prompt), with the logged call."""
    run, candidates = [], {}
    for e in capx.load_episodes(OUTPUTS_DIR / model / "vllm" / "capx").values():
        for c in e["calls"]:
            run.append(Request("", "capx", c["prompt_tokens"], c["completion_tokens"], c["latency_s"],
                               c["finish_reason"] == "length", error=bool(c["error"])))
        first_attempt = sorted((c for c in e["calls"] if c["attempt"] == 0), key=lambda c: c["time"])
        for k, c in enumerate(first_attempt[:2 if "multiturn" in e["config"] else 1]):
            rid = f"capx:{e['config']}/{e['trial']}/call{k + 1}"
            candidates[rid] = (Request(rid, "capx", c["prompt_tokens"], c["completion_tokens"], c["latency_s"],
                                       c["finish_reason"] == "length", float(e["task_completed"]), bool(c["error"])), c)
    return run, candidates


def fit(requests: list[Request]) -> list[float]:
    """Least-squares latency = a + b * prompt_tokens + c * completion_tokens (each term at least 0)."""
    usable = [r for r in requests if not r.error]
    X = np.array([(1.0, r.prompt_tokens, r.completion_tokens) for r in usable])
    coef, *_ = np.linalg.lstsq(X, np.array([r.latency_s for r in usable]), rcond=None)
    return [max(float(c), 0.0) for c in coef]


def device_seconds(coef: list[float], r: Request) -> float:
    """Estimated time of a request on the device."""
    s = DEVICE_SLOWDOWN
    return coef[0] * s["overhead"] + coef[1] * s["prefill"] * r.prompt_tokens + coef[2] * s["decode"] * r.completion_tokens


def input_seconds(r: Request, decode_s_per_token: float) -> float:
    """A request's time to its first output token: its latency minus its other output tokens at the decode
    speed (as the device measures it when streaming)."""
    return max(r.latency_s - max(r.completion_tokens - 1, 0) * decode_s_per_token, 0.005)


def _search(X: np.ndarray, k: int, rng: np.random.Generator, tries: int = 4000) -> list[int]:
    """k rows of X whose mean is closest to the mean of all rows (each column in units of its spread): all
    k-subsets if there are few, else random ones, then improved one row at a time."""
    n = len(X)
    if n <= k:
        return list(range(n))
    std = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)

    def cost(rows) -> float:
        return float(np.square(Z[list(rows)].mean(axis=0)).sum())

    if math.comb(n, k) <= tries:
        return list(min(itertools.combinations(range(n), k), key=cost))
    best = min((tuple(int(i) for i in rng.choice(n, k, replace=False)) for _ in range(tries)), key=cost)
    for _ in range(3):  # swap each row for the best of some others
        for slot in range(k):
            others = [best[:slot] + (int(c),) + best[slot + 1:] for c in rng.choice(n, min(n, 500), replace=False)
                      if c not in best]
            best = min([best, *others], key=cost)
    return list(best)


def pick(items: list[Item], names: list[str], seed: int = 0) -> list[Item]:
    """PER_STRATUM requests per stratum whose mean matches the stratum's (METRICS, every model in `names`)."""
    rng = np.random.default_rng(seed)
    by_bench: dict[str, dict[str, list[Item]]] = {}
    for it in items:
        by_bench.setdefault(it.benchmark, {}).setdefault(it.stratum, []).append(it)
    picked = []
    for bench, strata in by_bench.items():
        total = sum(len(group) for group in strata.values())
        for stratum in sorted(strata):
            group = strata[stratum]
            X = np.array([[getattr(it.records[m], q) for m in names for q in METRICS] for it in group], dtype=float)
            rows = _search(X, PER_STRATUM[bench], rng)
            for row in rows:
                it = group[row]
                share = len(group) / len(rows)
                picked.append(Item(it.id, it.benchmark, it.stratum, it.records, it.source, round(share), share / total))
    return picked


def estimate(items: list[Item], model: str, device: dict[str, tuple[float, float]], run: dict[str, list[Request]],
             decode_s_per_token: float) -> dict[str, float]:
    """The full run's mean time per request on the device, per benchmark, from the device's times on the
    subset (`device`: item id -> (time to first token, time after it)), as run_bench.py estimates it."""
    timed = [it for it in items if it.id in device]
    tokens = sum(max(it.records[model].completion_tokens - 1, 0) for it in timed)
    decode = tokens / sum(device[it.id][1] for it in timed)  # device's output tokens per second
    means = {}
    for bench, requests in run.items():
        sub = [it for it in timed if it.benchmark == bench]
        slowdown = (sum(it.weight * device[it.id][0] for it in sub)
                    / sum(it.weight * input_seconds(it.records[model], decode_s_per_token) for it in sub))
        usable = [r for r in requests if not r.error]
        means[bench] = sum(input_seconds(r, decode_s_per_token) * slowdown + max(r.completion_tokens - 1, 0) / decode
                           for r in usable) / len(usable)
    return means


def validate(items: list[Item], runs: dict, fits: dict, names: list[str]) -> dict:
    """The estimate's error on a simulated device, leaving the estimated model out of the pick: each full-run
    request's latency split by its benchmark's own fit into input and output time, slowed down by
    DEVICE_SLOWDOWN (the fit's residual goes with the input time). Returns model -> benchmark -> relative error."""
    s = DEVICE_SLOWDOWN
    errors = {}
    for m in names:
        picked = pick(items, [x for x in names if x != m])
        device, truth = {}, {}
        for bench, requests in runs[m].items():
            usable = [r for r in requests if not r.error]
            a, b, c = fit(usable)

            def simulate(r):  # (time to the first token, time after it)
                residual = r.latency_s - (a + b * r.prompt_tokens + c * r.completion_tokens)
                first = a * s["overhead"] + (b * r.prompt_tokens + residual) * s["prefill"] + c * min(r.completion_tokens, 1) * s["decode"]
                return max(first, 0.01), c * max(r.completion_tokens - 1, 0) * s["decode"]

            truth[bench] = sum(sum(simulate(r)) for r in usable) / len(usable)
            for it in picked:
                if it.benchmark == bench:
                    device[it.id] = simulate(it.records[m])
        means = estimate(picked, m, device, runs[m], fits[m][2])
        errors[m] = {bench: round(abs(means[bench] - truth[bench]) / truth[bench], 4) for bench in truth}
    return errors


def select(names: list[str], budget_s: float) -> Subset:
    items, runs = load(names)
    fits = {m: fit([r for requests in runs[m].values() for r in requests]) for m in names}
    picked = pick(items, names)
    totals = {m: sum(device_seconds(fits[m], it.records[m]) for it in picked) for m in names}
    over = {m: round(t) for m, t in totals.items() if t > budget_s}
    if over:
        raise SystemExit(f"the subset takes longer than {budget_s:.0f}s on the device for {over}; "
                         "lower PER_STRATUM or raise --budget-s")
    return Subset(names, picked, runs, fits, validate(items, runs, fits, names))


def measure_memory(subset: Subset, gpus: list[str], port: int = 8600, heaviest: int = 3) -> dict[str, dict]:
    """Each model's memory need at MAX_MODEL_LEN on this hardware, saved per GPU type (robovlmbench/server/
    vllm_memory.py); models not measured yet are started here once (on `gpus`, one model per GPU at a time) and
    sent their `heaviest` requests by prompt and by answer length (working memory grows with both), the most
    memory the server holds meanwhile being its need. Returns model -> {"kv_cache_bytes", "other_bytes"}."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from robovlmbench.server import start, vllm
    from robovlmbench.server.vllm_memory import group_pids, process_gpu_bytes

    free_gpus = queue.Queue()
    for k, gpu in enumerate(gpus):
        free_gpus.put((gpu, port + k))

    def key_of(m):
        config = dataclasses.replace(load_model_config(m), max_seq_len=MAX_MODEL_LEN)
        return config, vllm._needs().key(config.repo, config.revision, importlib.metadata.version("vllm"), MAX_MODEL_LEN)

    def measure(m):
        config, key = key_of(m)
        gpu, gpu_port = free_gpus.get()
        print(f"measuring {m}'s memory at {MAX_MODEL_LEN} tokens on GPU {gpu}: its server, sent its heaviest requests")
        by_prompt = sorted(subset.items, key=lambda it: -it.records[m].prompt_tokens)[:heaviest]
        by_answer = sorted(subset.items, key=lambda it: -it.records[m].completion_tokens)[:heaviest]
        client = Client("http://unused", m, config.request_settings())
        try:
            with start(config, "vllm", gpus=gpu, port=gpu_port) as server:
                pids, peak, done = group_pids(server.process.pid), [0], threading.Event()

                def sample():
                    while not done.wait(0.2):
                        peak[0] = max(peak[0], process_gpu_bytes(pids) or 0)

                sampler = threading.Thread(target=sample, daemon=True)
                sampler.start()
                api = OpenAI(base_url=server.url, api_key="none")
                for it in {it.id: it for it in by_prompt + by_answer}.values():
                    messages, sampling, extra_body = _request(it, m, config, client)
                    api.chat.completions.create(model=server.model, messages=messages, extra_body=extra_body, **sampling)
                done.set()
                sampler.join()
            vllm._needs().put(key, other_bytes=peak[0] - vllm._needs().get(key)["kv_cache_bytes"], peak_measured=1)
        finally:
            free_gpus.put((gpu, gpu_port))

    todo = [m for m in subset.models if not vllm._needs().get(key_of(m)[1]).get("peak_measured")]
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(measure, todo))
    return {m: {k: v for k, v in vllm._needs().get(key_of(m)[1]).items() if k != "peak_measured"}
            for m in subset.models}


def _request(it: Item, model: str, config, client: Client) -> tuple[list[dict], dict, dict]:
    """(messages, sampling, extra_body) of an item's request for `model`, as its full run sent it but with
    max_tokens filling CONTEXT."""
    max_tokens = CONTEXT - it.records[model].prompt_tokens
    if it.benchmark == "capx":  # the call as the run logged it
        call = it.source[model]["request"]
        return call["messages"], {**call["sampling"], "max_tokens": max_tokens}, call["extra_body"]
    request = BENCHMARKS[it.benchmark].build_request(it.source, POINT_FORMATS[config.point_format])
    return client.messages(request), {**client.sampling, "max_tokens": max_tokens}, client.extra_body or {}


def pack(subset: Subset, out_dir: Path, budget_s: float, memory: dict[str, dict]) -> Path:
    """Write the package: README.md, run_bench.py, manifest.json, requests/<model>.jsonl, media/. Returns
    the .tar.gz written next to it."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "requests").mkdir(parents=True)
    (out_dir / "media").mkdir()
    for path in PACKAGE_FILES:
        shutil.copy(path, out_dir / path.name)

    model_info = {}
    for m in subset.models:
        config = load_model_config(m)
        client = Client("http://unused", m, config.request_settings())
        with open(out_dir / "requests" / f"{m}.jsonl", "w") as f:
            for it in subset.items:
                messages, sampling, extra_body = _request(it, m, config, client)
                f.write(json.dumps({"id": it.id, "benchmark": it.benchmark, "stratum": it.stratum,
                                    "messages": _store_media(messages, out_dir / "media"),
                                    "sampling": sampling, "extra_body": extra_body or {}}) + "\n")
        decode_s = subset.fits[m][2]
        full_run = {}
        for bench, requests in subset.runs[m].items():
            usable = [r for r in requests if not r.error]
            full_run[bench] = {"requests": len(usable), "latency_s": round(sum(r.latency_s for r in usable), 3),
                               "first_token_s": round(sum(input_seconds(r, decode_s) for r in usable), 3),
                               "output_tokens_after_first": sum(max(r.completion_tokens - 1, 0) for r in usable),
                               "cut_off": sum(r.cut_off for r in usable)}
        need = memory[m]["kv_cache_bytes"] + memory[m]["other_bytes"]
        model_info[m] = {
            "repo": config.repo, "revision": config.revision,
            "max_model_len": MAX_MODEL_LEN,
            "vllm_args": config.engine_args("vllm"),
            "memory_measured": {**memory[m], "hardware": _full_run_hardware(m)},  # the device's own replaces it
            "memory_reserve": RESERVE,
            "memory_reserved_bytes": int(RESERVE * need),
            "vllm_version_of_the_full_runs": _full_run_engine(m),
            "weights_bytes": sum(f.stat().st_size for f in config.weights_path().glob("*.safetensors")),
            "estimated_device_s": round(sum(device_seconds(subset.fits[m], it.records[m]) for it in subset.items)),
            "full_run_decode_s_per_token": decode_s,
            "full_run": full_run,
        }

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "source": {"commit": _git_commit(), "hardware": _full_run_hardware(subset.models[0]), "mode": MODE},
        "budget_s": budget_s, "device_slowdown_assumed": DEVICE_SLOWDOWN, "per_stratum": PER_STRATUM,
        "models": model_info,
        "validation_relative_error_simulated_device": subset.validation,
        "items": [{"id": it.id, "benchmark": it.benchmark, "stratum": it.stratum, "weight": round(it.weight, 6),
                   "stands_for": it.stands_for,
                   "full_run": {m: {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                                    "latency_s": r.latency_s, "score": r.score}
                                for m, r in it.records.items()}} for it in subset.items],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    archive = out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return archive


def _store_media(messages: list[dict], media_dir: Path) -> list[dict]:
    """The messages with each data-URL image saved once under media/ and referred to by its path."""
    stored = json.loads(json.dumps(messages))
    for message in stored:
        if not isinstance(message["content"], list):
            continue
        for part in message["content"]:
            if part.get("type") != "image_url":
                continue
            part.pop("uuid", None)
            header, data = part["image_url"]["url"].split(",", 1)
            raw = base64.b64decode(data)
            name = f"{hashlib.sha256(raw).hexdigest()[:20]}.{'png' if 'png' in header else 'jpg'}"
            if not (media_dir / name).exists():
                (media_dir / name).write_bytes(raw)
            part["image_url"]["url"] = f"media/{name}"
    return stored


def _git_commit() -> str:
    run = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return run.stdout.strip() or "unknown"


def _full_run_engine(model: str) -> str:
    return json.loads((OUTPUTS_DIR / model / "vllm" / "robospatial" / "run.json").read_text())["engine"]


def _full_run_hardware(model: str) -> str:
    return json.loads((OUTPUTS_DIR / model / "vllm" / "robospatial" / "run.json").read_text())["hardware"]
