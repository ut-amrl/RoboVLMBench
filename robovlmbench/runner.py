"""Run one model on one benchmark: send every sample, save each response as it arrives, score.

Outputs go to outputs/<model>/<engine>/<bench>/:
    run.json          what produced the responses (weights, engine version, request settings, hardware)
    responses.jsonl   one line per response, appended as they arrive
    scores.json       the benchmark's scores over the samples in this run, and time per question

Questions are sent one at a time, as a robot would ask them, so each response's `latency_s` is the
time to answer that question alone on the recorded hardware. Every image is sent as new: servers
cache work on images they have seen, and benchmarks reuse images across questions, while a robot's
next question comes with a new camera frame. Nothing else should use the server's GPUs during a run.

Re-running resumes: samples that already have a successful response are not sent again. Responses
from different settings or hardware are never mixed: if run.json differs, the run stops.
"""

import contextlib
import json
import statistics
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from robovlmbench import REPO_ROOT
from robovlmbench.benchmarks import capx, epic_bench, instructpart, ovdeval, robobench, robospatial
from robovlmbench.config import load_model_config
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.models.client import Client, Request, Response
from robovlmbench.server import ENGINES, start

OUTPUTS_DIR = REPO_ROOT / "outputs"
BENCHMARKS = {"robospatial": robospatial, "instructpart": instructpart, "robobench": robobench,
              "epic_bench": epic_bench, "ovdeval": ovdeval, "capx": capx}


def run(model: str, bench: str, engine: str = "vllm", *, gpus: str = "0", port: int = 8000,
        limit: int | None = None, server_url: str | None = None, shard: tuple[int, int] | None = None) -> dict:
    """`gpus`: the GPUs the server runs on (with `server_url`, the ones that server uses).
    `server_url`: use a server that is already running instead of starting one.
    `shard` (i, n): run only part i of n (CaP-X only; see robovlmbench/benchmarks/capx)."""
    config = load_model_config(model)
    benchmark = BENCHMARKS[bench]
    if hasattr(benchmark, "run"):  # benchmarks that call the model themselves (CaP-X: from a simulator loop)
        return benchmark.run(config, engine, gpus=gpus, port=port, limit=limit, server_url=server_url, shard=shard)
    if shard:
        raise ValueError(f"--shard is only supported for capx, not {bench}")
    samples = benchmark.load_samples()[:limit]
    settings = config.request_settings()
    point_format = POINT_FORMATS[config.point_format]

    out_dir = OUTPUTS_DIR / config.name / engine / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_same_run(out_dir / "run.json", {
        "repo": config.repo, "revision": config.revision, "engine": ENGINES[engine].version(),
        "request_settings": settings, "point_format": config.point_format, "hardware": gpu_names(gpus),
        "concurrency": 1})

    responses_path = out_dir / "responses.jsonl"
    done = load_responses(responses_path)
    todo = [s for s in samples if s.id not in done]
    print(f"{bench}: {len(samples)} samples, {len(samples) - len(todo)} already answered, {len(todo)} to send")

    if todo:
        requests = [benchmark.build_request(s, point_format) for s in todo]
        with _server(config, engine, gpus, port, server_url) as (url, served_name):
            client = Client(url, served_name, settings, max_concurrency=1, fresh_images=True)
            client.ask(requests[:1])  # warm-up, not saved: a new server's first request can be slower
            _ask(client, requests, responses_path)
        done = load_responses(responses_path)

    scores = benchmark.score(samples, [done[s.id] for s in samples], point_format)
    scores["time_per_question_s"] = time_per_question([done[s.id] for s in samples])
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    return scores


def time_per_question(responses: list[Response]) -> dict | None:
    """Median, 90th percentile and mean of the response times, in seconds."""
    if len(responses) < 2:
        return None
    seconds = [r.latency_s for r in responses]
    return {"median": round(statistics.median(seconds), 3), "p90": round(statistics.quantiles(seconds, n=10)[-1], 3),
            "mean": round(statistics.mean(seconds), 3), "n": len(seconds)}


def gpu_names(gpus: str) -> str:
    """E.g. "1× NVIDIA RTX PRO 6000 Blackwell Server Edition" for gpus="0"."""
    names = subprocess.run(["nvidia-smi", "-i", gpus, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    return " + ".join(f"{names.count(name)}× {name}" for name in dict.fromkeys(names))


def _ask(client: Client, requests: list[Request], path: Path) -> None:
    """Send `requests`, appending each response to `path` as it arrives."""
    started, finished = time.monotonic(), 0
    with open(path, "a") as f:
        def save(response: Response) -> None:
            nonlocal finished
            f.write(json.dumps(response.to_dict()) + "\n")
            f.flush()
            finished += 1
            if finished % 25 == 0 or finished == len(requests):
                print(f"  {finished}/{len(requests)} answered ({time.monotonic() - started:.0f}s)")

        client.ask(requests, on_response=save)


def _check_same_run(path: Path, run: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != run:
            raise RuntimeError(f"{path.parent} holds responses from different settings:\n  before: {previous}\n"
                               f"  now:    {run}\nMove or delete that folder to start over.")
    else:
        path.write_text(json.dumps(run, indent=1))


def load_responses(path: Path) -> dict[str, Response]:
    """Latest successful response per sample id; failed ones are sent again."""
    responses = {}
    if path.exists():
        for line in path.read_text().splitlines():
            response = Response(**json.loads(line))
            if not response.error:
                responses[response.id] = response
    return responses


@contextlib.contextmanager
def _server(config, engine, gpus, port, server_url):
    """Yield (base url, model name): a running server's, or one started (and stopped) here."""
    if server_url:
        models = json.load(urllib.request.urlopen(server_url.rstrip("/") + "/models"))
        yield server_url, models["data"][0]["id"]
    else:
        with start(config, engine, gpus=gpus, port=port) as server:
            yield server.url, server.model
