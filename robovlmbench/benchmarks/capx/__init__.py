"""CaP-X (third_party/cap-x): the model writes Python that drives a simulated robot through CaP-X's perception
and control functions; an episode succeeds if the task is done when the code has run.

A *config* is one CaP-X task in one *variant*: the CaP-X YAML that sets the robot API (privileged simulator
state, or perception), single- or multi-turn, with or without images. Robosuite variants are CaP-X's own files,
named by their suffix (franka_robosuite_cube_lifting_privileged.yaml -> cube_lifting/privileged). LIBERO-PRO
tasks are made from one of CaP-X's LIBERO configs with the suite and task swapped in, as CaP-X's
capx/envs/scripts/run_libero_batch.py does (-> libero_goal_swap/3/privileged). Each config runs for some
*trials* (episodes): trial n resets the simulator with seed n (LIBERO: its n-th initial state). Which configs,
how many trials and how they run is the protocol: configs/benchmarks/capx.yaml.

CaP-X runs in its own env as a subprocess (run_trials.py; one env per suite, see scripts/setup_capx.sh). Its
model calls go to a local proxy (proxy.py) that queries the model with the model's own request settings and
logs every call. Prompts, code extraction, execution and success checks are CaP-X's, unchanged; CaP-X is
told the model is "google/gemini-3.1-pro-preview" (its default), the proxy answers with the evaluated model.
Configs that also query hosted models (the _vdm variants, ensembles) are not supported.

Suites that can't run on this node's hardware (no CaP-X env for it, or a failed CaP-X check in
scripts/check_env.py) are skipped and recorded as N/A (unavailable.json), for the report.

Outputs go to outputs/<model>/<engine>/capx/, next to run.json and scores.json:
    episodes/<config>/<trial>.json   one per finished episode: task done or not, final code, time
    episodes/<config>/<trial>.jpg    the camera image when the episode ended
    calls.jsonl                      every model call: request (images left out), response, time
                                     (calls-shard<i>of<n>.jsonl for a shard)
    capx/<config>/                   CaP-X's own files per trial (code, prompts, responses)
    logs/                            CaP-X's console output, one file per suite and run

Re-running resumes: finished episodes are not run again; episodes with a failed model call (server error)
or an error in CaP-X are.

A run can be split into shards that run at the same time, each with its own server (e.g. one per GPU): shard i
of n takes every n-th episode. A shard runs CaP-X's perception servers on their fixed ports, so shards on one
node need separate network namespaces (`unshare -rn`, with its loopback up). Run once more without a shard
afterwards to score the whole run (nothing is left to run then, so no server is started).
"""

import json
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from robovlmbench import REPO_ROOT, THIRD_PARTY_DIR
from robovlmbench.benchmarks.capx.proxy import Proxy
from robovlmbench.config import ModelConfig
from robovlmbench.server import ENGINES

CAPX_DIR = THIRD_PARTY_DIR / "cap-x"
PROTOCOL_PATH = REPO_ROOT / "configs" / "benchmarks" / "capx.yaml"
RUN_TRIALS = Path(__file__).parent / "run_trials.py"
LIBERO_TASK_FILES = REPO_ROOT / "cache" / "libero" / "bddl_files"  # every LIBERO(-PRO) suite, by scripts/setup_capx.sh
SUITES = ("robosuite", "libero")  # each needs its own CaP-X env
# The name CaP-X is given for the model (and for its visual-differencing model): CaP-X sends images only to
# models on its list of vision models, and its default model gets the plain request format. The proxy
# answers every call with the evaluated model, whatever the name.
CAPX_MODEL_NAME = "google/gemini-3.1-pro-preview"
# CaP-X's trial timeout and attempts (capx/envs/runner.py), recorded in run.json
TRIAL_TIMEOUT_S, TRIAL_ATTEMPTS = 1000, 3

# A few CaP-X files spell a variant differently (two_arm_*: "_reduced" for "_reduced_api")
_VARIANT_SPELLINGS = {"reduced": "reduced_api", "reduced_exampleless": "reduced_api_exampleless"}
# CaP-X config switches that make it query hosted models besides the evaluated one: a visual-differencing model
# (the _vdm variants; CaP-X only accepts GPT/Claude/Gemini-class models there) or a model ensemble
_HOSTED_MODEL_KEYS = ("use_img_differencing", "use_video_differencing", "use_parallel_ensemble")
# LIBERO-PRO variants: the CaP-X config each task is swapped into
LIBERO_VARIANTS = {
    "privileged": "libero/franka_libero_spatial_0_privileged.yaml",       # single-turn, simulator state
    "multiturn_privileged": "libero/franka_libero_goal_1.yaml",           # multi-turn, simulator state
    "multiturn": "libero/franka_libero_spatial_0.yaml",                   # multi-turn, perception (SAM3)
}


@dataclass(frozen=True)
class Config:
    id: str         # "<task>/<variant>", e.g. "cube_lifting/privileged", "libero_goal_swap/3/privileged"
    suite: str      # "robosuite" or "libero"
    task: str       # "cube_lifting", "libero_goal_swap/3"
    variant: str
    yaml: Path      # CaP-X's config; for LIBERO, the one the task is swapped into
    trials: int


@dataclass(frozen=True)
class Episode:
    id: str         # "<config id>/<trial>"
    config: Config
    trial: int      # 1-based, as in CaP-X


def support_records() -> dict[str, dict]:
    """Which suites run on which GPU type, by GPU type (cache/<gpu>/capx-check.json): written by
    scripts/check_env.py's CaP-X checks, or by scripts/setup_capx.sh when it skips a GPU node.
    Each: {"gpu": <name>, "suites": {<suite>: {"ok": bool, "detail": str, "date": str}}}."""
    return {path.parent.name: json.loads(path.read_text())
            for path in sorted((REPO_ROOT / "cache").glob("*/capx-check.json"))}


def record_support(gpu: str, gpu_name: str, suite: str, ok: bool, detail: str) -> None:
    """Record whether `suite` runs on GPU type `gpu` (RVB_GPU, e.g. "rtx-pro-6000-blackwell-server-edition")."""
    path = REPO_ROOT / "cache" / gpu / "capx-check.json"
    record = json.loads(path.read_text()) if path.exists() else {"gpu": gpu_name, "suites": {}}
    record["suites"][suite] = {"ok": ok, "detail": detail, "date": time.strftime("%Y-%m-%d")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))


def unavailable_suites(gpu: str) -> dict[str, str]:
    """Suites that can't run on this node, with why: no CaP-X env for this node's hardware (setup_capx.sh
    skipped it), or a failed CaP-X check recorded for this GPU type."""
    record = support_records().get(gpu, {"suites": {}})
    reasons = {}
    for suite in SUITES:
        checked = record["suites"].get(suite)
        if gpu == "cpu":
            reasons[suite] = "no NVIDIA GPU"
        elif checked and not checked["ok"]:
            reasons[suite] = checked["detail"]
        elif not (REPO_ROOT / ".venvs" / f"capx-{suite}-{os.uname().machine}").is_dir():
            reasons[suite] = "CaP-X env not installed for this hardware (scripts/setup_capx.sh)"
    return reasons


def load_protocol(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PROTOCOL_PATH).read_text())


def load_configs(protocol: dict) -> list[Config]:
    """The protocol's configs: every task with every variant of its suite."""
    configs = []
    robosuite = protocol.get("robosuite") or {}
    for task in robosuite.get("tasks", []):
        variants = robosuite_variants(task)
        for variant in robosuite.get("variants", []):
            if variant not in variants:
                raise ValueError(f"CaP-X has no {task} config for variant {variant!r}; it has: {', '.join(variants)}")
            configs.append(Config(f"{task}/{variant}", "robosuite", task, variant, variants[variant],
                                  robosuite["trials"]))
    libero = protocol.get("libero") or {}
    for suite in libero.get("suites", []):
        if not (LIBERO_TASK_FILES / suite).is_dir():
            raise ValueError(f"no LIBERO task files for suite {suite!r} in {LIBERO_TASK_FILES} "
                             "(run scripts/setup_capx.sh libero)")
        for task_id in libero.get("task_ids", []):
            for variant in libero.get("variants", []):
                if variant not in LIBERO_VARIANTS:
                    raise ValueError(f"unknown LIBERO variant {variant!r}; known: {', '.join(LIBERO_VARIANTS)}")
                task = f"{suite}/{task_id}"
                configs.append(Config(f"{task}/{variant}", "libero", task, variant,
                                      CAPX_DIR / "env_configs" / LIBERO_VARIANTS[variant], libero["trials"]))
    for config in configs:
        raw = yaml.safe_load(config.yaml.read_text())
        if any(raw.get(key) for key in _HOSTED_MODEL_KEYS):
            raise ValueError(f"{config.id}: this CaP-X config also queries a hosted model (GPT, Claude, Gemini: "
                             "visual differencing or ensembles); not supported")
    return configs


def robosuite_variants(task: str) -> dict[str, Path]:
    """Variant -> CaP-X config of a Robosuite task, e.g. "privileged" -> .../franka_robosuite_cube_lifting_privileged.yaml.
    The task's default config (variant "default") is the one with the shortest name; the others add a suffix."""
    paths = sorted((CAPX_DIR / "env_configs" / task).glob("*.yaml"), key=lambda p: (len(p.stem), p.stem))
    if not paths:
        raise ValueError(f"no CaP-X configs for task {task!r} in {CAPX_DIR / 'env_configs'}")
    variants = {}
    for path in paths:
        variant = path.stem.removeprefix(paths[0].stem).lstrip("_") or "default"
        variants[_VARIANT_SPELLINGS.get(variant, variant)] = path
    return variants


def episode_order(configs: list[Config]) -> list[Episode]:
    """Every config's trials, trial by trial (trial 1 of every config first), so any prefix covers all configs."""
    return [Episode(f"{c.id}/{trial}", c, trial)
            for trial in range(1, max((c.trials for c in configs), default=0) + 1)
            for c in configs if trial <= c.trials]


def run(config: ModelConfig, engine: str = "vllm", *, gpus: str = "0", port: int = 8000,
        limit: int | None = None, server_url: str | None = None, shard: tuple[int, int] | None = None) -> dict:
    """Run the protocol's episodes (the first `limit` only, if given) and score them; see the module doc.
    Called by robovlmbench.runner.run, with the same arguments. `shard` (i, n): run only shard i (1-based) of
    n; the scores still cover every episode."""
    from robovlmbench import runner  # runner imports this module

    protocol = load_protocol()
    configs = load_configs(protocol)
    unavailable = {suite: why for suite, why in unavailable_suites(os.environ.get("RVB_GPU", "cpu")).items()
                   if any(c.suite == suite for c in configs)}
    for suite, why in unavailable.items():
        print(f"capx: {suite} skipped, N/A on this hardware: {why}")
    episodes = episode_order([c for c in configs if c.suite not in unavailable])[:limit]
    settings = config.request_settings()
    out_dir = runner.OUTPUTS_DIR / config.name / engine / "capx"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unavailable.json").write_text(json.dumps(unavailable, indent=1))  # for the report: N/A, and why
    run_info = {
        "repo": config.repo, "revision": config.revision, "engine": ENGINES[engine].version(),
        "request_settings": settings, "max_seq_len": config.max_seq_len, "hardware": runner.gpu_names(gpus),
        "concurrency": 1,
        "capx": {"commit": capx_commit(), "model_name": CAPX_MODEL_NAME, "num_workers": protocol["num_workers"], "trial_timeout_s": TRIAL_TIMEOUT_S,
                 "trial_attempts": TRIAL_ATTEMPTS,
                 "timeout_includes_model_time": protocol["timeout_includes_model_time"],
                 "seed_by_trial": protocol["seed_by_trial"]}}
    run_path = out_dir / "run.json"
    if run_path.exists():  # the number of simulators may change between resumes: each episode is set by its seed
        previous = json.loads(run_path.read_text())
        if {**previous, "capx": {**previous["capx"], "num_workers": protocol["num_workers"]}} == run_info:
            run_path.write_text(json.dumps(run_info, indent=1))
    runner._check_same_run(run_path, run_info)

    mine, calls_name, tag = episodes, "calls.jsonl", ""
    if shard:
        i, n = shard
        mine, calls_name, tag = episodes[i - 1::n], f"calls-shard{i}of{n}.jsonl", f"-shard{i}of{n}"
    done = load_episodes(out_dir)
    todo = [e for e in mine if e.id not in done]
    n_configs = len({e.config.id for e in episodes})
    print(f"capx: {len(episodes)} episodes ({n_configs} configs){f', shard {tag[6:]}: {len(mine)}' if shard else ''}, "
          f"{len(mine) - len(todo)} already done, {len(todo)} to run")

    if todo:
        with runner._server(config, engine, gpus, port, server_url) as (url, served_name), \
                Proxy(url, served_name, settings, out_dir / calls_name, max_seq_len=config.max_seq_len,
                      seed_by_trial=protocol["seed_by_trial"]) as proxy:
            proxy.warm_up()
            for suite in SUITES:
                if suite_todo := [e for e in todo if e.config.suite == suite]:
                    _run_suite(suite, suite_todo, proxy, protocol, out_dir, gpus, tag)
        done = load_episodes(out_dir)

    scores = score(episodes, done)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    if missing := sum(e.id not in done for e in mine):
        raise RuntimeError(f"{missing} of {len(mine)} episodes did not finish (server errors or "
                           f"CaP-X errors; see {out_dir / 'logs'}); run again to retry them")
    return scores


def _run_suite(suite: str, episodes: list[Episode], proxy: Proxy, protocol: dict, out_dir: Path, gpus: str,
               tag: str = "") -> None:
    """Run `episodes` (all of one suite) in one CaP-X subprocess, printing progress as they finish."""
    trials = {}
    for e in episodes:
        trials.setdefault(e.config, []).append(e.trial)
    job = {
        "proxy_url": proxy.url, "model_name": CAPX_MODEL_NAME, "num_workers": protocol["num_workers"],
        "record_video": protocol["record_video"], "timeout_includes_model_time": protocol["timeout_includes_model_time"],
        "configs": [{"id": c.id, "yaml": str(_capx_config_file(c, out_dir)), "trials": t,
                     "artifacts_dir": str(out_dir / "capx" / c.id), "episodes_dir": str(out_dir / "episodes" / c.id)}
                    for c, t in trials.items()],
    }
    (out_dir / "logs").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{suite}-{stamp}{tag}"
    job_path, log_path = out_dir / "logs" / f"{name}.job.json", out_dir / "logs" / f"{name}.log"
    job_path.write_text(json.dumps(job, indent=1))
    command = ["bash", "-c", 'source scripts/capx_env.sh "$1" >/dev/null && cd "$CAPX_DIR" && exec python "$2" "$3"',
               "capx", suite, str(RUN_TRIALS), str(job_path)]
    # LIBERO's initial states are pickles that torch.load only opens with this (as CaP-X's LIBERO scripts set)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
    print(f"  {suite}: {len(episodes)} episodes of {len(trials)} configs; CaP-X log: {log_path}")

    started, reported = time.monotonic(), 0
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)  # own process group: stopping it reaches its servers
        try:
            while process.poll() is None:
                time.sleep(10)
                finished = sum(_episode_path(out_dir, e).exists() for e in episodes)
                if finished >= reported + 25 or (finished == len(episodes) and finished > reported):
                    print(f"  {finished}/{len(episodes)} episodes done ({time.monotonic() - started:.0f}s)")
                    reported = finished
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"CaP-X ({suite}) exited with code {process.returncode}; end of {log_path}:\n{tail}")


def _capx_config_file(config: Config, out_dir: Path) -> Path:
    """The CaP-X YAML to run `config` with: CaP-X's own for Robosuite; for LIBERO, CaP-X's config with the
    task swapped in, written next to the config's CaP-X files."""
    if config.suite == "robosuite":
        return config.yaml
    raw = yaml.safe_load(config.yaml.read_text())
    suite_name, task_id = config.task.split("/")
    cfg = raw["env"]["cfg"]
    cfg["low_level"] = {  # as capx/envs/scripts/run_libero_batch.py sets it
        "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv", "suite_name": suite_name, "task_id": int(task_id),
        "privileged": cfg.get("privileged", False), "max_steps": 8000, "seed": None, "enable_render": True,
        "viser_debug": False}
    if all(api.endswith("PrivilegedApi") for api in cfg["apis"]):
        # CaP-X's privileged LIBERO config also starts SAM3 and GraspNet, which privileged APIs never call
        raw["api_servers"] = [s for s in raw.get("api_servers", []) if "pyroki" in s["_target_"]]
    path = out_dir / "capx" / config.id / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"config.yaml.{os.getpid()}")  # another shard may be reading it
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False))
    os.replace(temporary, path)
    return path


def _episode_path(out_dir: Path, episode: Episode) -> Path:
    return out_dir / "episodes" / episode.config.id / f"{episode.trial:04d}.json"


def load_calls(out_dir: Path) -> dict[str, list[dict]]:
    """The proxy's logged model calls, per episode token."""
    calls = {}
    for path in sorted(out_dir.glob("calls*.jsonl")):  # calls.jsonl, and one per shard
        for line in path.read_text().splitlines():
            call = json.loads(line)
            calls.setdefault(call["token"], []).append(call)
    return calls


def load_episodes(out_dir: Path) -> dict[str, dict]:
    """Finished episodes by id, each with its model calls ("calls"). Episodes that hit an error in CaP-X or a
    server error on a model call are left out (to be run again)."""
    calls = load_calls(out_dir)
    episodes = {}
    for path in sorted((out_dir / "episodes").rglob("*.json")):
        episode = json.loads(path.read_text())
        episode["calls"] = calls.get(episode["token"], [])
        if not episode["error"] and not any(c["server_error"] for c in episode["calls"]):
            episodes[episode["id"]] = episode
    return episodes


def outcome(episode: dict) -> str:
    """success; else why not: timeout (CaP-X's trial timeout), no_code (the model gave no code: cut off at
    max_tokens, prompt too long, or an empty answer), code_error (the code raised), or incomplete (it ran,
    but the task is not done)."""
    if episode["task_completed"]:
        return "success"
    if episode["timed_out"]:
        return "timeout"
    code = "\n".join(line for line in episode["code"].splitlines() if not line.startswith("# Code block "))
    if not code.strip():
        return "no_code"
    return "code_error" if episode["sandbox_rc"] else "incomplete"


OUTCOMES = ("success", "incomplete", "code_error", "no_code", "timeout")


def score(episodes: list[Episode], done: dict[str, dict]) -> dict:
    """Task success rate per config, and its mean over configs: overall, per suite and per variant.
    Plus why episodes failed, model calls per episode, and time per call and per episode.
    Only finished episodes count; `missing` is how many of `episodes` are not."""
    finished = [(e, done[e.id]) for e in episodes if e.id in done]
    per_sample = []
    for e, d in finished:
        calls = d["calls"]
        per_sample.append({
            "id": e.id, "config": e.config.id, "trial": e.trial, "success": d["task_completed"], "outcome": outcome(d),
            "calls": len(calls), "model_s": round(sum(c["latency_s"] for c in calls), 3),
            # what the episode would take alone: its time minus waiting for other episodes' model calls
            "episode_s": round(d["wall_s"] - sum(c["wait_s"] for c in calls), 3)})

    by_config = {}
    for row in per_sample:
        by_config.setdefault(row["config"], []).append(row["success"])
    configs = {e.config.id: e.config for e in episodes}
    rate = {c: sum(s) / len(s) for c, s in by_config.items()}

    def mean_rate(ids):
        ids = [c for c in ids if c in rate]
        return round(statistics.mean(rate[c] for c in ids), 4) if ids else None

    def group(key):
        groups = {}
        for c in configs.values():
            groups.setdefault(key(c), []).append(c.id)
        return {g: {"success_rate": mean_rate(ids), "configs": len(ids)} for g, ids in groups.items()}

    calls = [c for _, d in finished for c in d["calls"]]
    outcomes = [row["outcome"] for row in per_sample]
    return {
        "success_rate": mean_rate(configs),
        "by_suite": group(lambda c: c.suite),
        "by_variant": group(lambda c: f"{c.suite}/{c.variant}"),
        "by_config": {c: {"success_rate": round(rate[c], 4), "n": len(by_config[c])} for c in configs if c in rate},
        "n": len(per_sample),
        "missing": len(episodes) - len(per_sample),
        "outcomes": {o: outcomes.count(o) for o in OUTCOMES},
        "calls": {
            "n": len(calls),
            "per_episode_median": statistics.median(r["calls"] for r in per_sample) if per_sample else None,
            "output_tokens_median": statistics.median(c["completion_tokens"] for c in calls) if calls else None,
            "cut_off": sum(c["finish_reason"] == "length" for c in calls),
            "max_tokens_lowered": sum(c["max_tokens_lowered"] for c in calls),
            "prompt_too_long": sum(c["error"] is not None for c in calls),
        },
        "time_per_call_s": timing([c["latency_s"] for c in calls]),
        "time_per_episode_s": timing([r["episode_s"] for r in per_sample]),
        "per_sample": per_sample,
    }


def timing(seconds: list[float]) -> dict | None:
    """Median, 90th percentile and mean, in seconds (as runner.time_per_question)."""
    if len(seconds) < 2:
        return None
    return {"median": round(statistics.median(seconds), 3), "p90": round(statistics.quantiles(seconds, n=10)[-1], 3),
            "mean": round(statistics.mean(seconds), 3), "n": len(seconds)}


def capx_commit() -> str:
    return subprocess.run(["git", "-C", str(CAPX_DIR), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
