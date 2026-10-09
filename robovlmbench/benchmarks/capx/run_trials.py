"""Run CaP-X trials for RoboVLMBench. Runs in a CaP-X env (Python 3.10 or 3.12), never in the main env:
    source scripts/capx_env.sh robosuite && cd "$CAPX_DIR" && python <this file> job.json

Started by robovlmbench/benchmarks/capx (see there), which writes the job file: the CaP-X configs to run,
the trials of each, where to save, and the URL of the model proxy. Per config this does what CaP-X's own
capx/envs/launch.py does (same functions, same per-trial timeout and attempts), except:
- each trial sends its model calls to its own proxy URL, so every call can be traced to its trial;
- the time spent waiting for the model can be left out of CaP-X's trial timeout (job["timeout_includes_model_time"]);
- each trial's result, final code and last camera image are saved as soon as it ends, to episodes/<config>/;
- the perception servers of all configs are started once, before the first config, and must all be up;
- all configs share one pool of workers, at most half of them on one config at a time (a worker builds a
  config's simulator when it moves to it): a trial that runs long (e.g. model code that loops until the
  timeout) holds up only its own worker; CaP-X's pool runs one config at a time, so one such trial idles the
  others, and a config whose every trial loops would hold up all of them;
- a worker that ends (its simulator failed to start, or the watchdog ended a stuck episode) is replaced.
"""

import dataclasses
import json
import multiprocessing
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import uuid

# An episode still running this long (model time included) is ended and recorded as timed out; a backstop for
# model code that outlasts CaP-X's timeout (3 attempts of 1000 s without the model's time), e.g. a bare
# `except:` inside `while True`.
EPISODE_WATCHDOG_S = 2 * 3600


class TrialTimeout(BaseException):
    """CaP-X's trial timeout. CaP-X raises the built-in TimeoutError, an Exception, so model code that wraps
    its work in `while True: try: ... except Exception: ...` catches the timeout and never ends; this one
    passes through `except Exception`. CaP-X's sandbox still catches it (it catches BaseException) and runs
    the trial's next code block, which may loop again; so once the timeout has fired it fires again every
    second (each raise re-arms the alarm) until the trial ends. CaP-X disarms it when the trial returns."""

    fired = False  # in this process, since the current attempt started

    def __init__(self, *args):
        super().__init__(*args)
        TrialTimeout.fired = True
        signal.alarm(1)


# This file's folder holds the harness's other modules; keep their names out of CaP-X's imports.
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))


def main(job_path: str) -> None:
    from capx.envs.launch import LaunchArgs
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _load_config

    with open(job_path) as f:
        job = json.load(f)
    loaded = []
    for cfg in job["configs"]:
        args = LaunchArgs(config_path=cfg["yaml"], model=job["model_name"], num_workers=job["num_workers"], record_video=job["record_video"],
                          output_dir=cfg["artifacts_dir"], use_oracle_code=False)
        env_factory, config, api_servers = _load_config(args)
        config["output_dir"] = cfg["artifacts_dir"]  # as is (launch.py would add the model name to it)
        loaded.append((cfg, args, env_factory, config, api_servers or []))

    servers = list({server["port"]: server for *_, api_servers in loaded for server in api_servers}.values())
    busy = [s["port"] for s in servers if _port_open(s["port"])]
    if busy:
        sys.exit(f"ports {busy} are in use. CaP-X's perception servers have fixed ports, so only one CaP-X run "
                 "can use a node at a time.")
    processes = _start_api_servers(servers)
    try:
        for cfg, args, env_factory, config, api_servers in loaded:
            down = [s["_target_"].split(".")[-2] + f" (port {s['port']})" for s in api_servers if not _port_open(s["port"])]
            if down:
                sys.exit(f"{cfg['id']}: perception servers not running: {', '.join(down)}. See the log above "
                         "(SAM3's weights need access granted on huggingface.co/facebook/sam3).")
            os.makedirs(config["output_dir"], exist_ok=True)
        _run_pool(job, [(cfg, args, env_factory, config) for cfg, args, env_factory, config, _ in loaded])
    finally:
        _stop_api_servers(processes)


def _run_pool(job: dict, loaded: list) -> None:
    """Run every config's trials on job["num_workers"] processes, each with its own simulator. A worker keeps
    taking trials of its config (its simulator is built per config), then moves to the next config with trials
    left. At most half the workers run one config at once: a config whose trials all run until the timeout
    (model code that loops) then holds up at most half of them. A worker that ends early is replaced (up to
    num_workers times, in case simulators can't start)."""
    context = multiprocessing.get_context("spawn")  # as CaP-X: CUDA and MuJoCo don't survive a fork
    manager = context.Manager()
    shared = {"lock": manager.Lock(), "held": manager.dict(),  # held: worker pid -> config it runs a trial of
              "remaining": manager.dict({i: list(cfg["trials"]) for i, (cfg, *_) in enumerate(loaded)}),
              "active": manager.dict({i: 0 for i in range(len(loaded))})}
    n_workers = min(job["num_workers"], sum(len(cfg["trials"]) for cfg, *_ in loaded))
    cap = max(1, n_workers // 2)

    def start():
        worker = context.Process(target=_worker, args=(shared, cap, job, loaded))
        worker.start()
        return worker

    workers, replaced = [start() for _ in range(n_workers)], 0
    while any(worker.is_alive() for worker in workers):
        time.sleep(5)
        for k, worker in enumerate(workers):
            if worker.is_alive() or worker.exitcode == 0:
                continue
            with shared["lock"]:  # free the slot of the trial it was running
                if (i := shared["held"].pop(worker.pid, None)) is not None:
                    shared["active"][i] -= 1
                left = any(shared["remaining"].values())
            if left and replaced < n_workers:
                print(f"a worker ended (exit code {worker.exitcode}); starting another", flush=True)
                workers[k], replaced = start(), replaced + 1
    if failed := sum(worker.exitcode != 0 for worker in workers):
        print(f"{failed} of {n_workers} workers failed (see above)", flush=True)
    manager.shutdown()


def _next_trial(shared: dict, cap: int, current: int | None) -> tuple[int, int] | str | None:
    """(config, trial) for this worker: its current config's next trial, else the first config with trials left
    and a free slot; "wait" if every config with trials left is at its cap; None when none are left."""
    with shared["lock"]:
        remaining, active = shared["remaining"], shared["active"]
        order = ([current] if current is not None else []) + list(range(len(remaining)))
        for i in order:
            if remaining[i] and active[i] < cap:
                trials = remaining[i]
                remaining[i], active[i] = trials[1:], active[i] + 1
                shared["held"][os.getpid()] = i
                return i, trials[0]
        return "wait" if any(remaining.values()) else None


def _worker(shared: dict, cap: int, job: dict, loaded: list) -> None:
    from capx.envs.runner import _worker_setup

    current, state = None, None
    while (item := _next_trial(shared, cap, current)) is not None:
        if item == "wait":
            time.sleep(2)
            continue
        i, trial = item
        cfg, args, env_factory, config = loaded[i]
        if i != current:  # this config's simulator, built when the worker moves to it
            if state is not None and hasattr(state[0], "close"):
                try:
                    state[0].close()
                except Exception as e:
                    print(f"closing the simulator of {loaded[current][0]['id']}: {type(e).__name__}: {e}", flush=True)
            print(f"=== {cfg['id']}: trial {trial}", flush=True)
            current, state = i, _worker_setup(env_factory=env_factory)
        _trial(state, trial, job=job, cfg=cfg, args=args, config=config)
        with shared["lock"]:
            shared["held"].pop(os.getpid(), None)
            shared["active"][i] -= 1


def _trial(state, trial: int, *, job: dict, cfg: dict, args, config: dict) -> None:
    """One episode, as CaP-X's runner runs it (_run_trial_with_retries); the result goes to a file."""
    from capx.envs.runner import MAX_TRIAL_RETRIES, TRIAL_TIMEOUT_SECONDS, _run_single_trial_with_timeout

    import capx.envs.runner as runner_module

    runner_module.TimeoutError = TrialTimeout  # the name CaP-X's trial timeout raises and checks
    env, multi_turn_prompt = state
    if not job["timeout_includes_model_time"]:
        _stop_timeout_clock_during_model_calls()
    _fill_goal_into_prompts(_goal(env))
    token = uuid.uuid4().hex  # names this episode's calls in the proxy's log
    started = time.monotonic()
    watchdog = threading.Timer(EPISODE_WATCHDOG_S, _end_stuck_episode,
                               kwargs=dict(cfg=cfg, trial=trial, token=token, goal=_goal(env), started=started))
    watchdog.daemon = True
    watchdog.start()
    summary, error, timed_out = None, None, False
    for attempt in range(MAX_TRIAL_RETRIES):
        attempt_args = dataclasses.replace(
            args, server_url=f"{job['proxy_url']}/{token}/{trial}/{attempt}/coder/chat/completions")
        TrialTimeout.fired, summary = False, None
        try:
            summary = _run_single_trial_with_timeout(
                env=env, trial=trial, args=attempt_args, config=config, multi_turn_prompt=multi_turn_prompt,
                timeout_s=TRIAL_TIMEOUT_SECONDS, raise_on_timeout=attempt < MAX_TRIAL_RETRIES - 1)
            # the last attempt returns instead of raising; a timeout CaP-X's sandbox caught returns too
            timed_out = "timed out after" in summary.log or TrialTimeout.fired
        except TrialTimeout:
            timed_out = True
        except Exception:
            error = traceback.format_exc()
            print(f"Trial {trial} failed:\n{error}", flush=True)
            break
        if not timed_out or attempt == MAX_TRIAL_RETRIES - 1:
            break
        print(f"Trial {trial} timed out (attempt {attempt + 1}/{MAX_TRIAL_RETRIES}). Retrying...", flush=True)
    signal.alarm(0)
    watchdog.cancel()
    wall_s = time.monotonic() - started

    episode = {"id": f"{cfg['id']}/{trial}", "config": cfg["id"], "trial": trial, "token": token,
               "attempts": attempt + 1, "goal": _goal(env), "wall_s": round(wall_s, 3), "error": error}
    if summary is not None:
        code = ""
        if summary.code_path and os.path.exists(summary.code_path):
            with open(summary.code_path) as f:
                code = f.read()
        episode.update(task_completed=bool(summary.task_completed), reward=float(summary.reward),
                       sandbox_rc=summary.sandbox_rc, timed_out=timed_out, num_code_blocks=summary.num_code_blocks,
                       num_regenerations=summary.num_regenerations, num_finishes=summary.num_finishes, code=code)
    elif not error:  # the last attempt timed out before CaP-X could sum it up
        episode.update(task_completed=False, reward=0.0, sandbox_rc=None, timed_out=True, num_code_blocks=None,
                       num_regenerations=None, num_finishes=None, code="")
    path = os.path.join(cfg["episodes_dir"], f"{trial:04d}")
    os.makedirs(cfg["episodes_dir"], exist_ok=True)
    _save_frame(env, path + ".jpg")
    _save_episode(path, episode)


def _save_episode(path: str, episode: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".json.tmp", "w") as f:
        json.dump(episode, f, indent=1)
    os.replace(path + ".json.tmp", path + ".json")  # the episode counts as done once this file exists


def _end_stuck_episode(*, cfg: dict, trial: int, token: str, goal: str, started: float) -> None:
    """Watchdog: record the episode as timed out and end this worker (its simulator is stuck in the model's
    code); the other workers take its remaining trials."""
    wall_s = time.monotonic() - started
    print(f"Trial {trial} still running after {wall_s:.0f}s: recorded as timed out, worker ends", flush=True)
    _save_episode(os.path.join(cfg["episodes_dir"], f"{trial:04d}"), {
        "id": f"{cfg['id']}/{trial}", "config": cfg["id"], "trial": trial, "token": token, "attempts": None,
        "goal": goal, "wall_s": round(wall_s, 3), "error": None, "task_completed": False, "reward": 0.0,
        "sandbox_rc": None, "timed_out": True, "num_code_blocks": None, "num_regenerations": None,
        "num_finishes": None, "code": ""})
    os._exit(1)


def _stop_timeout_clock_during_model_calls() -> None:
    """CaP-X's trial timeout (a SIGALRM) counts the model's time too; pause it while the model answers.
    Patches the name CaP-X's trial loop calls the model by (capx.envs.trial._query_model), once per process."""
    import capx.envs.trial as trial_module

    query = trial_module._query_model
    if getattr(query, "clock_stopped", False):
        return

    def query_with_clock_stopped(args, prompt):
        remaining = signal.alarm(0)
        try:
            return query(args, prompt)
        finally:
            if remaining:
                signal.alarm(remaining)

    query_with_clock_stopped.clock_stopped = True
    trial_module._query_model = query_with_clock_stopped


def _fill_goal_into_prompts(goal: str) -> None:
    """CaP-X writes LIBERO's goal into the first prompt only: its multi-turn prompts are built from the latest
    observation, whose prompt still reads "Goal: {libero_environment_goal}". Fill the current trial's goal into
    every prompt. Patches the name CaP-X's trial loop calls the model by (capx.envs.trial._query_model), once
    per process; installed after _stop_timeout_clock_during_model_calls, whose mark it carries over."""
    import capx.envs.trial as trial_module

    _fill_goal_into_prompts.goal = goal
    query = trial_module._query_model
    if getattr(query, "fills_goal", False):
        return

    def fill(text: str) -> str:
        return text.replace("{libero_environment_goal}", _fill_goal_into_prompts.goal)

    def query_with_goal(args, prompt):
        for message in prompt:
            if isinstance(message["content"], str):
                message["content"] = fill(message["content"])
            else:
                for part in message["content"]:
                    if part.get("type") == "text":
                        part["text"] = fill(part["text"])
        return query(args, prompt)

    query_with_goal.fills_goal = True
    query_with_goal.clock_stopped = getattr(query, "clock_stopped", False)
    trial_module._query_model = query_with_goal


def _goal(env) -> str:
    """The task in words: LIBERO's instruction, or the "Goal:" line of the prompt."""
    language = getattr(getattr(env.low_level_env, "handle", None), "task_language", None)
    if language:
        return language
    match = re.search(r"^\s*Goal:\s*(.+)$", env._task_prompt or "", re.M)
    return match.group(1).strip() if match else ""


def _save_frame(env, path: str, width: int = 480) -> None:
    """The main camera's current image, as a small JPEG (for the report)."""
    from PIL import Image

    try:
        image = Image.fromarray(env.render())
    except Exception as e:  # an image is nice to have, not a reason to lose the episode
        print(f"no final image: {type(e).__name__}: {e}", flush=True)
        return
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image.convert("RGB").save(path, "JPEG", quality=85)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", int(port))) == 0


if __name__ == "__main__":
    main(sys.argv[1])
