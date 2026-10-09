"""Start and stop an inference engine serving one model on an OpenAI-style API.

    with start(load_model_config("qwen3.8-27b-nvfp4"), engine="vllm", gpus="0") as server:
        ...  # send requests to server.url, as model server.model
"""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from robovlmbench import REPO_ROOT
from robovlmbench.config import ModelConfig
from robovlmbench.server import vllm

ENGINES = {"vllm": vllm}


@dataclass
class Server:
    engine: str
    url: str      # OpenAI-style base URL, e.g. http://127.0.0.1:8000/v1
    model: str    # model name to put in requests
    log_path: Path
    process: subprocess.Popen

    def stop(self, timeout: float = 120) -> None:
        """Stop the server and every process it started (engines spawn GPU worker processes)."""
        if self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def start(config: ModelConfig, engine: str = "vllm", gpus: str = "0", port: int = 8000,
          log_path: Path | None = None, timeout: float = 1800) -> Server:
    """Start `engine` serving `config` on the given GPUs and wait until it answers.

    The first start of a model on a GPU type compiles kernels (several minutes); later starts reuse
    cache/<gpu>/. Raises with the end of the server log if it exits or isn't ready within `timeout`.
    """
    weights = config.weights_path()
    command = ENGINES[engine].command(config, weights, port)
    log_path = log_path or REPO_ROOT / "logs" / f"server-{config.name}-{engine}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "HF_HUB_OFFLINE": "1"}  # weights are local by now
    with open(log_path, "w") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)  # own process group, so stop() reaches workers
    server = Server(engine, f"http://127.0.0.1:{port}/v1", config.name, log_path, process)
    try:
        _wait_until_ready(server, timeout)
    except BaseException:
        server.stop()
        raise
    return server


def _wait_until_ready(server: Server, timeout: float) -> None:
    health = server.url.removesuffix("/v1") + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"{server.engine} server exited (code {server.process.returncode}):\n"
                               + _log_tail(server.log_path))
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"{server.engine} server not ready after {timeout:.0f}s:\n" + _log_tail(server.log_path))


def _log_tail(path: Path, lines: int = 20) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
