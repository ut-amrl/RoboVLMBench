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
from dataclasses import dataclass, field
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
    memory: dict = field(default_factory=dict)  # what the server reports about its memory (vLLM)

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
    cache/<gpu>/. vLLM gets the memory the model needs (server/vllm_memory.py); the first start of a model at
    its length is preceded by a short one that finds its KV cache need. Raises with the end of the server log if
    it exits or isn't ready within `timeout`.
    """
    weights = config.weights_path()
    command = ENGINES[engine].command(config, weights, port)
    log_path = log_path or REPO_ROOT / "logs" / f"server-{config.name}-{engine}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "HF_HUB_OFFLINE": "1"}  # weights are local by now
    memory_key = None

    def probe(args):
        probe_log = log_path.with_name(log_path.stem + "-memory-probe.log")
        probe_server = Server(engine, "", config.name, probe_log, _launch(command + args, env, probe_log))
        deadline = time.monotonic() + timeout
        while probe_server.process.poll() is None and time.monotonic() < deadline:
            if any(marker in probe_log.read_text(errors="replace") for marker in vllm.PROBE_DONE):
                break
            time.sleep(2)
        probe_server.stop()
        return probe_log.read_text(errors="replace")

    for attempt in range(3):  # vLLM: again with more KV cache if a start finds it short
        args = []
        if engine == "vllm":
            args, memory_key = vllm.memory_args(config, weights, gpus, probe)
        server = Server(engine, f"http://127.0.0.1:{port}/v1", config.name, log_path,
                        _launch(command + args, env, log_path))
        try:
            _wait_until_ready(server, timeout)
            break
        except BaseException:
            server.stop()
            if not (memory_key and attempt < 2 and vllm.add_shortfall(memory_key, log_path.read_text(errors="replace"))):
                raise
    if memory_key:
        server.memory = vllm.record_memory(memory_key, server.process.pid, log_path.read_text(errors="replace"))
    return server


def _launch(command: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    with open(log_path, "w") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)  # own process group, so stop() reaches workers


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
