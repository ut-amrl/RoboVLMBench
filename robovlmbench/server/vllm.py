"""vLLM: serve a model with `vllm serve` from the project env."""

import importlib.metadata
import sys
from pathlib import Path

from robovlmbench.config import ModelConfig


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
