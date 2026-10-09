"""Model configs (configs/models/<model>.yaml): which weights, how each engine serves them, how to query them.

A config can define modes (e.g. think / nothink); select one with "<model>_<mode>", e.g.
"qwen3.8-27b-nvfp4_think". A mode overrides top-level fields; its `generation` is merged into the
top-level one. Server-side settings (engines, max_seq_len) are shared, so one server can answer
every mode that uses the same weights.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from robovlmbench import REPO_ROOT

MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"


@dataclass(frozen=True)
class ModelConfig:
    name: str       # "<model>" or "<model>_<mode>"; names the results folder
    repo: str       # Hugging Face repo id
    revision: str   # pinned commit
    max_seq_len: int
    files: list[str] | None = None  # which files to download (glob patterns); None = the whole repo
    engines: dict[str, list[str]] = field(default_factory=dict)  # engine -> extra server arguments
    # Request settings: temperature, max_tokens, top_p, ... passed through to the API;
    # system_prompt (text) or system_prompt_file (a file in the model repo) is sent as the system prompt;
    # prompt_suffix is appended to every question;
    # extra_body holds engine-specific request fields (e.g. chat_template_kwargs).
    generation: dict = field(default_factory=dict)
    point_format: str = "normalized"  # how the model writes points; see robovlmbench/formats.py

    def engine_args(self, engine: str) -> list[str]:
        return [str(a) for a in self.engines.get(engine, [])]

    def weights_path(self) -> Path:
        """Local copy of the pinned weights, downloaded into models/ on first use."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = dict(revision=self.revision, allow_patterns=self.files)
        try:  # no network needed once downloaded
            return Path(snapshot_download(self.repo, local_files_only=True, **kwargs))
        except LocalEntryNotFoundError:
            return Path(snapshot_download(self.repo, **kwargs))

    def request_settings(self) -> dict:
        """`generation` with system_prompt_file replaced by the prompt text."""
        settings = dict(self.generation)
        if prompt_file := settings.pop("system_prompt_file", None):
            settings["system_prompt"] = (self.weights_path() / prompt_file).read_text()
        return settings


def load_model_config(name: str) -> ModelConfig:
    """Load "<model>" or "<model>_<mode>" from configs/models/<model>.yaml."""
    model, _, mode = name.partition("_")
    raw = yaml.safe_load((MODEL_CONFIG_DIR / f"{model}.yaml").read_text())
    modes = raw.pop("modes", {})
    if modes and not mode:
        raise ValueError(f"{model} has modes; pick one: {', '.join(f'{model}_{m}' for m in modes)}")
    if mode:
        if mode not in modes:
            raise ValueError(f"{model} has no mode {mode!r}; modes: {', '.join(modes) or 'none'}")
        override = dict(modes[mode])
        raw["generation"] = {**raw.get("generation", {}), **override.pop("generation", {})}
        raw.update(override)
    return ModelConfig(name=name, **raw)
