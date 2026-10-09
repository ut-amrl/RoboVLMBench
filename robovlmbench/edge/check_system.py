#!/usr/bin/env python3
"""Check that this device can run the package, reading only: the GPU, the system's vLLM (the `vllm` on PATH: its
version, and whether its source has the models' architectures and quantization and the arguments run_bench.py
gives it), TensorRT-LLM, the memory and disk the models need, and the server port. Python 3.8+, standard library
only.

    python3 check_system.py      # also: python3 run_bench.py --check-system

Nothing is installed, changed or written: what is missing or unsuitable is reported with the commands that would
fix it, for the user to run. vLLM isn't imported (that writes ~/.config/vllm): its version and files are found by
the Python its `vllm` command runs with (importlib.metadata and importlib.util.find_spec, which read the installed
files without running vLLM's code).
"""

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True  # write nothing (no __pycache__ for vllm_memory.py either)

GiB = 1 << 30
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"  # run_bench.py's
HEADROOM = 0.2  # memory to spare over a model's need, under which it is a warning
# The models' architecture and quantization (method, then the algorithms vLLM's source has to name), by the name of
# their repo without the organization, as in their config.json: for models whose config isn't in the HF cache yet
KNOWN_MODELS = {"Qwen3.5-": ("Qwen3_5ForConditionalGeneration", "modelopt", ("NVFP4",)),
                "Qwen3.8-": ("Qwen3_5ForConditionalGeneration", "modelopt", ("MIXED_PRECISION", "NVFP4", "FP8")),
                "Cosmos-Reason2-": ("Qwen3VLForConditionalGeneration", "compressed-tensors", ("nvfp4",))}
# Where vLLM's source names them (paths in its package)
REGISTRY = "model_executor/models/registry.py"
QUANTIZATION = {"modelopt": "model_executor/layers/quantization/modelopt.py",
                "compressed-tensors": "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"}
REASONING = "reasoning"
ENGINE_ARGS = "engine/arg_utils.py"
NGC_VLLM = "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm"
ENV = {**os.environ, "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
# Run by a console script's Python: a distribution's version and its package's directories, without importing it
PROBE = """
import json, sys, sysconfig
out = {"python": sys.version.split()[0], "sys_path": [p for p in sys.path if p]}
try:
    out["stdlib"] = sysconfig.get_path("stdlib")
    out["scripts"] = [sysconfig.get_path("scripts"), sysconfig.get_path("scripts", "posix_user")]
except Exception:
    pass
try:
    import importlib.metadata as metadata
    for name in (sys.argv[1], sys.argv[1].replace("_", "-")):
        try:
            out["version"] = metadata.version(name)
            break
        except Exception:
            pass
except ImportError:
    pass
try:
    import importlib.util
    spec = importlib.util.find_spec(sys.argv[2])
    out["package"] = list(spec.submodule_search_locations or []) if spec else []
except Exception:
    pass
print(json.dumps(out))
"""


class Report:
    """The check's lines, printed as they come, and the commands that would fix what failed."""

    def __init__(self):
        self.failed = False
        self.fixes = []

    def line(self, status: str, text: str, fix: str = "") -> None:
        print(f"{status}  {text}", flush=True)
        if status == "FAIL":
            self.failed = True
            if fix and fix not in self.fixes:
                self.fixes.append(fix)


def check(manifest: dict, here: Path, base_url: str = DEFAULT_BASE_URL) -> bool:
    """Print a PASS / WARN / FAIL line per check (INFO: for information) and the commands that would fix what
    failed. Returns whether nothing failed."""
    report = Report()
    models = manifest["models"]
    hub = _hub_cache()
    compute_cap = _device(report)
    _vllm(report, models, hub, here, compute_cap)
    _tensorrt_llm(report)
    _memory(report, models, here)
    _disk(report, models, hub)
    _port(report, models, base_url)
    if report.fixes:
        print("\nTo fix (commands for you to run; this check changed nothing):")
        for fix in report.fixes:
            print("  " + fix.replace("\n", "\n  "))
    print("\n" + ("not ready: see the FAIL lines" if report.failed else "ready (WARN lines are worth a look)"))
    return not report.failed


def _device(report: Report) -> str:
    """The GPU (run_bench.py's: the first of CUDA_VISIBLE_DEVICES) and the system. Returns the GPU's compute
    capability ("" if unknown)."""
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    name, compute_cap, driver = (_smi(field, gpu) for field in ("name", "compute_cap", "driver_version"))
    cuda = re.search(r"CUDA (?:UMD )?Version: ([\d.]+)", _run(["nvidia-smi"]) or "")  # the driver's
    if name:
        report.line("PASS", f"GPU {name}, compute capability {compute_cap or 'unknown'}, driver {driver or 'unknown'}"
                            + (f", CUDA {cuda.group(1)}" if cuda else ""))
    else:
        report.line("WARN", "no GPU named by nvidia-smi" + ("" if shutil.which("nvidia-smi") else " (not on PATH)")
                    + ": vLLM needs an NVIDIA GPU")
    system = [platform.machine(), f"Python {platform.python_version()} ({sys.executable})"]
    tegra = Path("/etc/nv_tegra_release")
    if tegra.is_file():  # e.g. "# R38 (release), REVISION: 2.0, GCID: ..."
        first = (tegra.read_text(errors="replace").splitlines() or [""])[0]
        match = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", first)
        system.append(f"L4T R{match.group(1)}.{match.group(2)}" if match else first.strip("# "))
        jetpack = _run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
        if jetpack and jetpack.strip():
            system.append(f"JetPack {jetpack.strip()}")
    if sys.version_info >= (3, 8):
        report.line("PASS", "system " + ", ".join(system))
    else:
        report.line("FAIL", "system " + ", ".join(system) + ": run_bench.py needs Python 3.8+",
                    "Run the package with a Python 3.8+ (e.g. python3.8 run_bench.py ...)")
    return compute_cap


def _vllm(report: Report, models: dict, hub: Path, here: Path, compute_cap: str) -> None:
    """The `vllm` on PATH (what run_bench.py starts): its version against the full runs', and its source."""
    wanted = sorted({info["vllm_version_of_the_full_runs"].split()[-1] for info in models.values()})  # "vllm 0.31.0"
    full = ", ".join(f"vllm {v}" for v in wanted)
    script = shutil.which("vllm")
    if script is None:
        python = shutil.which("python3") or sys.executable
        found = _probe(python, "vllm", "vllm")
        scripts = [d for d in found.get("scripts") or [] if d and Path(d, "vllm").is_file()]
        if found.get("version") and scripts:  # e.g. installed with pip --user, ~/.local/bin not on PATH
            report.line("FAIL", f"no `vllm` on PATH, but vLLM {found['version']} is installed for {python}, its "
                                f"command in {scripts[0]} (full runs: {full})",
                        f"Put vLLM's command on PATH:\n    export PATH={scripts[0]}:$PATH")
        else:
            report.line("FAIL", f"no `vllm` on PATH (full runs: {full})",
                        _install_fix(python, found, wanted[0], here, hub))
        return
    python = _interpreter(script)
    found = _probe(python, "vllm", "vllm") if python else {}
    version = found.get("version") or _metadata_version(found.get("sys_path", []), "vllm")
    where = f"at {script} with Python {python}" if python else f"at {script}"
    if not version:
        report.line("WARN", f"vLLM {where}: couldn't read its version" + ("" if python else
                            " (not a Python script: no #! line to a Python)"))
    elif version.split("+")[0] in wanted:  # a local build's label (+cu130, ...) is the same release
        report.line("PASS", f"vLLM {version} {where} (full runs: {full})")
    else:
        report.line("WARN", f"vLLM {version} {where}, the full runs had {full}: timings compare only on the same "
                            "version")
    package = next(iter(found.get("package") or []), None) or next(
        (str(Path(p, "vllm")) for p in found.get("sys_path", []) if Path(p, "vllm", "__init__.py").is_file()), None)
    if package is None:
        report.line("WARN", "couldn't find vLLM's installed files: whether it supports the models isn't checked")
        return
    _vllm_source(report, models, hub, Path(package), _install_fix(python, found, wanted[0], here, hub), compute_cap)


def _vllm_source(report: Report, models: dict, hub: Path, package: Path, fix: str, compute_cap: str) -> None:
    """Whether vLLM's source names each model's architecture and quantization, and the arguments run_bench.py gives
    it: its reasoning parser and --kv-cache-memory-bytes (vllm_memory.py). Read from the files, not run."""
    needs = {}  # (kind, name, source, text it has to have) -> models that need it
    for model, info in models.items():
        facts = _model_facts(info["repo"], info["revision"], hub)
        if facts is None:
            report.line("WARN", f"{model}: architecture of {info['repo']} unknown, so whether vLLM supports it "
                                "isn't checked")
            continue
        architecture, method, algorithms = facts
        needs.setdefault(("architecture", architecture, REGISTRY, f'"{architecture}"'), []).append(model)
        for algorithm in algorithms:
            source = QUANTIZATION.get(method, f"model_executor/layers/quantization/{method}.py")
            needs.setdefault(("quantization", f"{method} {algorithm}", source, algorithm), []).append(model)
        args = info.get("vllm_args", [])
        for k, arg in enumerate(args):
            if arg.startswith("--reasoning-parser"):
                parser = arg.split("=", 1)[1] if "=" in arg else args[k + 1]
                needs.setdefault(("reasoning parser", parser, REASONING, f'"{parser}"'), []).append(model)
        needs.setdefault(("argument", "--kv-cache-memory-bytes", ENGINE_ARGS, "kv_cache_memory_bytes"), []).append(model)
    has = {}
    for (kind, name, source, text), names in needs.items():
        found = _mentions(package / source, text)
        if found is None:
            report.line("WARN", f"vLLM has no {source}: its {kind} {name} isn't checked")
        elif found:
            has.setdefault(kind, []).append(name)
        else:
            report.line("FAIL", f"this vLLM lacks the {kind} {name} ({source}), needed by "
                                + (", ".join(names) if len(names) < len(models) else "every model"), fix)
    if has:
        report.line("PASS", "vLLM's source has the " + "; ".join(f"{kind} {', '.join(names)}"
                                                                 for kind, names in has.items()))
    report.line("INFO", f"whether vLLM's FP4 kernels run on this GPU (compute capability {compute_cap or 'unknown'}) "
                        "shows only when a model starts (run_bench.py starts each)")


def _tensorrt_llm(report: Report) -> None:
    script = shutil.which("trtllm-serve")
    if script is None:
        report.line("INFO", "TensorRT-LLM: no `trtllm-serve` on PATH (not needed: the package times vLLM)")
        return
    python = _interpreter(script)
    found = _probe(python, "tensorrt_llm", "tensorrt_llm") if python else {}
    version = found.get("version") or _metadata_version(found.get("sys_path", []), "tensorrt_llm")
    report.line("INFO", f"TensorRT-LLM {version or '(version unknown)'} at {script} (not used: the package times vLLM)")


def _memory(report: Report, models: dict, here: Path) -> None:
    """Each model's need against the device's available memory: what run_bench.py reserves for it (the manifest's
    memory_reserved_bytes), else its weights (a lower bound)."""
    total, available, source = _device_memory(here)
    fits = []
    for model, info in models.items():
        need = info.get("memory_reserved_bytes") or info["weights_bytes"]
        what = "what run_bench.py reserves" if info.get("memory_reserved_bytes") else \
            "its weights only, a lower bound: vLLM also needs its KV cache and working memory"
        text = f"memory: {model} needs {_size(need)} ({what}), {_size(available)} of {_size(total)} available ({source})"
        if need > available:
            report.line("FAIL", text, "Free memory for the models that don't fit (`free -h` shows it; run_bench.py "
                                      "runs one model at a time), or leave them out")
        elif available < need * (1 + HEADROOM):
            report.line("WARN", f"{text}: under {HEADROOM:.0%} to spare")
        else:
            fits.append(need)
    if fits:
        basis = "what run_bench.py reserves" if all(i.get("memory_reserved_bytes") for i in models.values()) else \
            "weights only for some, a lower bound"
        others = "every model fits" if len(fits) == len(models) else f"the other {len(fits)} models fit"
        report.line("PASS", f"memory: {_size(available)} of {_size(total)} available ({source}): {others} with "
                            f"{HEADROOM:.0%} to spare (largest need {_size(max(fits))}, {basis})")


def _disk(report: Report, models: dict, hub: Path) -> None:
    """Free space where the Hugging Face cache is (or would be made) against the weights still to download."""
    missing = {}
    for model, info in models.items():
        snapshot = hub / f"models--{info['repo'].replace('/', '--')}" / "snapshots" / info["revision"]
        left = info["weights_bytes"] - _safetensors_bytes(snapshot)
        if left > 0:
            missing[model] = left
    existing = hub
    while not existing.is_dir() and existing != existing.parent:  # the disk it would be made on
        existing = existing.parent
    free = shutil.disk_usage(str(existing)).free
    need = sum(missing.values())
    where = f"{existing} (Hugging Face cache)" if existing == hub else \
        f"{existing} (where the Hugging Face cache {hub} would be)"
    text = (f"disk: {_size(free)} free at {where}; weights to download: {_size(need)} "
            f"({len(missing)} of {len(models)} models)")
    if free < need:
        report.line("FAIL", text, f"Free {_size(need - free)} at {existing}, or put the Hugging Face cache on a disk "
                                  f"with {_size(need)} free:\n    export HF_HUB_CACHE=<directory on that disk>")
    else:
        report.line("PASS", text)


def _port(report: Report, models: dict, base_url: str) -> None:
    """Whether the port run_bench.py uses is free, else what answers there."""
    url = urllib.parse.urlparse(base_url)
    host, port = url.hostname or "127.0.0.1", url.port or 8000
    try:
        socket.create_connection((host, port), timeout=2).close()
    except OSError:
        report.line("PASS", f"port {port} free: run_bench.py starts vLLM there")
        return
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            served = [m["id"] for m in json.load(r)["data"]]
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        report.line("WARN", f"port {port} is taken by something that doesn't answer {base_url}/models: stop it, or give "
                            "run_bench.py --base-url with a free port")
        return
    ours = [m for m, info in models.items() if m in served or info["repo"] in served]
    report.line("WARN", f"a server at {base_url} serves {', '.join(served) or 'no model'}: run_bench.py uses it for "
                        f"{', '.join(ours) or 'none of the models'}; for the others, stop it or give --base-url "
                        "with a free port")


def _install_fix(python: str, found: dict, version: str, here: Path, hub: Path) -> str:
    """The commands that would get the full runs' vLLM: pip into `python` (a dry run first: what it would install or
    replace), or NVIDIA's vLLM container running the package."""
    lines = [f"Get vLLM {version} (the full runs'). Either install it into {python}; the dry run shows what it would "
             "install or replace:",
             f"    {python} -m pip install --dry-run vllm=={version}",
             f"    {python} -m pip install vllm=={version}"]
    if found.get("stdlib") and Path(found["stdlib"], "EXTERNALLY-MANAGED").is_file():
        lines.append(f"  (pip refuses to install into {python}: it is marked as managed by the system's packages)")
    lines += [f"Or run the package in NVIDIA's vLLM container, with the tag of nvcr.io/nvidia/vllm for this device's "
              f"JetPack/CUDA ({NGC_VLLM})" + ("" if shutil.which("docker") else "; there is no `docker` on PATH") + ":",
              f"    docker run --rm -it --runtime nvidia --network host --ipc host -v {here}:{here} -w {here} "
              f"-v {hub}:{hub} -e HF_HUB_CACHE={hub} nvcr.io/nvidia/vllm:<tag> python3 run_bench.py --model <model>"]
    return "\n".join(lines)


def _model_facts(repo: str, revision: str, hub: Path):
    """(architecture, quantization method, its algorithms) from the model's config.json if it is in the Hugging Face
    cache, else KNOWN_MODELS; None if neither knows it."""
    config = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / "config.json"
    try:
        c = json.loads(config.read_text())
        q = c.get("quantization_config") or {}
        if q.get("quant_method") == "modelopt":  # MIXED_PRECISION: each layer's algorithm too
            layers = sorted({layer.get("quant_algo") for layer in (q.get("quantized_layers") or {}).values()} - {None})
            algorithms = [q.get("quant_algo")] + layers
        else:  # compressed-tensors: its format, e.g. nvfp4-pack-quantized
            algorithms = ["nvfp4" if "nvfp4" in q.get("format", "") else q.get("format")]
        return c["architectures"][0], q.get("quant_method"), tuple(a for a in algorithms if a)
    except (OSError, ValueError, KeyError, IndexError, AttributeError):
        pass
    name = repo.split("/")[-1]
    return next((facts for prefix, facts in KNOWN_MODELS.items() if name.startswith(prefix)), None)


def _mentions(source: Path, text: str):
    """Whether a file of vLLM's source (or any .py file of a directory) has `text`; None if it isn't there."""
    files = sorted(source.rglob("*.py")) if source.is_dir() else [source] if source.is_file() else []
    if not files:
        return None
    return any(text in f.read_text(errors="replace") for f in files)


def _interpreter(script: str):
    """The Python a console script runs with: its #! line (also through `env`, and pip's /bin/sh launcher for long
    paths), else a python3 next to it (a venv's bin/); None if neither."""
    try:
        with open(script, "rb") as f:
            lines = f.read(2048).decode(errors="replace").splitlines()
    except OSError:
        lines = []
    words = lines[0][2:].split() if lines and lines[0].startswith("#!") else []
    if words and Path(words[0]).name == "env":  # #!/usr/bin/env python3
        words = [w for w in words[1:] if not w.startswith("-") and "=" not in w]
        words = [shutil.which(words[0]) or ""] if words else []
    elif words and Path(words[0]).name in ("sh", "bash"):  # '''exec' "/path/to/python" "$0" "$@"
        match = next((re.match(r"'''exec' \"?([^\" ]+)", line) for line in lines[1:3]
                      if line.startswith("'''exec'")), None)
        words = [match.group(1)] if match else []
    if words and os.path.isfile(words[0]) and os.access(words[0], os.X_OK):
        return words[0]
    for name in ("python3", "python"):
        sibling = Path(os.path.realpath(script)).parent / name
        if sibling.is_file() and os.access(str(sibling), os.X_OK):
            return str(sibling)
    return None


def _probe(python: str, dist: str, package: str) -> dict:
    """What `python` knows of a distribution, without importing it (PROBE); {} if it can't be run."""
    out = _run([python, "-B", "-c", PROBE, dist, package], timeout=60)
    try:
        return json.loads(out.strip().splitlines()[-1]) if out and out.strip() else {}
    except ValueError:
        return {}


def _metadata_version(paths: list, dist: str):
    """A distribution's version from its dist-info METADATA in one of `paths` (an interpreter's sys.path)."""
    for entry in paths:
        try:
            for metadata in sorted(Path(entry).glob(f"{dist}-*.dist-info/METADATA")):
                for line in metadata.read_text(errors="replace").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            continue
    return None


def _device_memory(here: Path) -> tuple:
    """(total, available, source) in bytes: vllm_memory.device_memory, as run_bench.py measures it (the GPU's own
    memory, or the system's where the GPU uses it), else /proc/meminfo."""
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        import vllm_memory  # shipped next to this file
        total, free, source = vllm_memory.device_memory(gpu)
        return total, free, source
    except Exception:  # not next to this file, or changed: the system's memory
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return int(info["MemTotal"].split()[0]) * 1024, int(info["MemAvailable"].split()[0]) * 1024, "/proc/meminfo"


def _hub_cache() -> Path:
    """The Hugging Face hub cache, as huggingface_hub (and so vLLM) finds it."""
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache", "huggingface")
        hub = os.path.join(home, "hub")
    return Path(os.path.abspath(os.path.expanduser(hub)))


def _size(n: int) -> str:
    return f"{n / GiB:.1f} GiB" if n < 1024 * GiB else f"{n / (1024 * GiB):.1f} TiB"


def _safetensors_bytes(snapshot: Path) -> int:
    total = 0
    for f in snapshot.glob("*.safetensors"):
        try:
            total += f.stat().st_size  # through the cache's link to its blob
        except OSError:
            pass
    return total


def _smi(field: str, gpu: str) -> str:
    """A field of nvidia-smi's --query-gpu; "" if it has none (e.g. [N/A] on Jetson)."""
    out = _run(["nvidia-smi", "-i", gpu, f"--query-gpu={field}", "--format=csv,noheader"])
    value = out.strip().splitlines()[0].strip() if out and out.strip() else ""
    return "" if "N/A" in value or "Not Supported" in value else value


def _run(command: list, timeout: float = 30):
    """A command's output; None if it is missing or fails. Run from / with vLLM's usage stats off."""
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=ENV, cwd="/")
        return run.stdout if run.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    sys.exit(0 if check(json.loads((HERE / "manifest.json").read_text()), HERE) else 1)
