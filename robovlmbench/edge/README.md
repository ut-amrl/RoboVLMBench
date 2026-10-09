# RoboVLMBench edge subset

A small, fixed set of model requests for timing vision-language models on an edge device, and the memory they
take. The requests are taken from RoboVLMBench's full runs of six benchmarks: RoboSpatial-Home, InstructPart,
RoboBench, EPIC-Bench, OVDEval, and CaP-X's code-writing calls. Every capability of every benchmark is covered,
and the requests within each one match the full runs' mean prompt length, answer length and score, for every
model. The set is sized for the slowest model to take about 10 minutes on an NVIDIA Jetson AGX Thor, at the
answer lengths of the full runs. Every request may use 16,384 tokens (prompt and answer: its `max_tokens` is
16,384 minus its prompt), so a model that answers at length here takes longer. Only inference is timed: no
benchmark data, scoring code or robot simulator is needed.

Models are served with the device's own vLLM: nothing is installed, and nothing is written outside this folder
except the model weights vLLM downloads (Hugging Face cache, about 44 GiB for all six models).

## Contents

| Path | What |
|---|---|
| `requests/<model>.jsonl` | One line per request, as the full run sent it (messages, sampling settings with the seed, extra request fields), but with `max_tokens` filling 16,384 tokens |
| `media/` | The images the requests refer to (`media/<hash>.png` or `.jpg`) |
| `manifest.json` | Per request: benchmark, capability (`stratum`), weight, and the full run's tokens and time per model. Per model: Hugging Face repo and revision, the server's context (`max_model_len`: 16,384 plus 256 tokens should this device's vLLM count image tokens differently), vLLM arguments, the full run's vLLM version, memory measured on the full run's hardware, and the full run's totals per benchmark |
| `run_bench.py` | The client: Python 3.8+, standard library only |
| `check_system.py` | Read-only check of the device (used by `run_bench.py --check-system`) |
| `vllm_memory.py` | How much memory a vLLM server gets (used by `run_bench.py`) |

## Run

```bash
python3 run_bench.py --check            # every request and image file in place
python3 run_bench.py --check-system     # the device: vLLM, GPU, memory, disk (reads only; prints what to fix)
python3 run_bench.py --model qwen3.8-27b-nvfp4_nothink   # one model; repeat for each in manifest.json
```

Send back the `results/` folder: per model the timings, the memory and the server's log.

The server:

- If a server at `--base-url` (default `http://127.0.0.1:8000/v1`) serves the model (as `--served-model-name`,
  default `--model`, or as its repo), it is used and left running. If it serves another model, the script
  stops and says so.
- Otherwise the system's `vllm` (the one on `PATH`) is started on that port with the memory below, and stopped
  when the timing is done. Its first start of a model downloads the weights; the start is given up only if
  the server's log stops growing for 15 minutes. vLLM's usage statistics are turned off.

The package is made for vLLM (the full runs used vLLM 0.31.0; the same version keeps timings comparable). Other
OpenAI-compatible servers already running can be used, but may reject vLLM's request fields.

## Memory

What a model needs to answer requests, and so what a server started here takes:

| Part | Depends on | E.g. Qwen3.8-27B |
|---|---|---|
| Weights: the model itself | model size | 19.9 GiB |
| KV cache: the attention memory of the request being answered | request length (one request at a time) | 1.4 GiB for 16,640 tokens |
| Working memory: activations, CUDA graphs | model, vLLM, requests | ~2.4 GiB |

A server started here reserves 1.5 times the model's need (`memory_reserve` in the manifest); vLLM holds all of
it, what's over the need as more KV cache (`--kv-cache-memory-bytes`). `--gpu-memory-utilization` is only vLLM's
startup check that this much memory is free: it refuses to start otherwise. The need is the most memory the
server held while answering requests: at first as measured on the full runs' hardware (`memory_measured` in
the manifest), then as this device measured it on its last timing (`results/vllm_memory_needs.json`). If vLLM
still finds the KV cache short for the longest request, it says by how much, and the server is started again
with that added.

## Timing

Requests go one at a time, streamed, after one warm-up request. Re-running sends only requests not yet timed.
Each image is sent with a new id (vLLM's per-image `uuid`) so the server can't reuse an earlier request's image
work, as in the full runs; `--no-fresh-images` turns this off for servers that don't accept the field.

## Results

`results/<model>/requests.jsonl` has one line per request: time to the first token (`ttft_s`), total time
(`latency_s`), prompt and output tokens, finish reason, error.

`results/<model>/memory.json` (also in the summary): the device's total memory and its free memory before the
server started, after it loaded and at its lowest during the timing (sampled twice a second; on a GPU with its
own memory as `nvidia-smi` reports it, on a GPU sharing the system's memory from `/proc/meminfo`, where other
programs' use shows too). For a server started here: what it took when loaded and at its peak, what its
processes hold (where `nvidia-smi` reports it), and vLLM's own breakdown (weights, KV cache, CUDA graphs). A
server that was already running can't be told apart from other programs.

`results/<model>/summary.json`:

- `decode_tok_s`: output tokens per second after the first token, over all requests.
- `per_benchmark.<benchmark>.subset`: weighted means over the subset's requests (each stands for a share of its
  benchmark), with `full_run_completion_tokens` to compare answer lengths with the full run's.
- `per_benchmark.<benchmark>.full_benchmark_estimate`: the estimated mean time per request (`mean_s`) and
  total time (`total_h`) of the whole benchmark on this device, and the slowdown against the full run's
  hardware (`full_run_hardware`). It adds the full run's output tokens after the first at this device's decode
  speed and the full run's time to the first token scaled by this device's slowdown on it, measured per
  benchmark on the subset. On a simulated device this estimate was within a few percent per benchmark
  (`manifest.json` → `validation_relative_error_simulated_device`).

Answers can differ in length from the full run's: sampling follows the same seed, but a different GPU computes
slightly different numbers. The estimate uses the full run's answer lengths, so it isn't affected; the subset's
own times are.

## Cleanup

Everything the client writes is under `results/` in this folder, and a server it started is stopped before it
exits. Removing this folder and the model weights vLLM downloaded (Hugging Face cache) removes everything.
