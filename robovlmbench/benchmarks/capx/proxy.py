"""A local OpenAI-style endpoint that CaP-X sends its model calls to, in front of the model's server.

CaP-X asks with its own sampling (temperature, max_tokens). The proxy sends each call with the model's own
request settings instead, the way robovlmbench/models/client.py does: system prompt before CaP-X's, prompt
suffix after its last text, sampling and extra_body (e.g. chat_template_kwargs) as configured, every image
as new. It answers CaP-X with the final answer only (reasoning removed, "" if cut off at max_tokens), and
logs every call with its time to a JSON-lines file.

    with Proxy(server.url, server.model, config.request_settings(), log_path, max_seq_len=20480) as proxy:
        proxy.endpoint(token, trial, attempt, "coder")  # -> URL for CaP-X's --server-url

The path of each URL names the episode (a token), its trial number, the attempt and the role (only "coder":
the code-writing calls), so every logged call can be traced to its episode.
With seed_by_trial, the model's seed is offset by the trial number: CaP-X's trials of a config differ only in
the simulator's initial state, so with one fixed seed the model would write the same code for every trial.
It is also offset by the attempt (1000 per attempt): CaP-X runs a trial again after a timeout, and the same
seed would bring back the same code, and the same timeout.
Calls are sent one at a time (as for the other benchmarks): `latency_s` is the time to answer one call
alone, `wait_s` the time it waited for the calls before it.

Two CaP-X behaviors are handled here:
- CaP-X gives up on a call after 200 s without a byte back. The proxy answers 200 right away and sends
  JSON-insignificant spaces while the model works, so long answers (reasoning) don't time out.
- When the prompt plus the model's max_tokens exceed the server's context (long multi-turn prompts), the
  call is sent with max_tokens lowered to the room left (logged); a prompt longer than the whole context
  gets an empty answer, like a model that wrote no code.
"""

import base64
import concurrent.futures
import copy
import datetime
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
from openai import OpenAI

from robovlmbench.models.client import split_answer

_KEEPALIVE_S = 30  # well under CaP-X's 200 s read timeout
_CONTEXT_ERROR = "maximum context length"  # vLLM's message when prompt + max_tokens don't fit


class Proxy:
    def __init__(self, url: str, model: str, settings: dict, log_path, *, max_seq_len: int,
                 seed_by_trial: bool = False, max_concurrency: int = 1, timeout: float = 1800, retries: int = 3):
        """`url`, `model`: the model's server (OpenAI-style base URL) and served name.
        `settings`: ModelConfig.request_settings(). `max_seq_len`: the server's context length."""
        settings = dict(settings)
        self.system_prompt = settings.pop("system_prompt", None)
        self.prompt_suffix = settings.pop("prompt_suffix", None)
        self.extra_body = settings.pop("extra_body", None)
        self.sampling = settings  # passed straight to chat.completions.create, as by Client
        self.model = model
        self.max_seq_len = max_seq_len
        self.seed_by_trial = seed_by_trial
        self.log_path = log_path
        self.server_errors = 0  # calls that failed for reasons other than the model's prompt
        self._client = OpenAI(base_url=url, api_key="none", timeout=timeout, max_retries=retries)
        self._slots = threading.Semaphore(max_concurrency)
        self._log_lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._http.daemon_threads = True
        self._http.proxy = self
        self.url = f"http://127.0.0.1:{self._http.server_address[1]}"

    def endpoint(self, token: str, trial: int, attempt: int, role: str) -> str:
        return f"{self.url}/{token}/{trial}/{attempt}/{role}/chat/completions"

    def warm_up(self) -> None:
        """One short call, not logged: a new server's first request can be slower."""
        self._send([{"role": "user", "content": "Write a Python comment."}], {**self.sampling, "max_tokens": 16})

    def __enter__(self):
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._http.shutdown()
        self._http.server_close()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def complete(self, token: str, trial: int, attempt: int, role: str, messages: list[dict]) -> dict:
        """Answer one CaP-X call (its `messages`) with the model; log it; return an OpenAI-style response."""
        messages = self._prepare(messages)
        sampling = dict(self.sampling)
        if self.seed_by_trial and sampling.get("seed") is not None:
            sampling["seed"] += trial + 1000 * attempt  # a retry after a timeout samples new code
        queued = time.monotonic()
        with self._slots:
            started = time.monotonic()
            completion, error, server_error = None, None, False
            try:
                completion, sampling["max_tokens"] = self._create(messages, sampling)
            except openai.BadRequestError as e:
                error = f"{type(e).__name__}: {e}"
                server_error = _CONTEXT_ERROR not in str(e)  # else: the prompt alone is longer than the context
            except Exception as e:  # after the client's own retries
                error, server_error = f"{type(e).__name__}: {e}", True
            latency = time.monotonic() - started

        answer, reasoning, finish_reason, usage = "", None, "error", None
        if completion is not None:
            choice = completion.choices[0]
            answer, reasoning = split_answer(choice)
            finish_reason, usage = choice.finish_reason, completion.usage
        self._log({
            "token": token, "trial": trial, "attempt": attempt, "role": role,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "wait_s": round(started - queued, 3), "latency_s": round(latency, 3),
            "request": {"model": self.model, "messages": _without_image_data(messages),
                        "sampling": sampling, "extra_body": self.extra_body},
            "text": completion.choices[0].message.content if completion else None,
            "reasoning": reasoning, "answer": answer, "finish_reason": finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "max_tokens_lowered": sampling.get("max_tokens") != self.sampling.get("max_tokens"),
            "error": error, "server_error": server_error,
        })
        return {
            "id": completion.id if completion else f"error-{uuid.uuid4().hex}", "object": "chat.completion",
            "model": self.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer, "reasoning": reasoning},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": usage.prompt_tokens if usage else 0,
                      "completion_tokens": usage.completion_tokens if usage else 0},
        }

    def _prepare(self, messages: list[dict]) -> list[dict]:
        """CaP-X's messages with the model's system prompt and prompt suffix added, as Client adds them."""
        messages = copy.deepcopy(messages)
        if self.system_prompt:
            if messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str):
                messages[0]["content"] = f"{self.system_prompt}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        if self.prompt_suffix:
            last = messages[-1]
            if isinstance(last["content"], str):
                last["content"] = f"{last['content']}\n\n{self.prompt_suffix}"
            else:
                texts = [part for part in last["content"] if part.get("type") == "text"]
                if texts:
                    texts[-1]["text"] = f"{texts[-1]['text']}\n\n{self.prompt_suffix}"
                else:
                    last["content"].append({"type": "text", "text": self.prompt_suffix})
        return messages

    def _create(self, messages: list[dict], sampling: dict):
        """The completion, and the max_tokens it was asked with: the model's own, or what is left of the
        server's context after the prompt, if less (the prompt's length is found by asking for one token)."""
        sampling = dict(sampling)
        try:
            return self._send(messages, sampling), sampling.get("max_tokens")
        except openai.BadRequestError as e:
            if _CONTEXT_ERROR not in str(e):
                raise
        probe = self._send(messages, {**sampling, "max_tokens": 1})  # raises if the prompt alone doesn't fit
        sampling["max_tokens"] = self.max_seq_len - probe.usage.prompt_tokens
        return self._send(messages, sampling), sampling["max_tokens"]

    def _send(self, messages: list[dict], sampling: dict):
        messages = copy.deepcopy(messages)
        for message in messages:  # every image as new, so the server can't reuse work on an earlier call
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        part["uuid"] = uuid.uuid4().hex
        return self._client.chat.completions.create(model=self.model, messages=messages,
                                                    extra_body=self.extra_body, **sampling)

    def _log(self, record: dict) -> None:
        with self._log_lock, open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            self.server_errors += record["server_error"]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        proxy = self.server.proxy
        parts = self.path.strip("/").split("/")  # <token>/<trial>/<attempt>/<role>/chat/completions
        if len(parts) != 6 or parts[4:] != ["chat", "completions"] or parts[3] != "coder":
            self.send_error(400, f"unexpected path {self.path}")  # not 404: CaP-X retries those forever
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()  # the body follows when the model is done; the connection's end marks its end
        future = proxy._pool.submit(proxy.complete, parts[0], int(parts[1]), int(parts[2]), parts[3], body["messages"])
        try:
            while True:
                try:
                    response = future.result(timeout=_KEEPALIVE_S)
                    break
                except concurrent.futures.TimeoutError:
                    self.wfile.write(b" ")
                    self.wfile.flush()
            self.wfile.write(json.dumps(response).encode())
        except OSError:  # CaP-X hung up (e.g. its trial timed out); the call is logged anyway
            pass

    def log_message(self, *args):  # no line per request on the console
        pass


def _without_image_data(messages: list[dict]) -> list[dict]:
    """`messages` with each image's data replaced by its type and size, to keep the log small."""
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    head, _, data = url.partition(",")
                    if head.startswith("data:"):
                        size = len(base64.b64decode(data))
                        part["image_url"]["url"] = f"{head.removesuffix(';base64')}, {size} bytes (not logged)"
    return messages
