"""Send image + text requests to an OpenAI-style server: vLLM, TensorRT-LLM or a hosted API.

    client = Client(server.url, server.model, config.request_settings())
    responses = client.ask([Request("q1", "Is the mug left of the laptop?", [png_bytes])])
    responses[0].answer  # final answer: reasoning removed, "" if cut off at max_tokens

Requests are sent concurrently so the server can batch them; `latency_s` therefore includes
time queued at the server. With max_concurrency=1 it is the time to answer one request alone.
"""

import asyncio
import base64
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

from openai import AsyncOpenAI

# Markers models use around reasoning, for servers that return it inline instead of separately.
_REASONING_ENDS = ("</think>", "[/THINK]")


@dataclass
class Request:
    id: str
    text: str
    images: list[bytes]  # encoded image files (PNG or JPEG)
    images_first: bool = True  # images before the text, or after it; ignored if the text places them
    system: str | None = None  # the benchmark's system message, after the model's own system prompt if any

    IMAGE_MARKER = "<image>"  # one per image in `text` places each image there instead


@dataclass
class Response:
    id: str
    answer: str            # final answer: reasoning removed; "" if truncated or failed
    text: str              # raw output as returned by the server
    reasoning: str | None  # reasoning, if any
    finish_reason: str | None  # "stop", or "length" if cut off at max_tokens
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None   # set if the request failed after retries

    def to_dict(self) -> dict:
        return asdict(self)


class Client:
    def __init__(self, url: str, model: str, settings: dict | None = None, *,
                 api_key: str = "none", max_concurrency: int = 32, timeout: float = 1800, retries: int = 3,
                 fresh_images: bool = False):
        """`settings`: ModelConfig.request_settings() — temperature, max_tokens, system_prompt, extra_body, ...
        `fresh_images`: give every image sent a new id (vLLM's per-image `uuid`), so the server can't
        reuse its work on an earlier request with the same image."""
        settings = dict(settings or {})
        self.system_prompt = settings.pop("system_prompt", None)
        self.prompt_suffix = settings.pop("prompt_suffix", None)  # appended to every question, e.g. to ask for reasoning
        self.extra_body = settings.pop("extra_body", None)
        self.sampling = settings  # passed straight to chat.completions.create
        self.model = model
        self.max_concurrency = max_concurrency
        self.fresh_images = fresh_images
        self._client_args = dict(base_url=url, api_key=api_key, timeout=timeout, max_retries=retries)

    def ask(self, requests: Iterable[Request],
            on_response: Callable[[Response], None] | None = None) -> list[Response]:
        """Send all requests; return responses in request order.

        `on_response` is called as each one arrives (e.g. to save it), in completion order.
        """
        return asyncio.run(self._ask_all(list(requests), on_response))

    async def _ask_all(self, requests: list[Request], on_response) -> list[Response]:
        client = AsyncOpenAI(**self._client_args)
        limit = asyncio.Semaphore(self.max_concurrency)

        async def one(request: Request) -> Response:
            async with limit:
                response = await self._ask_one(client, request)
            if on_response:
                on_response(response)
            return response

        try:
            return await asyncio.gather(*(one(r) for r in requests))
        finally:
            await client.close()

    async def _ask_one(self, client: AsyncOpenAI, request: Request) -> Response:
        messages = self.messages(request)
        start = time.monotonic()
        try:
            completion = await client.chat.completions.create(
                model=self.model, messages=messages, extra_body=self.extra_body, **self.sampling)
        except Exception as e:  # after the client's own retries
            return Response(request.id, "", "", None, None, time.monotonic() - start, 0, 0,
                            error=f"{type(e).__name__}: {e}")
        latency = time.monotonic() - start

        choice = completion.choices[0]
        text = choice.message.content or ""
        answer, reasoning = split_answer(choice)
        usage = completion.usage
        return Response(request.id, answer, text, reasoning, choice.finish_reason, latency,
                        usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)

    def messages(self, request: Request) -> list[dict]:
        """The chat messages sent for `request`: the model's system prompt and prompt suffix added, images
        as data URLs where the text places them (or before/after it)."""
        images = [{"type": "image_url", "image_url": {"url": _data_url(image)}} for image in request.images]
        if self.fresh_images:
            for part in images:
                part["uuid"] = uuid.uuid4().hex
        text = f"{request.text}\n\n{self.prompt_suffix}" if self.prompt_suffix else request.text
        texts = text.split(Request.IMAGE_MARKER)
        if len(texts) == len(images) + 1 and images:  # images where the text places them
            content = [texts[0], *(part for image, t in zip(images, texts[1:]) for part in (image, t))]
            content = [{"type": "text", "text": c} if isinstance(c, str) else c for c in content if c != ""]
        else:
            part = {"type": "text", "text": text}
            content = [*images, part] if request.images_first else [part, *images]
        messages = [{"role": "user", "content": content}]
        system = "\n\n".join(s for s in (self.system_prompt, request.system) if s)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages


def split_answer(choice) -> tuple[str, str | None]:
    """(final answer, reasoning) of a chat completion choice: the answer has the reasoning removed, and is ""
    if cut off at max_tokens."""
    text = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None)
    answer = text
    if reasoning is None:  # server didn't separate it; split on the model's marker
        for marker in _REASONING_ENDS:
            if marker in text:
                reasoning, answer = text.rsplit(marker, 1)
                break
    if choice.finish_reason == "length":
        answer = ""  # cut off: whatever is there is unfinished, not an answer
    return answer.strip(), reasoning


def _data_url(image: bytes) -> str:
    if image.startswith(b"\x89PNG"):
        mime = "image/png"
    elif image.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        raise ValueError("image must be PNG or JPEG bytes")
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"
