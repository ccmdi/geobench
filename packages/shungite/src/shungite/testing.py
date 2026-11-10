import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

from aiohttp import web


@dataclass(frozen=True)
class ToolCall:
    finish_reason: ClassVar[str] = "tool_calls"

    name: str
    arguments: dict[str, Any]

    def delta(self, call_id: str) -> dict[str, Any]:
        function = {"name": self.name, "arguments": json.dumps(self.arguments)}
        return {"role": "assistant", "tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": function}]}


@dataclass(frozen=True)
class Reply:
    finish_reason: ClassVar[str] = "stop"

    text: str

    def delta(self, call_id: str) -> dict[str, Any]:
        return {"role": "assistant", "content": self.text}


Turn = ToolCall | Reply
Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def serving(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
    finally:
        await runner.cleanup()


@dataclass
class FakeModel:
    turns: list[Turn]
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        turn = self.turns[len(self.requests) - 1]
        base = {"id": f"fake-{len(self.requests)}", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        chunks = [
            base | {"choices": [{"index": 0, "delta": turn.delta(f"call_{len(self.requests)}"), "finish_reason": None}]},
            base | {"choices": [{"index": 0, "delta": {}, "finish_reason": turn.finish_reason}]},
            base | {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    def serving(self) -> AbstractAsyncContextManager[str]:
        return serving(self.complete)


@dataclass
class FakeJudge:
    reply: Callable[[str], str]
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompts(self) -> list[str]:
        return [request["messages"][-1]["content"] for request in self.requests]

    async def complete(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        message = {"role": "assistant", "content": self.reply(body["messages"][-1]["content"])}
        return web.json_response(
            {
                "id": f"judge-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def serving(self) -> AbstractAsyncContextManager[str]:
        return serving(self.complete)
