import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web


@dataclass
class FakeModel:
    replies: list[str]
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        delta = {"role": "assistant", "content": self.replies[len(self.requests) - 1]}
        base = {"id": f"fake-{len(self.requests)}", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        chunks = [
            base | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            base | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            base | {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[str]:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
        finally:
            await runner.cleanup()
