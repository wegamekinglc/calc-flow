"""Serve two sample orders on loopback for example 21 (requires websockets)."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve


async def send_orders(connection: ServerConnection) -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "orders.jsonl"
    payload = await asyncio.to_thread(path.read_text, encoding="utf-8")
    for line in payload.splitlines():
        await connection.send(line)
    await connection.wait_closed()


async def main() -> None:
    async with serve(send_orders, "127.0.0.1", 8765) as server:
        print("Sample orders: ws://127.0.0.1:8765 (Ctrl-C to stop)", flush=True)
        await server.serve_forever()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
