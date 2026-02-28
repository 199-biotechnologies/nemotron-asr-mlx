#!/usr/bin/env python3
"""Streaming transcription demo — WebSocket server.

Run:
    python demo/server.py

Then open http://localhost:8765 in your browser.
"""
import asyncio
import json
import time
from pathlib import Path

import numpy as np

# Lazy model loading
_model = None


def get_model():
    global _model
    if _model is None:
        import nemotron_asr_mlx as nm
        print("Loading model...")
        t0 = time.time()
        # Use local model if available, otherwise download from HuggingFace
        import os
        local_path = os.path.expanduser("~/Models/nemotron-asr-mlx")
        if os.path.isdir(local_path):
            _model = nm.from_pretrained(local_path)
        else:
            _model = nm.from_pretrained("199-biotechnologies/nemotron-asr-mlx")
        print(f"Model loaded in {time.time() - t0:.2f}s")
    return _model


async def handler(websocket):
    """Handle a single WebSocket connection."""
    model = get_model()
    pcm_buffer = np.zeros(0, dtype=np.float32)
    last_text = ""
    chunk_count = 0

    # Transcription interval: run every N chunks to avoid too-frequent inference
    TRANSCRIBE_INTERVAL = 5  # every ~0.5s at 100ms chunks

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Decode float32 PCM from browser
                chunk = np.frombuffer(message, dtype=np.float32)
                pcm_buffer = np.concatenate([pcm_buffer, chunk])
                chunk_count += 1

                # Transcribe periodically
                if chunk_count % TRANSCRIBE_INTERVAL == 0 and len(pcm_buffer) > 4800:
                    t0 = time.time()
                    result = model.transcribe(pcm_buffer)
                    elapsed = time.time() - t0
                    duration = len(pcm_buffer) / 16000

                    new_text = result.text
                    if new_text != last_text:
                        await websocket.send(json.dumps({
                            "type": "transcript",
                            "text": new_text,
                            "tokens": len(result.tokens),
                            "duration": round(duration, 1),
                            "inference_ms": round(elapsed * 1000),
                            "rtfx": round(duration / elapsed, 1) if elapsed > 0 else 0,
                        }))
                        last_text = new_text

            elif isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "stop":
                    # Final transcription
                    if len(pcm_buffer) > 1600:
                        result = model.transcribe(pcm_buffer)
                        await websocket.send(json.dumps({
                            "type": "final",
                            "text": result.text,
                            "tokens": len(result.tokens),
                            "duration": round(len(pcm_buffer) / 16000, 1),
                        }))
                    pcm_buffer = np.zeros(0, dtype=np.float32)
                    last_text = ""
                    chunk_count = 0

                elif data.get("type") == "clear":
                    pcm_buffer = np.zeros(0, dtype=np.float32)
                    last_text = ""
                    chunk_count = 0
                    await websocket.send(json.dumps({"type": "cleared"}))

                elif data.get("type") == "config":
                    interval = data.get("interval", 5)
                    TRANSCRIBE_INTERVAL = max(1, min(20, interval))
                    await websocket.send(json.dumps({
                        "type": "config_ack",
                        "interval": TRANSCRIBE_INTERVAL,
                    }))

    except Exception as e:
        print(f"Connection error: {e}")


def serve_html(connection, request):
    """Serve the HTML page on HTTP GET requests."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    if "Upgrade" not in request.headers:
        html_path = Path(__file__).parent / "index.html"
        body = html_path.read_bytes()
        headers = Headers([("Content-Type", "text/html; charset=utf-8")])
        return Response(200, "OK", headers, body)


async def main():
    import websockets

    # Pre-load model
    get_model()

    print("\n  Demo running at http://localhost:8765\n")

    async with websockets.serve(
        handler,
        "localhost",
        8765,
        process_request=serve_html,
        max_size=2**22,  # 4MB max message
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
