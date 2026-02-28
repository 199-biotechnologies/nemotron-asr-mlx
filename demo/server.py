#!/usr/bin/env python3
"""Streaming transcription demo — terminal mic + browser display.

Run:
    python demo/server.py

Then open http://localhost:8765 in your browser.
Mic is captured in the terminal (via sounddevice). Browser only shows output.
"""
import asyncio
import json
import os
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

# Lazy model loading
_model = None


def get_model():
    global _model
    if _model is None:
        import nemotron_asr_mlx as nm
        print("Loading model...")
        t0 = time.time()
        local_path = os.path.expanduser("~/Models/nemotron-asr-mlx")
        if os.path.isdir(local_path):
            _model = nm.from_pretrained(local_path)
        else:
            _model = nm.from_pretrained("199-biotechnologies/nemotron-asr-mlx")
        print(f"Model loaded in {time.time() - t0:.2f}s")
    return _model


# Shared state between mic thread and websocket handler
class MicState:
    def __init__(self):
        self.pcm_buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.recording = False
        self.stream = None
        self.last_text = ""
        self.chunk_count = 0

    def start(self):
        self.recording = True
        self.pcm_buffer = np.zeros(0, dtype=np.float32)
        self.last_text = ""
        self.chunk_count = 0
        self.stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32",
            blocksize=1600,  # 100ms chunks
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def clear(self):
        self.stop()
        with self.lock:
            self.pcm_buffer = np.zeros(0, dtype=np.float32)
            self.last_text = ""
            self.chunk_count = 0

    def _callback(self, indata, frames, time_info, status):
        if not self.recording:
            return
        with self.lock:
            self.pcm_buffer = np.concatenate([self.pcm_buffer, indata[:, 0]])
            self.chunk_count += 1

    def get_audio(self):
        with self.lock:
            return self.pcm_buffer.copy(), self.chunk_count


mic = MicState()
clients = set()


async def handler(websocket):
    """Handle a single WebSocket connection."""
    clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "start":
                mic.start()
                print("  Recording started")

            elif msg_type == "stop":
                mic.stop()
                print("  Recording stopped — running final transcription...")
                audio, _ = mic.get_audio()
                if len(audio) > 1600:
                    model = get_model()
                    result = model.transcribe(audio)
                    await broadcast({
                        "type": "final",
                        "text": result.text,
                        "tokens": len(result.tokens),
                        "duration": round(len(audio) / 16000, 1),
                    })

            elif msg_type == "clear":
                mic.clear()
                await broadcast({"type": "cleared"})

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        clients.discard(websocket)


async def broadcast(msg):
    """Send a message to all connected clients."""
    data = json.dumps(msg)
    for ws in list(clients):
        try:
            await ws.send(data)
        except Exception:
            clients.discard(ws)


async def transcription_loop():
    """Periodically transcribe accumulated audio and broadcast results."""
    model = get_model()
    transcribe_interval = 5  # every 5 chunks (~500ms)

    while True:
        await asyncio.sleep(0.2)

        if not mic.recording:
            continue

        audio, count = mic.get_audio()
        if count % transcribe_interval != 0 or len(audio) < 4800:
            continue

        t0 = time.time()
        result = model.transcribe(audio)
        elapsed = time.time() - t0
        duration = len(audio) / 16000

        new_text = result.text
        if new_text != mic.last_text:
            mic.last_text = new_text
            await broadcast({
                "type": "transcript",
                "text": new_text,
                "tokens": len(result.tokens),
                "duration": round(duration, 1),
                "inference_ms": round(elapsed * 1000),
                "rtfx": round(duration / elapsed, 1) if elapsed > 0 else 0,
            })


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

    print("\n  Demo running at http://localhost:8765")
    print("  Mic is captured here in the terminal.")
    print("  Open the browser to see live transcription.\n")

    # Start transcription background task
    asyncio.create_task(transcription_loop())

    async with websockets.serve(
        handler,
        "localhost",
        8765,
        process_request=serve_html,
        max_size=2**22,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
