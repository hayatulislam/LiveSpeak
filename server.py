import asyncio
import json
import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from deepgram import AsyncDeepgramClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("livespeak")

PORT = 3000

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("FATAL ERROR: DEEPGRAM_API_KEY is not set in your .env file.")

deepgram_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

app = FastAPI()

viewers: set[WebSocket] = set()
viewers_lock = asyncio.Lock()


async def broadcast_to_viewers(text: str, is_final: bool) -> None:
    payload = json.dumps({"text": text, "isFinal": is_final})
    logger.info(f"[BROADCAST SUCCESS] Sending to viewers: {payload}")
    dead = []
    async with viewers_lock:
        for viewer in viewers:
            try:
                await viewer.send_text(payload)
            except Exception:
                dead.append(viewer)
        for viewer in dead:
            viewers.discard(viewer)


@app.websocket("/speak")
async def speak_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("A Speaker connected.")

    try:
        async with deepgram_client.listen.v1.connect(
            model="nova-2",
            language="en-US",
            interim_results=True,
            smart_format=True,
            punctuate=True,
        ) as dg_connection:

            async def forward_audio() -> None:
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        logger.info(f"[SERVER] Forwarding audio chunk of size: {len(data)} bytes.")
                        await dg_connection.send_media(data)
                except WebSocketDisconnect:
                    logger.info("Speaker disconnected.")
                except Exception as exc:
                    logger.error(f"Error forwarding audio: {exc}")

            async def receive_transcripts() -> None:
                try:
                    async for result in dg_connection:
                        if not hasattr(result, "channel"):
                            continue
                        transcript = result.channel.alternatives[0].transcript
                        if transcript:
                            await broadcast_to_viewers(transcript, bool(result.is_final))
                except Exception as exc:
                    logger.error(f"[DEEPGRAM] Error: {exc}")

            forward_task = asyncio.create_task(forward_audio())
            listen_task = asyncio.create_task(receive_transcripts())

            done, pending = await asyncio.wait(
                {forward_task, listen_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            try:
                await dg_connection.send_close_stream()
            except Exception:
                pass

            logger.info("[DEEPGRAM] Connection closed.")

    except Exception as exc:
        logger.error(f"FAILED to create Deepgram connection: {exc}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/view")
async def view_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    async with viewers_lock:
        viewers.add(websocket)
    logger.info(f"A Viewer connected. Total viewers: {len(viewers)}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with viewers_lock:
            viewers.discard(websocket)
        logger.info(f"Viewer disconnected. Total viewers: {len(viewers)}")


app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    logger.info("Server initialized. Waiting for connections...")
    logger.info(f"LiveSpeak server is listening on http://localhost:{PORT}")
    logger.info(f"Access Speaker page at http://localhost:{PORT}/speaker.html")
    logger.info(f"Access Viewer page at http://localhost:{PORT}/viewer.html")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
