"""FastAPI app: dashboard, MJPEG video stream, live metrics websocket, and
session start/stop/calibrate controls.

The engine owns a background thread for capture + inference; this module only
ever reads snapshots of its state, keeping the async event loop non-blocking.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import download_model
from engine import Engine
from report import generate_report
from session import SessionRecorder

cfg = config.parse_args(sys.argv[1:])
config.ensure_dirs()

engine = Engine(cfg)
recorder = SessionRecorder(engine, cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PATH.exists():
        ok = download_model.download()
        if not ok:
            raise RuntimeError(
                "Face landmarker model is missing and could not be downloaded. "
                "Run `py download_model.py` manually and restart."
            )
    engine.start()
    try:
        yield
    finally:
        if recorder.active:
            recorder.stop()
        engine.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/video")
async def video():
    async def frame_generator():
        boundary = b"--frame\r\n"
        while True:
            jpeg = engine.get_jpeg()
            if jpeg is not None:
                yield boundary
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                yield jpeg
                yield b"\r\n"
            await asyncio.sleep(1.0 / max(cfg.target_infer_fps, 1.0))

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def metrics_ws(ws: WebSocket) -> None:
    await ws.accept()
    interval = 1.0 / cfg.metrics_hz
    try:
        while True:
            metrics = engine.get_metrics()
            payload = {
                "metrics": asdict(metrics) if metrics is not None else None,
                "calibrated": cfg.calibration.calibrated,
                "calibrating": engine.aggregator.is_calibrating,
                "recording": recorder.active,
                "error": engine.error,
            }
            await ws.send_json(payload)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass


@app.post("/api/calibrate")
async def calibrate() -> JSONResponse:
    engine.calibrate(duration_s=2.0)
    return JSONResponse({"status": "calibrating", "duration_s": 2.0})


@app.post("/api/session/start")
async def session_start() -> JSONResponse:
    path = recorder.start()
    return JSONResponse({"status": "recording", "csv": path.name})


@app.post("/api/session/stop")
async def session_stop() -> JSONResponse:
    path = recorder.stop()
    if path is None:
        return JSONResponse({"status": "not_recording"}, status_code=400)
    report_path = generate_report(path, cfg)
    return JSONResponse(
        {
            "status": "stopped",
            "csv": path.name,
            "report_url": f"/reports/{report_path.name}",
        }
    )


@app.get("/reports/{name}", response_class=HTMLResponse)
async def get_report(name: str) -> FileResponse:
    safe_name = config.REPORTS_DIR.joinpath(name).resolve()
    if not str(safe_name).startswith(str(config.REPORTS_DIR.resolve())) or not safe_name.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(safe_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)
