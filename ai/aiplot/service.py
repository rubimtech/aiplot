from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .push import BaseError, DigitizerClient, push_recognition
from .recognize import Recognition, recognize

app = FastAPI(title="aiplot recognition", version="0.1.0")


class PushResult(BaseModel):
    session_id: str
    app_url: str
    curves_pushed: int
    calibration_sent: bool


class Health(BaseModel):
    status: str = "ok"
    ocr: bool = False
    default_base: str = ""


@app.get("/health", response_model=Health)
def health() -> Health:
    from . import ocr

    return Health(
        ocr=ocr.available(),
        default_base=os.environ.get("AIPLOT_BASE", "http://127.0.0.1:8000"),
    )


def _recognize_upload(
    data: bytes,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    x_scale: str | None,
    y_scale: str | None,
    use_ocr: bool,
    max_curves: int,
    max_points: int,
) -> Recognition:
    x_range = (x_min, x_max) if x_min is not None and x_max is not None else None
    y_range = (y_min, y_max) if y_min is not None and y_max is not None else None
    try:
        return recognize(
            data,
            x_range=x_range,
            y_range=y_range,
            x_scale=x_scale,
            y_scale=y_scale,
            use_ocr=use_ocr,
            max_curves=max_curves,
            max_points=max_points,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recognize")
async def recognize_endpoint(
    file: Annotated[UploadFile, File()],
    x_min: Annotated[float | None, Form()] = None,
    x_max: Annotated[float | None, Form()] = None,
    y_min: Annotated[float | None, Form()] = None,
    y_max: Annotated[float | None, Form()] = None,
    x_scale: Annotated[str | None, Form()] = None,
    y_scale: Annotated[str | None, Form()] = None,
    use_ocr: Annotated[bool, Form()] = True,
    max_curves: Annotated[int, Form()] = 8,
    max_points: Annotated[int, Form()] = 120,
) -> dict:
    data = await file.read()
    result = _recognize_upload(
        data, x_min, x_max, y_min, y_max, x_scale, y_scale, use_ocr, max_curves, max_points
    )
    payload = result.as_dict()
    payload["data_curves"] = result.data_points()
    payload["filename"] = file.filename
    return payload


@app.post("/recognize/push", response_model=PushResult)
async def recognize_and_push(
    file: Annotated[UploadFile, File()],
    base: Annotated[str, Form()] = "",
    x_min: Annotated[float | None, Form()] = None,
    x_max: Annotated[float | None, Form()] = None,
    y_min: Annotated[float | None, Form()] = None,
    y_max: Annotated[float | None, Form()] = None,
    x_scale: Annotated[str | None, Form()] = None,
    y_scale: Annotated[str | None, Form()] = None,
    use_ocr: Annotated[bool, Form()] = True,
    max_curves: Annotated[int, Form()] = 8,
    max_points: Annotated[int, Form()] = 120,
) -> PushResult:
    data = await file.read()
    result = _recognize_upload(
        data, x_min, x_max, y_min, y_max, x_scale, y_scale, use_ocr, max_curves, max_points
    )
    origin = base or os.environ.get("AIPLOT_BASE", "http://127.0.0.1:8000")
    try:
        pushed = push_recognition(origin, file.filename or "plot.png", data, result)
    except BaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PushResult(**pushed)


@app.get("/sessions/last")
def last_session(base: str = "") -> dict:
    origin = base or os.environ.get("AIPLOT_BASE", "http://127.0.0.1:8000")
    with DigitizerClient(origin) as client:
        session = client.last_session()
        app_url = client.app_url()
    return {"session_id": session.get("id"), "app_url": app_url}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "aiplot.service:app",
        host=os.environ.get("AIPLOT_HOST", "127.0.0.1"),
        port=int(os.environ.get("AIPLOT_PORT", "8010")),
    )