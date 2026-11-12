from __future__ import annotations

from urllib.parse import urlparse

import httpx


class BaseError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class DigitizerClient:
    def __init__(self, origin: str, timeout: float = 120.0):
        self.origin = origin.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DigitizerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _call(self, method: str, path: str, **kwargs) -> dict:
        response = self._client.request(method, f"{self.origin}{path}", **kwargs)
        if response.status_code >= 400:
            raise BaseError(
                f"{method} {path} failed with {response.status_code}",
                response.status_code,
                response.text[:2000],
            )
        return response.json() if response.content else {}

    def health(self) -> bool:
        try:
            return self._call("GET", "/health").get("status") == "ok"
        except Exception:
            return False

    def create_session(self, filename: str, data: bytes) -> dict:
        return self._call(
            "POST",
            "/sessions",
            files={"file": (filename, data, "image/png")},
        )

    def set_calibration(self, session_id: str, calibration: dict) -> dict:
        return self._call(
            "POST",
            f"/sessions/{session_id}/calibration",
            json={"calibration": calibration, "manual_calibration": True},
        )

    def set_curves(self, session_id: str, curves: list[dict]) -> dict:
        return self._call("PATCH", f"/sessions/{session_id}/curves", json={"curves": curves})

    def last_session(self) -> dict:
        return self._call("GET", "/sessions/last")

    def app_url(self) -> str:
        parsed = urlparse(self.origin)
        host = parsed.hostname or "127.0.0.1"
        port = 5173 if parsed.port in (None, 8000) else parsed.port
        return f"http://{host}:{port}"


def push_recognition(
    origin: str,
    filename: str,
    data: bytes,
    recognition,
) -> dict:
    with DigitizerClient(origin) as client:
        if not client.health():
            raise BaseError(f"digitizer backend is not reachable at {origin}")
        session = client.create_session(filename, data)
        session_id = session["id"]
        calibration_sent = False
        if recognition.calibration is not None:
            client.set_calibration(session_id, recognition.calibration)
            calibration_sent = True
        curves = recognition.base_curves()
        if curves:
            client.set_curves(session_id, curves)
        return {
            "session_id": session_id,
            "app_url": client.app_url(),
            "curves_pushed": len(curves),
            "calibration_sent": calibration_sent,
        }