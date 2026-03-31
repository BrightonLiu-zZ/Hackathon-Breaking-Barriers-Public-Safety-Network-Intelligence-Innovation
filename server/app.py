"""
Local inference server — mirrors the AWS Lambda interface.

Endpoint : POST /2015-03-31/functions/function/invocations
           GET  /health

Request body (same as lambda_inference/lambda_function.py):
    {"features": [[f1, f2, ...], ...], "feature_names": [...optional...]}

Response (same envelope as Lambda):
    {"statusCode": 200, "body": "{\"proba\":[...],\"pred\":[...],\"threshold\":0.74,
                                  \"inference_ms\":12.3,\"n_rows\":1}"}

CORS: Access-Control-Allow-Origin: * so the browser at localhost:8000 can call us.

Run from repo root:
    python server/app.py            # default port 8001
    python server/app.py --port 8002
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]

# ── Model (loaded once at startup) ───────────────────────────────────────────
_MODEL: xgb.Booster | None = None
_FEATURE_NAMES: list[str] = []
_THRESHOLD: float = 0.5


def _load_model(model_path: Path, threshold_path: Path) -> None:
    global _MODEL, _FEATURE_NAMES, _THRESHOLD
    print(f"Loading model  : {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    meta = json.loads(threshold_path.read_text(encoding="utf-8"))
    _MODEL = booster
    _FEATURE_NAMES = list(meta["feature_names"])
    _THRESHOLD = float(meta["threshold"])
    print(f"Loading threshold: {threshold_path}")
    print(f"  threshold={_THRESHOLD:.6f}  features={len(_FEATURE_NAMES)}")
    print("Model ready.\n")


# ── CORS / response helpers ───────────────────────────────────────────────────

def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for k, v in _cors_headers().items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


# ── Request handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args) -> None:
        # Suppress default request log; we print our own below
        pass

    # ── OPTIONS (preflight) ───────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    # ── GET /health ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        if self.path == "/health":
            _send_json(self, 200, {
                "status": "ok",
                "model_loaded": _MODEL is not None,
                "threshold": _THRESHOLD,
                "n_features": len(_FEATURE_NAMES),
            })
        else:
            _send_json(self, 404, {"error": "not found"})

    # ── POST /2015-03-31/functions/function/invocations ───────────────────────
    def do_POST(self) -> None:
        if self.path != "/2015-03-31/functions/function/invocations":
            _send_json(self, 404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            _send_json(self, 400, {"error": f"invalid JSON: {exc}"})
            return

        # Same dual-body logic as the real Lambda handler
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event

        feature_names = body.get("feature_names") or _FEATURE_NAMES
        rows = body.get("features")

        if rows is None:
            _send_json(self, 400, {"error": "missing 'features' key"})
            return

        t0 = time.perf_counter()
        try:
            X = np.asarray(rows, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim != 2 or X.shape[1] != len(feature_names):
                raise ValueError(
                    f"expected shape (N, {len(feature_names)}), got {list(X.shape)}"
                )
            dmat = xgb.DMatrix(X, feature_names=feature_names)
            proba = _MODEL.predict(dmat)  # type: ignore[union-attr]
            pred  = (proba >= _THRESHOLD).astype(int)
        except Exception as exc:
            _send_json(self, 400, {"error": str(exc)})
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        inner = {
            "proba":        [round(float(p), 6) for p in proba],
            "pred":         [int(p) for p in pred],
            "threshold":    _THRESHOLD,
            "inference_ms": round(elapsed_ms, 3),
            "n_rows":       int(X.shape[0]),
        }
        response = {"statusCode": 200, "body": json.dumps(inner)}
        _send_json(self, 200, response)

        # Human-readable terminal log
        max_p = float(max(proba))
        alert = "⚠  ALERT" if any(pred) else "   ok   "
        print(
            f"[{alert}]  n={X.shape[0]:3d}  "
            f"max_prob={max_p:.4f}  threshold={_THRESHOLD:.4f}  "
            f"{elapsed_ms:.1f} ms"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Local Lambda-compatible inference server")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument(
        "--model", type=Path,
        default=ROOT / "models" / "xgboost_56day.json",
    )
    ap.add_argument(
        "--threshold", type=Path,
        default=ROOT / "models" / "threshold_56day.json",
    )
    args = ap.parse_args()

    _load_model(args.model, args.threshold)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Listening on http://localhost:{args.port}")
    print(f"Lambda endpoint: POST http://localhost:{args.port}"
          "/2015-03-31/functions/function/invocations")
    print(f"Health check   : GET  http://localhost:{args.port}/health\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
