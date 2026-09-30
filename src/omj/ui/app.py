"""Root FastAPI app for `omj ui` (REQ-U01, U02, U04, U06, U09, U11, U12).

Layout:
  GET  /                      single-page UI (no external assets); redirects to /reports without targets
  GET  /reports               finished bench runs: summary, breakdowns, reliability, comparison
  GET  /ui/api/reports        the runs behind /reports (report.json summaries, read-only)
  GET  /ui/api/health         identifies an omj ui server (used by `omj bench --view`)
  GET  /ui/api/targets        target list (name, backend, model; never keys)
  GET  /ui/api/presets        bundled example presets, one entry per preset with all locales
  POST /t/<name>/v1/systemone each target's own, unchanged gateway app

When remote access is on, every route requires the access code: the browser
gets it once via ``/?token=...`` (moved into an HttpOnly cookie), API clients
send the ``x-omj-ui-token`` header.
"""

from __future__ import annotations

import hmac
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from omj.gateway.schema import SchemaError, parse_request
from omj.ui.policy import PolicyError, validate_policy
from omj.ui.targets import TargetSpec
from omj import __version__

UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
PRESETS_DIR = UI_DIR / "presets"
LOCALES = ("en", "ko")
DEFAULT_LOCALE = "en"
ACCESS_COOKIE_NAME = "omj_ui_access"
ACCESS_HEADER_NAME = "x-omj-ui-token"
ACCESS_QUERY_PARAM = "token"


def _question_shape(questions: dict) -> dict:
    """What must match across locales: question ids, types and choice keys / option counts."""
    shape = {}
    for qid, q in questions.items():
        crit = q.get("criteria")
        shape[qid] = (q["type"], sorted(crit) if isinstance(crit, dict) else len(crit or []))
    return shape


def load_presets(directory: Path = PRESETS_DIR) -> list[dict[str, Any]]:
    """Every bundled preset, validated per locale against the gateway schema and its policy (REQ-U06).

    All locales of a preset must share question ids, types and choice keys so
    one language-neutral policy applies to every language.
    """
    presets = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            locales = data["locales"]
            if DEFAULT_LOCALE not in locales:
                raise KeyError(f"missing '{DEFAULT_LOCALE}' locale")
            shapes = set()
            for code, loc in locales.items():
                if code not in LOCALES:
                    raise KeyError(f"unsupported locale {code!r}")
                parse_request({"model": "jev-latest", "state": loc["state"], "questions": loc["questions"]})
                for key in ("title", "description", "labels", "act"):
                    if key not in loc:
                        raise KeyError(f"locale {code!r} is missing {key!r}")
                shapes.add(json.dumps(_question_shape(loc["questions"]), sort_keys=True))
                validate_policy(data["policy"], set(loc["questions"]))
            if len(shapes) != 1:
                raise KeyError("locales disagree on question ids, types or choice keys")
        except (SchemaError, PolicyError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid preset {path.name}: {exc}") from exc
        presets.append({"id": path.stem, "policy": data["policy"], "locales": locales})
    return presets


@lru_cache(maxsize=1)
def _index_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _reports_html() -> str:
    return (STATIC_DIR / "reports.html").read_text(encoding="utf-8")


APP_ID = "omj-ui"
UI_ASSETS = ("logo.png", "logo-dark.png", "favicon.png")


def _target_info(spec: TargetSpec, app: FastAPI) -> dict[str, Any]:
    backend = app.state.backend
    calibrated = app.state.calibration is not None or bool(backend.capabilities.calibrated)
    return {
        "name": spec.name,
        "backend": backend.name,
        "model": backend.model_id,
        "adapter": Path(spec.adapter).name if spec.adapter else "",
        "calibrated": calibrated,
        "remote_api": backend.name in ("typesafe", "kev"),
        "endpoint": f"/t/{spec.name}/v1/systemone",
    }


def _matches(given: str | None, expected: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), expected.encode())


def create_ui_app(
    targets: list[tuple[TargetSpec, FastAPI]],
    *,
    presets_dir: Path = PRESETS_DIR,
    access_code: str | None = None,
) -> FastAPI:
    presets = load_presets(presets_dir)
    info = [_target_info(spec, app) for spec, app in targets]

    root = FastAPI(title="oh-my-jev ui", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)

    if access_code:
        @root.middleware("http")
        async def require_access(request: Request, call_next):
            # First browser visit: accept ?token= once, move it into a cookie, drop it from the URL.
            if request.url.path == "/" and _matches(request.query_params.get(ACCESS_QUERY_PARAM), access_code):
                response = RedirectResponse("/", status_code=303)
                response.set_cookie(ACCESS_COOKIE_NAME, access_code, httponly=True, samesite="strict")
                return response
            if _matches(request.cookies.get(ACCESS_COOKIE_NAME), access_code) or _matches(
                request.headers.get(ACCESS_HEADER_NAME), access_code
            ):
                return await call_next(request)
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "open the URL printed by omj ui (it carries ?token=)"}},
                status_code=401,
            )

    @root.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        if not targets:  # a reports-only server (omj bench --view) has no playground
            return RedirectResponse("/reports", status_code=307)
        return HTMLResponse(_index_html())

    @root.get("/ui/{asset}", include_in_schema=False)
    async def ui_asset(asset: str):
        """The logo and favicon; only these files, nothing else from disk."""
        if asset not in UI_ASSETS:
            return JSONResponse({"error": {"type": "not_found", "message": asset}}, status_code=404)
        return FileResponse(STATIC_DIR / asset, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @root.get("/reports", response_class=HTMLResponse, include_in_schema=False)
    async def reports_page() -> HTMLResponse:
        return HTMLResponse(_reports_html())

    @root.get("/ui/api/reports")
    async def list_reports() -> JSONResponse:
        from omj.bench.reference import bundled
        from omj.bench.report_index import load_runs

        ref = bundled()
        return JSONResponse({"runs": load_runs(), "playground": bool(targets),
                             "bundled_reference": {"name": ref["name"], "measured_at": ref["measured_at"], "suites": ref["suites"]}})

    @root.get("/ui/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"app": APP_ID, "playground": bool(targets)})

    @root.get("/ui/api/targets")
    async def list_targets() -> JSONResponse:
        return JSONResponse({"targets": info})

    @root.get("/ui/api/presets")
    async def list_presets() -> JSONResponse:
        return JSONResponse({"presets": presets, "default_locale": DEFAULT_LOCALE, "locales": list(LOCALES)})

    for spec, app in targets:
        root.mount(f"/t/{spec.name}", app)

    root.state.targets = info
    root.state.access_required = bool(access_code)
    return root
