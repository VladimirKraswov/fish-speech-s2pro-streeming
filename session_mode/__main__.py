from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    import uvicorn

    host = os.getenv("SESSION_MODE_HOST", "0.0.0.0")
    port = int(os.getenv("SESSION_MODE_PORT", "8765"))
    reload = _env_flag("SESSION_MODE_RELOAD", False)
    log_level = os.getenv("SESSION_MODE_LOG_LEVEL", "info").lower()
    ws_impl = os.getenv("SESSION_MODE_WS", "auto")

    uvicorn.run(
        "session_mode.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        ws=ws_impl,
    )


if __name__ == "__main__":
    main()