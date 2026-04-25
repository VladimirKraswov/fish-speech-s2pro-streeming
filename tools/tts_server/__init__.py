"""Unified HTTP and proxy server layer built on top of fish_speech.driver."""

__all__ = ["API", "create_app"]


def create_app(*args, **kwargs):
    from tools.tts_server.app import create_app as _create_app

    return _create_app(*args, **kwargs)


def __getattr__(name: str):
    if name == "API":
        from tools.tts_server.app import API

        return API
    raise AttributeError(name)
