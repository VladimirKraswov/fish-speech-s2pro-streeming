from fish_speech.driver import (
    DriverAudioChunkEvent,
    DriverErrorEvent,
    DriverEvent,
    DriverFinalAudioEvent,
    DriverGenerationOptions,
    DriverHealth,
    DriverReference,
    DriverSegmentRequest,
    DriverSession,
    DriverSessionConfig,
    DriverStats,
    DriverSynthesisRequest,
    DriverTokenChunkEvent,
    FishSpeechDriver,
    GenerationOptions,
    SynthesisContext,
)

__all__ = [
    "FishSpeechDriver",
    "DriverAudioChunkEvent",
    "DriverErrorEvent",
    "DriverEvent",
    "DriverFinalAudioEvent",
    "DriverGenerationOptions",
    "DriverHealth",
    "DriverReference",
    "DriverSegmentRequest",
    "DriverSession",
    "DriverSessionConfig",
    "DriverStats",
    "DriverSynthesisRequest",
    "DriverTokenChunkEvent",
    "GenerationOptions",
    "SynthesisContext",
    "DriverConfig",
    "load_driver_config",
    "load_runtime_config",
]


def __getattr__(name: str):
    if name in {"DriverConfig", "load_driver_config", "load_runtime_config"}:
        from fish_speech.driver import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
