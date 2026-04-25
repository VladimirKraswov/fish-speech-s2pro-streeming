from fish_speech.driver.api import FishSpeechDriver
from fish_speech.driver.session import DriverSession, DriverSessionConfig, SynthesisContext
from fish_speech.driver.types import (
    DriverAudioChunkEvent,
    DriverContext,
    DriverErrorEvent,
    DriverEvent,
    DriverFinalAudioEvent,
    DriverGenerationOptions,
    DriverHealth,
    DriverReference,
    DriverSegmentRequest,
    DriverStats,
    DriverSynthesisRequest,
    DriverTokenChunkEvent,
    GenerationOptions,
)

__all__ = [
    "FishSpeechDriver",
    "DEFAULT_RUNTIME_CONFIG_PATH",
    "DriverConfig",
    "ModelConfig",
    "PathsConfig",
    "WarmupConfig",
    "load_driver_config",
    "load_runtime_config",
    "DriverSession",
    "DriverSessionConfig",
    "SynthesisContext",
    "DriverAudioChunkEvent",
    "DriverContext",
    "DriverErrorEvent",
    "DriverEvent",
    "DriverFinalAudioEvent",
    "DriverGenerationOptions",
    "DriverHealth",
    "DriverReference",
    "DriverSegmentRequest",
    "DriverStats",
    "DriverSynthesisRequest",
    "DriverTokenChunkEvent",
    "GenerationOptions",
]


def __getattr__(name: str):
    if name in {
        "DEFAULT_RUNTIME_CONFIG_PATH",
        "DriverConfig",
        "ModelConfig",
        "PathsConfig",
        "WarmupConfig",
        "load_driver_config",
        "load_runtime_config",
    }:
        from fish_speech.driver import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
