import io
import os
import re
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger
from starlette.responses import Response
from typing_extensions import Annotated

from fish_speech import DriverErrorEvent, DriverFinalAudioEvent
from fish_speech_server.schema import (
    AddEncodedReferenceResponse,
    AddReferenceResponse,
    CloseSynthesisSessionResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    OpenSynthesisSessionRequest,
    OpenSynthesisSessionResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    StatefulTTSRequest,
    SynthesisSessionInfoResponse,
    UpdateReferenceResponse,
)
from fish_speech_server.api.utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from fish_speech_server.services.adapter import (
    api_tts_to_driver_request,
    stateful_tts_to_driver_request,
)
from fish_speech_server.services.continuation import build_continuation_debug_summary
from fish_speech_server.services.stateful_inference import stateful_inference_async
from fish_speech_server.services.model_manager import ModelManager
from fish_speech_server.services.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


# =============================================================================
# Endpoint: Health check (проверка работоспособности)
# =============================================================================
@routes.http("/v1/health")
class Health(HttpView):
    """GET/POST /v1/health — проверка, что сервер жив и модель загружена.
    Используется Docker healthcheck и мониторингом.
    """

    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


# =============================================================================
# Вспомогательные функции для получения информации о памяти GPU
# =============================================================================
def _model_param_memory_gb(module: torch.nn.Module) -> tuple[float, int]:
    """Возвращает память (ГБ), занимаемую только весами модели, и количество параметров."""
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    count = sum(p.numel() for p in module.parameters())
    return (round(total_bytes / (1024**3), 3), count)


def _gpu_memory_info(model_manager: ModelManager | None = None):
    """Собирает текущую информацию об использовании видеопамяти CUDA.
    Возвращает словарь с allocated, reserved, max, а также информацию о моделях (DAC, LLaMA).
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    gb = 1024**3
    out = {
        "cuda_available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
        "device_count": torch.cuda.device_count(),
    }
    if model_manager is not None:
        models = {}
        if (
            hasattr(model_manager, "decoder_model")
            and model_manager.decoder_model is not None
        ):
            dac_gb, dac_count = _model_param_memory_gb(model_manager.decoder_model)
            models["dac"] = {"param_gb": dac_gb, "param_count": dac_count}
        if getattr(model_manager, "_worker_memory_info", None):
            wi = model_manager._worker_memory_info
            if "llama_param_gb" in wi:
                models["llama"] = {
                    "param_gb": wi["llama_param_gb"],
                    "param_count": wi.get("llama_param_count"),
                }
        if models:
            out["models"] = models
    if torch.cuda.device_count() > 1:
        out["devices"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                out["devices"].append(
                    {
                        "device": i,
                        "name": torch.cuda.get_device_name(i),
                        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
                        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
                        "max_allocated_gb": round(
                            torch.cuda.max_memory_allocated() / gb, 3
                        ),
                    }
                )
    try:
        out["memory_summary"] = torch.cuda.memory_summary(abbreviated=True)
    except Exception:
        pass
    return out


def _gpu_memory_text(info: dict) -> str:
    """Форматирует информацию о памяти GPU в читаемый текст (таблица)."""
    if not info.get("cuda_available"):
        return "CUDA not available.\n"
    lines = [
        "GPU memory (current)",
        f"  allocated_gb:    {info['allocated_gb']}",
        f"  reserved_gb:     {info['reserved_gb']}",
        f"  max_allocated_gb: {info['max_allocated_gb']}",
        f"  device_count:    {info['device_count']}",
        "",
    ]
    if "models" in info:
        lines.append("Model weights (params only)")
        for name, m in info["models"].items():
            lines.append(
                f"  {name}: param_gb={m.get('param_gb')} param_count={m.get('param_count', 'N/A')}"
            )
        lines.append("")
    if "devices" in info:
        for d in info["devices"]:
            lines.append(f"  device {d['device']} ({d['name']})")
            lines.append(
                f"    allocated_gb: {d['allocated_gb']}  reserved_gb: {d['reserved_gb']}  max_allocated_gb: {d['max_allocated_gb']}"
            )
        lines.append("")
    if "memory_summary" in info:
        lines.append(info["memory_summary"])
    return "\n".join(lines)


def _dump_memory_snapshot(out_dir: str = "/workspace") -> tuple[str | None, str | None]:
    """Сохраняет снапшот памяти CUDA в pickle-файл для последующего анализа (memory viz)."""
    if not torch.cuda.is_available():
        return None, "CUDA not available"
    dump_fn = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dump_fn is None:
        return None, "_dump_snapshot not found (PyTorch too old?)"
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, f"memory_snapshot_{int(time.time())}.pickle")
        dump_fn(path)
        return path, None
    except Exception as e:
        return (
            None,
            f"Dump failed: {e!r} (need PyTorch built with CUDA memory snapshot support?)",
        )


# =============================================================================
# Endpoint: Отладка памяти GPU
# =============================================================================
@routes.http.get("/v1/debug/memory")
async def debug_memory():
    """
    GET /v1/debug/memory — возвращает текущее использование видеопамяти.
    Параметры:
        ?format=text  -> текстовый формат (таблица)
        ?dump=1       -> дополнительно создать .pickle снапшот (требуется FISH_RECORD_MEMORY_HISTORY=1)
    """
    model_manager = getattr(request.app.state, "model_manager", None)
    info = _gpu_memory_info(model_manager)
    if request.query_params.get("dump", "").strip() in ("1", "true", "True"):
        out_dir = request.query_params.get("dump_dir", "").strip() or "/workspace"
        snapshot_path, snapshot_err = _dump_memory_snapshot(out_dir=out_dir)
        if snapshot_path:
            info["snapshot_path"] = snapshot_path
            info["snapshot_note"] = (
                "Open at https://pytorch.org/memory_viz or: python -m torch.cuda._memory_viz trace_plot <path> -o out.html"
            )
        else:
            info["snapshot_error"] = (
                snapshot_err
                or "Set FISH_RECORD_MEMORY_HISTORY=1 at startup for alloc history in snapshot."
            )
    if request.query_params.get("format", "").strip().lower() == "text":
        return Response(_gpu_memory_text(info), media_type="text/plain; charset=utf-8")
    return JSONResponse(info)


# =============================================================================
# Endpoint: VQGAN кодирование аудио в токены
# =============================================================================
@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    POST /v1/vqgan/encode — кодирует аудио в дискретные коды (токены) с помощью VQGAN (DAC).
    Используется для предвычисления референсов.
    """
    try:
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


# =============================================================================
# Endpoint: VQGAN декодирование токенов в аудио
# =============================================================================
@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    POST /v1/vqgan/decode — декодирует токены обратно в аудио (PCM).
    """
    try:
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


# =============================================================================
# Основной эндпоинт: синтез речи (Text‑to‑Speech)
# =============================================================================
@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    POST /v1/tts — синтезирует речь из текста.
    Поддерживает:
        - Потоковый режим (streaming=True) -> отдаёт WAV чанками.
        - Пакетный режим -> возвращает полный аудиофайл.
    Обязательно указать reference_id или загруженные references.
    """
    try:
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        if not req.reference_id and not req.references:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Reference is required for this test mode: provide reference_id or references",
            )

        sample_rate = driver.sample_rate

        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, driver),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
        else:
            chunks = []
            for event in driver.synthesize(api_tts_to_driver_request(req)):
                if isinstance(event, DriverErrorEvent) and event.error:
                    raise RuntimeError(str(event.error))
                if isinstance(event, DriverFinalAudioEvent):
                    chunks.append(event.audio)
            if not chunks:
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content="No audio generated, please check the input text.",
                )
            fake_audios = np.concatenate(chunks, axis=0)
            buffer = io.BytesIO()
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

            return StreamResponse(
                iterable=buffer_to_async_generator(buffer.getvalue()),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


# =============================================================================
# Управление референсными голосами (Reference Voices)
# =============================================================================

# 1. Добавление нового референса (загружается аудиофайл и текст)
@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    POST /v1/references/add — добавляет новый референсный голос.
    Параметры:
        id: уникальное имя референса (латиница, цифры, -, _, пробел)
        audio: WAV‑файл
        text: транскрипция (текст, произнесённый в аудио)
    """
    temp_file_path = None

    try:
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")
        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        driver.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    "Failed to clean up temporary file %s: %s", temp_file_path, e
                )


# 2. Добавление предварительно закодированного референса (минуя кодирование на сервере)
@routes.http.post("/v1/references/add_encoded")
async def add_reference_encoded(
    id: str = Body(...),
    codes: UploadFile = Body(...),
    lab: UploadFile = Body(...),
    stem: str | None = Body(None),
):
    """
    POST /v1/references/add_encoded — добавляет или обновляет референс, уже закодированный в .codes.pt и .lab.
    Позволяет избежать повторного кодирования аудио на сервере.
    """
    try:
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")
        codes_bytes = await codes.read()
        lab_bytes = await lab.read()
        if not codes_bytes:
            raise ValueError("Codes file is empty")
        lab_text = lab_bytes.decode("utf-8", errors="replace").strip()
        if not lab_text:
            raise ValueError("Lab content is empty")
        stem_val = stem.strip() if stem and stem.strip() else None

        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        status = driver.add_reference_encoded(id, codes_bytes, lab_text, stem=stem_val)

        response = AddEncodedReferenceResponse(
            success=True,
            status=status,
            message=f"Reference '{id}' {status}",
            reference_id=id,
        )
        return format_response(response)

    except ValueError as e:
        logger.warning("add_encoded invalid input: %s", e)
        return format_response(
            AddEncodedReferenceResponse(
                success=False, status="error", message=str(e), reference_id=id or ""
            ),
            status_code=400,
        )
    except Exception as e:
        logger.error("add_encoded error: %s", e, exc_info=True)
        return format_response(
            AddEncodedReferenceResponse(
                success=False,
                status="error",
                message="Internal server error",
                reference_id=id,
            ),
            status_code=500,
        )


# 3. Получение списка всех доступных референсов
@routes.http.get("/v1/references/list")
async def list_references():
    """
    GET /v1/references/list — возвращает массив имён (ID) всех загруженных референсных голосов.
    """
    try:
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        reference_ids = driver.list_reference_ids()

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


# 4. Удаление референса по ID
@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    DELETE /v1/references/delete — удаляет референсный голос и все его файлы.
    Тело запроса: { "reference_id": "имя" }
    """
    try:
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        driver.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


# =============================================================================
# Stateful Synthesis Sessions (Управление сессиями синтеза)
# =============================================================================


@routes.http.post("/v1/synthesis/sessions/open")
async def open_synthesis_session(
    req: Annotated[OpenSynthesisSessionRequest, Body(exclusive=True)]
):
    """
    POST /v1/synthesis/sessions/open — создает новую сессию синтеза.
    Обеспечивает сохранение контекста (истории) между запросами.
    """
    try:
        store = request.app.state.synthesis_session_store
        ctx = await store.create(
            reference_id=req.reference_id,
            max_history_turns=req.max_history_turns,
            max_history_chars=req.max_history_chars,
            max_history_code_frames=req.max_history_code_frames,
        )

        return JSONResponse(
            OpenSynthesisSessionResponse(
                ok=True,
                synthesis_session_id=ctx.synthesis_session_id,
                reference_id=ctx.reference_id,
                context=ctx.to_public_dict(),
            ).model_dump(mode="json")
        )
    except ValueError as e:
        raise HTTPException(HTTPStatus.BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.error(f"Error opening synthesis session: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to open synthesis session"
        )


@routes.http.get("/v1/synthesis/sessions/{synthesis_session_id}")
async def get_synthesis_session(synthesis_session_id: str):
    """
    GET /v1/synthesis/sessions/{id} — возвращает информацию о сессии и её историю.
    """
    store = request.app.state.synthesis_session_store
    ctx = await store.get(synthesis_session_id, touch=True)

    if ctx is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, content="Synthesis session not found")

    return JSONResponse(
        SynthesisSessionInfoResponse(
            ok=True,
            synthesis_session_id=ctx.synthesis_session_id,
            context=ctx.to_public_dict(),
        ).model_dump(mode="json")
    )


@routes.http.post("/v1/synthesis/sessions/close")
async def close_synthesis_session(synthesis_session_id: str = Body(...)):
    """
    POST /v1/synthesis/sessions/close — закрывает сессию.
    """
    store = request.app.state.synthesis_session_store
    closed = await store.close(synthesis_session_id)

    return JSONResponse(
        CloseSynthesisSessionResponse(
            ok=True,
            closed=closed,
            synthesis_session_id=synthesis_session_id,
        ).model_dump(mode="json")
    )


@routes.http.post("/v1/synthesis/synthesize")
async def stateful_synthesize(req: Annotated[StatefulTTSRequest, Body(exclusive=True)]):
    """
    POST /v1/synthesis/synthesize — stateful синтез речи.
    Автоматически обновляет историю сессии после завершения генерации.
    """
    try:
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver
        store = app_state.synthesis_session_store

        ctx = await store.get(req.synthesis_session_id, touch=True)
        if ctx is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND, content="Synthesis session not found"
            )

        # Проверка длины текста (как в обычном /v1/tts)
        if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {app_state.max_text_length}",
            )

        # Пока поддерживаем только streaming=True для stateful
        if not req.streaming:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Stateful synthesis currently only supports streaming=True",
            )

        if req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        # Build stateful request with continuation history
        stateful_req = stateful_tts_to_driver_request(req, ctx)

        # Logging diagnostics
        summary = build_continuation_debug_summary(ctx)
        logger.info(
            f"Stateful synthesize session={req.synthesis_session_id[:8]} "
            f"history_turns={summary['selected_turns']} history_chars={summary['selected_chars']}"
        )

        # We need to pass the original request info (commit_seq etc) for session tracking
        # even if stateful_req is a DriverSynthesisRequest.
        # Let's adjust stateful_inference_async to take both.
        return StreamResponse(
            iterable=stateful_inference_async(stateful_req, driver, ctx, original_req=req),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stateful synthesis: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


# 5. Переименование референса (старый ID -> новый ID)
@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    POST /v1/references/update — переименовывает референсный голос.
    Тело: { "old_reference_id": "старое_имя", "new_reference_id": "новое_имя" }
    """
    try:
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        driver = model_manager.driver

        refs_base = driver.references_dir
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        driver.rename_reference(old_reference_id, new_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)