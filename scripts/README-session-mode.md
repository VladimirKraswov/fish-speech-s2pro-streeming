Вот готовый `README-session-mode.md`.

````md id="3v6t6n"
# Session Mode

`session_mode` — это realtime-менеджер сессий для потоковой озвучки текста.

Он нужен для сценария, где текст приходит не целиком, а дельтами:
- LLM постепенно генерирует ответ,
- клиент отправляет эти дельты в `session_mode`,
- `session_mode` буферизует текст,
- режет его на подходящие чанки,
- отправляет чанки в основной TTS backend,
- тут же начинает возвращать аудио-стрим и события о состоянии.

---

## Что делает session_mode

`session_mode` решает несколько задач:

1. **Принимает поток текста**
   - по WebSocket,
   - в виде `text_delta`,
   - с опциональным флагом `final`.

2. **Буферизует текст**
   - не отправляет в TTS слишком короткие куски,
   - старается дождаться естественной границы фразы,
   - при паузе умеет сделать `force flush`.

3. **Управляет TTS-запросами**
   - последовательно стримит чанки в основной backend,
   - умеет отменять активный TTS stream при приходе нового текста, если это включено политикой.

4. **Возвращает наружу**
   - JSON-события,
   - бинарные audio chunks.

5. **Делает cleanup**
   - вручную,
   - либо автоматически при простое сессии.

---

## Архитектура

Типовой поток такой:

```text
LLM text deltas
    ↓
WebSocket client
    ↓
session_mode
    ↓
text buffer / chunking
    ↓
main TTS backend (/v1/tts)
    ↓
audio stream back
    ↓
WebSocket client playback
````

`session_mode` сам модель не держит.
Он работает как orchestration-слой поверх основного TTS API.

---

## Зависимости

Для работы нужны:

* основной TTS backend,
* Python runtime для `session_mode`,
* `fastapi`,
* `uvicorn`,
* `httpx`,
* `loguru`,
* `pydantic`.

Обычно сервис поднимается скриптом:

```bash id="wrum92"
bash scripts/run_session_mode.sh
```

или вместе со всем стеком:

```bash id="kz3v5n"
bash scripts/up_5090.sh
```

---

## Запуск

## Отдельный запуск

```bash id="uj0yq0"
bash scripts/run_session_mode.sh
```

По умолчанию:

* host: `0.0.0.0`
* port: `8765`

Health:

```bash id="x4lyvc"
curl http://127.0.0.1:8765/health
```

---

## Переменные окружения

Основные настройки запуска:

```bash id="0m4y1b"
SESSION_HOST=0.0.0.0
SESSION_PORT=8765
SESSION_LOG_LEVEL=info
```

---

## HTTP endpoints

## `GET /health`

Проверка, что сервис жив.

Пример ответа:

```json id="8rnpxr"
{
  "ok": true,
  "service": "session_mode",
  "active_sessions": 1,
  "session_ids": ["a1b2c3..."]
}
```

## `GET /ready`

Упрощенный readiness-check.

Пример ответа:

```json id="h1u7ax"
{
  "ok": true,
  "service": "session_mode",
  "ready": true,
  "active_sessions": 1
}
```

## `GET /sessions`

Список активных сессий и их snapshot.

## `GET /sessions/{session_id}`

Состояние конкретной сессии.

---

## WebSocket endpoint

Основной realtime endpoint:

```text
ws://127.0.0.1:8765/ws
```

После подключения создается новая сессия.

---

## Формат обмена

Есть два типа исходящих данных:

1. **JSON event**
2. **binary audio frame**

Обычно клиент должен:

* читать JSON-события отдельно,
* бинарные фреймы сразу складывать в playback buffer.

---

## Сообщения клиент → сервер

## 1. `start_session`

Явно стартовать сессию и передать конфиг.

```json id="9q9r3w"
{
  "type": "start_session",
  "config": {
    "sample_rate": 44100,
    "channels": 1,
    "buffer": {
      "min_words": 3,
      "soft_limit_chars": 120,
      "hard_limit_chars": 220
    },
    "tts": {
      "base_url": "http://127.0.0.1:8080",
      "endpoint": "/v1/tts",
      "format": "pcm",
      "streaming": true,
      "stream_tokens": true,
      "stream_chunk_size": 8,
      "max_new_tokens": 512,
      "top_p": 0.8,
      "repetition_penalty": 1.1,
      "temperature": 0.8,
      "cleanup_mode": "session_idle",
      "use_memory_cache": "off",
      "normalize": true,
      "reference_id": null
    },
    "policy": {
      "session_idle_timeout_sec": 15.0,
      "cleanup_after_idle_sec": 2.0,
      "force_flush_after_sec": 0.35,
      "max_pending_emit_chunks": 32,
      "close_tts_stream_on_new_text": false
    }
  }
}
```

На практике можно и не отправлять `start_session` сразу — менеджер умеет лениво стартовать сессию при первом нормальном сообщении.

---

## 2. `text_delta`

Основное сообщение для подачи текста.

```json id="3294q9"
{
  "type": "text_delta",
  "text": "Привет! Это тест стриминга.",
  "final": false,
  "trace_id": "chunk-001"
}
```

Поля:

* `text` — новая дельта текста,
* `final` — это последняя дельта логического ответа,
* `trace_id` — опциональный id для трассировки.

---

## 3. `flush`

Принудительно сбросить текущий буфер в TTS.

```json id="j9w1p1"
{
  "type": "flush",
  "final": false,
  "reason": "manual_flush"
}
```

---

## 4. `clear`

Очистить только текстовый буфер, не трогая backend cleanup.

```json id="ah4tq9"
{
  "type": "clear",
  "reason": "user_interrupt"
}
```

---

## 5. `cleanup`

Попросить backend сделать heavy cleanup.

```json id="aiw5hx"
{
  "type": "cleanup",
  "reason": "manual_cleanup"
}
```

---

## 6. `patch_config`

Частично обновить конфиг сессии.

```json id="h6h77v"
{
  "type": "patch_config",
  "patch": {
    "buffer": {
      "min_words": 2,
      "soft_limit_chars": 80
    },
    "policy": {
      "force_flush_after_sec": 0.2
    }
  }
}
```

---

## 7. `ping`

```json id="lkz07g"
{
  "type": "ping",
  "ts_ms": 1710000000000
}
```

Ответит событием `pong`.

---

## 8. `close_session`

```json id="sglniv"
{
  "type": "close_session",
  "reason": "client_done"
}
```

---

## Упрощенный режим

Если отправить в WebSocket просто строку, без JSON, сервер воспримет это как:

```json id="7y0f3o"
{
  "type": "text_delta",
  "text": "эта строка пришла как plain text"
}
```

---

## События сервер → клиент

## `session_started`

Сессия создана.

```json id="6pjvln"
{
  "event": "session_started",
  "session_id": "abc123",
  "config": { "...": "..." }
}
```

## `text_accepted`

Новая дельта принята и помещена в буфер.

```json id="0o5nst"
{
  "event": "text_accepted",
  "session_id": "abc123",
  "buffered_text_len": 42,
  "buffered_words": 7,
  "trace_id": "chunk-001",
  "final": false
}
```

## `chunk_queued`

Очередной текстовый chunk поставлен в очередь на TTS.

```json id="7x3r3d"
{
  "event": "chunk_queued",
  "session_id": "abc123",
  "chunk": {
    "session_id": "abc123",
    "chunk_id": "def456",
    "text": "Привет! Это тест стриминга.",
    "reason": "punct",
    "words": 4,
    "chars": 28,
    "trace_id": "chunk-001",
    "final": false
  },
  "queue_size": 1
}
```

## `tts_started`

Началась обработка chunk'а в TTS backend.

## `audio_meta`

Метаданные аудио-потока.

```json id="mjlwmw"
{
  "event": "audio_meta",
  "session_id": "abc123",
  "chunk_id": "def456",
  "meta": {
    "sample_rate": 44100,
    "channels": 1,
    "format": "pcm"
  }
}
```

## `audio_chunk`

JSON-событие о бинарном чанке, который будет отправлен отдельным binary frame.

```json id="0vm8q0"
{
  "event": "audio_chunk",
  "session_id": "abc123",
  "chunk_id": "def456",
  "seq": 1,
  "size_bytes": 4096,
  "trace_id": "chunk-001"
}
```

Сразу после этого клиент обычно получает и сам бинарный фрейм.

## `tts_finished`

TTS закончил chunk.

## `buffer_cleared`

Буфер очищен.

## `cleanup_done`

Cleanup успешно выполнен.

## `pong`

Ответ на `ping`.

## `session_closed`

Сессия закрыта.

## `error`

Ошибка.

```json id="txx1hu"
{
  "event": "error",
  "code": "invalid_client_message",
  "message": "Unsupported client message type: 'foo'",
  "session_id": "abc123",
  "details": null,
  "fatal": false
}
```

---

## Chunking / buffering

Внутри используется буфер текста.

Основные параметры:

* `min_words`
  Минимум слов, после которого chunk можно выпускать.

* `soft_limit_chars`
  Мягкий порог длины текста.

* `hard_limit_chars`
  Жесткий лимит. Если буфер слишком вырос, он будет разрезан принудительно.

* `force_flush_after_sec`
  Если после последней дельты наступила пауза, буфер можно принудительно выпустить.

### Логика примерно такая

1. Приходит текст.
2. Текст добавляется в буфер.
3. Если найден естественный split по пунктуации — chunk уходит в TTS.
4. Если текст долго молчит — делается `force flush`.
5. Если буфер разросся слишком сильно — режется по hard limit.

### Причины chunk emit

В `chunk.reason` можно увидеть:

* `punct` — разрезано по пунктуации,
* `hard_limit` — разрезано по жесткому лимиту,
* `force` — принудительный flush,
* `final` — финальный кусок.

---

## Cleanup режимы

Параметр:

```json id="1c2z4o"
"cleanup_mode": "session_idle"
```

Поддерживаются:

* `request_end`
* `session_idle`
* `none`

### `request_end`

После запроса backend может делать тяжелую очистку памяти.

### `session_idle`

Основной режим для session mode.
Backend держится горячим, а cleanup делается только после простоя.

### `none`

Ничего автоматически не чистить.

---

## Типичный сценарий работы

1. Клиент открывает WebSocket.
2. Получает `session_started`.
3. Начинает слать `text_delta`.
4. Сервер отправляет:

   * `text_accepted`,
   * `chunk_queued`,
   * `tts_started`,
   * `audio_meta`,
   * `audio_chunk` + binary frames,
   * `tts_finished`.
5. Когда ответ LLM закончен — клиент отправляет финальный `text_delta` с `final=true` или делает `flush`.
6. При завершении работы — `close_session`.

---

## Минимальный пример клиента

Ниже логика, без production-обвязки:

```python id="2va3es"
import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8765/ws"
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "text_delta",
            "text": "Привет! Это тест.",
            "final": False,
            "trace_id": "demo-1",
        }))

        await ws.send(json.dumps({
            "type": "text_delta",
            "text": " Сейчас должна начаться озвучка.",
            "final": True,
            "trace_id": "demo-2",
        }))

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print("audio bytes:", len(msg))
            else:
                event = json.loads(msg)
                print("event:", event["event"])
                if event["event"] == "session_closed":
                    break

asyncio.run(main())
```

---

## Что важно учитывать клиенту

## 1. JSON и audio идут вперемешку

Клиент обязан различать:

* текстовый WebSocket frame → JSON,
* бинарный frame → audio bytes.

## 2. `audio_chunk` и binary frame — это пара

JSON `audio_chunk` говорит, что аудио-пакет сейчас будет или уже отправлен бинарно.

## 3. Один `text_delta` не равен одному audio chunk

Из-за буферизации:

* несколько маленьких `text_delta` могут слиться в один TTS chunk,
* один длинный `text_delta` может порезаться на несколько TTS chunks.

## 4. Сессия живет отдельно от одного chunk

Сессия может содержать:

* несколько входящих текстовых дельт,
* несколько TTS chunk'ов,
* несколько cleanup-событий.

---

## Диагностика

## Проверить health

```bash id="ca3hk5"
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/ready
```

## Проверить активные сессии

```bash id="f2fpm5"
curl http://127.0.0.1:8765/sessions
```

## Посмотреть логи

```bash id="dpqjve"
tail -f logs/session_mode.log
```

или:

```bash id="t5mc00"
bash scripts/logs_5090.sh
```

---

## Частые проблемы

## 1. `session_mode` поднялся, но нет аудио

Проверить:

* жив ли основной TTS backend,
* правильный ли `tts.base_url`,
* доступен ли `tts.endpoint`,
* нет ли ошибок в `logs/session_mode.log`,
* не вернул ли backend ошибку в `error` event.

## 2. Аудио приходит с задержкой

Смотреть на:

* `min_words`,
* `soft_limit_chars`,
* `force_flush_after_sec`,
* поведение клиента: как быстро он шлет дельты,
* скорость основного TTS backend.

Для уменьшения TTFA обычно уменьшают:

* `min_words`,
* `soft_limit_chars`,
* `force_flush_after_sec`.

## 3. Слишком много мелких чанков

Нужно увеличить:

* `min_words`,
* `soft_limit_chars`.

## 4. Сессии висят после отключения клиента

Проверить корректное закрытие WebSocket и cleanup в клиенте.

---

## Рекомендуемые стартовые значения

Для realtime-режима обычно хорошо начинать с такого:

```json id="r4r4ap"
{
  "buffer": {
    "min_words": 2,
    "soft_limit_chars": 80,
    "hard_limit_chars": 180
  },
  "policy": {
    "force_flush_after_sec": 0.2,
    "session_idle_timeout_sec": 15.0,
    "cleanup_after_idle_sec": 2.0,
    "close_tts_stream_on_new_text": false
  },
  "tts": {
    "cleanup_mode": "session_idle",
    "streaming": true,
    "stream_tokens": true,
    "stream_chunk_size": 8
  }
}
```

---

## Куда смотреть в коде

Основные файлы:

* `session_mode/app.py`
  HTTP + WebSocket сервис, lifecycle, registry активных сессий.

* `session_mode/manager.py`
  Основная логика:

  * прием сообщений,
  * буферизация,
  * очередь TTS,
  * idle cleanup,
  * события.

* `session_mode/buffer.py`
  Правила накопления текста и split на chunks.

* `session_mode/schema.py`
  Pydantic-схемы сообщений и событий.

* `scripts/run_session_mode.sh`
  Удобный локальный запуск сервиса.

---

## Итог

`session_mode` — это слой orchestration для low-latency speech streaming.
Он особенно полезен, когда:

* текст появляется постепенно,
* важен быстрый старт звука,
* нужен контроль буфера, чанков и cleanup без перезапуска основного TTS backend.

```
```
