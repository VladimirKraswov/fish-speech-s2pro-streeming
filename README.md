# Fish Speech S2 Pro — стриминговая TTS на RTX 5090

Репозиторий содержит полный стек для запуска **Fish Speech** в стриминговом режиме:

- **server** — сервер синтеза речи: FastAPI + PyTorch + Fish Speech
- **proxy** — session API, commit logic и PCM/NDJSON streaming
- **webui** — веб-интерфейс для интерактивного тестирования

Проект запускается через **Docker Compose** и поддерживает два режима:

- **prod** — собранные Docker images, без монтирования исходников
- **dev** — исходники монтируются в контейнеры, правки Python/Frontend подхватываются без полной пересборки

Оптимизировано под **NVIDIA RTX 5090**, но подойдёт для любых GPU с достаточным объёмом VRAM.

---

## Структура сервисов

| Сервис | Порт | Назначение |
|---|---:|---|
| `server` | `8080` | TTS inference server |
| `proxy` | `9000` | Stateful session proxy + PCM stream |
| `webui` | `9001` | Browser UI для тестирования |

Health endpoints:

```bash
curl http://localhost:8080/v1/health
curl http://localhost:9000/health
curl http://localhost:9001/health
```

---

## Быстрый старт, если модели и референс уже подготовлены

Нужны файлы:

```text
checkpoints/fs-1.2-int8-s2-pro-int8/model.pth
checkpoints/s2-pro/codec.pth
references/voice/voice.codes.pt
references/voice/voice.lab
```

Запуск production-стека:

```bash
git clone <url-репозитория>
cd fish-speech-s2pro-streeming

docker compose up -d --build
```

Или через `Makefile`:

```bash
make prod-up
```

После запуска:

- TTS server: http://localhost:8080
- Proxy: http://localhost:9000
- WebUI: http://localhost:9001

---

# Режимы запуска

## Production mode

Production использует основной `compose.yml`.

```bash
docker compose up -d --build
```

Через `Makefile`:

```bash
make prod-up
```

Остановка:

```bash
make prod-down
```

Логи:

```bash
make prod-logs
make prod-logs SERVICE=server
make prod-logs SERVICE=proxy
make prod-logs SERVICE=webui
```

Production подходит для стабильного запуска, где код не меняется на лету.

---

## Development mode

Development использует overlay-файл:

```text
compose.dev.yml
```

В dev-режиме:

- `fish_speech/` монтируется в контейнер
- `fish_speech_server/` монтируется в контейнер
- `config/` монтируется в контейнер
- proxy запускается через `uvicorn --reload`
- webui запускается через Vite dev server
- после правок `pcm.py` обычно не нужна пересборка
- после правок proxy часто не нужен даже restart, потому что работает reload

Запуск dev-стека:

```bash
docker compose -f compose.yml -f compose.dev.yml up -d
```

Через `Makefile`:

```bash
make dev-up
```

Запуск с пересборкой:

```bash
make dev-build-up
```

Запуск только конкретных сервисов:

```bash
make dev-up SERVICES="proxy webui"
make dev-up SERVICES="server"
```

Перезапуск одного сервиса:

```bash
make restart SERVICE=proxy
make restart SERVICE=server
make restart SERVICE=webui
```

Пересборка одного сервиса:

```bash
make build SERVICE=proxy
make build SERVICE=server
make build SERVICE=webui
```

Полная пересборка одного сервиса без cache:

```bash
make rebuild SERVICE=proxy
```

Поднять один сервис без зависимостей:

```bash
make up SERVICE=proxy
```

Логи одного сервиса:

```bash
make dev-logs SERVICE=proxy
```

---

# Почему proxy больше не должен скачивать torch

Proxy — это транспортный/session слой. Ему не нужен PyTorch и не нужен полный `fish-speech[cu128]`.

В dev/prod Dockerfile для proxy должны ставиться только лёгкие зависимости:

```text
fastapi
httpx
uvicorn
pydantic
loguru
```

Это важно, потому что иначе при каждом изменении Python-кода Docker может заново качать:

```text
torch-2.8.0
torchaudio
descript-audio-codec
librosa
transformers
...
```

Правильный dev-cycle для proxy:

```bash
# один раз
make build SERVICE=proxy

# после правок Python-кода
make restart SERVICE=proxy
```

В dev-режиме из-за `uvicorn --reload` часто достаточно просто сохранить файл.

---

# Полная установка с нуля

## 1. Требования

- Linux: Ubuntu 22.04 / 24.04
- Docker
- Docker Compose plugin
- NVIDIA Container Toolkit
- Python 3.10+
- `uv`
- `ffmpeg`
- GPU с достаточным объёмом VRAM

Проверка GPU в Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

---

## 2. Клонирование

```bash
git clone <url>
cd fish-speech-s2pro-streeming
```

Локальные Python-зависимости нужны только для вспомогательных скриптов вне Docker:

```bash
uv sync --extra cu128
```

или:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[cu128]"
```

---

## 3. Сборка Docker-образов

Production:

```bash
make prod-up
```

Development:

```bash
make dev-build-up
```

Или напрямую:

```bash
docker compose build
```

---

## 4. Скачивание модели s2-pro

Скачайте модель `fishaudio/s2-pro`:

```bash
mkdir -p checkpoints/s2-pro

docker run --rm -v "$PWD":/workspace -w /workspace \
  --entrypoint uv \
  fish-speech-server:cuda \
  run huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

Если `huggingface-cli` недоступен внутри образа, можно поставить локально:

```bash
uv pip install huggingface_hub
uv run huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

---

## 5. Квантизация модели в int8

Для RTX 5090 рекомендуется **int8 weight-only** квантизация.

```bash
docker run --rm --gpus all --user root \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint uv \
  fish-speech-server:cuda \
  run python tools/llama/quantize.py \
    --checkpoint-path checkpoints/s2-pro \
    --mode int8 \
    --timestamp s2-pro-int8
```

После этого должен появиться файл:

```text
checkpoints/fs-1.2-int8-s2-pro-int8/model.pth
```

Если файлы созданы от root:

```bash
sudo chown -R "$USER:$USER" checkpoints/fs-1.2-int8-s2-pro-int8
```

---

## 6. Подготовка референсного голоса

Создайте папку:

```bash
mkdir -p input_ref
```

Положите туда:

```text
input_ref/voice.wav
input_ref/voice.lab
```

Требования к `voice.wav`:

- 5–15 секунд
- моно
- 44100 Hz
- без музыки и сильного шума

Конвертация:

```bash
ffmpeg -i my_voice.mp3 \
  -ac 1 \
  -ar 44100 \
  -c:a pcm_s16le \
  -ss 0 \
  -t 15 \
  input_ref/voice.wav
```

`voice.lab` должен содержать точный текст, произнесённый в `voice.wav`.

---

## 7. Предварительное кодирование референса

```bash
docker run --rm --user root \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint uv \
  fish-speech-server:cuda \
  run python tools/preencode_references.py \
    -i input_ref \
    --ref-id voice \
    --checkpoint-path checkpoints/s2-pro/codec.pth \
    --device cpu
```

Результат:

```text
references/voice/voice.codes.pt
references/voice/voice.lab
```

---

# Конфигурация

Основной конфиг:

```text
config/runtime.json
```

Он монтируется в контейнеры:

```yaml
./config:/app/config
```

Поэтому после изменения `runtime.json` обычно достаточно перезапустить нужный сервис.

Для server:

```bash
make restart SERVICE=server
```

Для proxy:

```bash
make restart SERVICE=proxy
```

---

## Важные параметры для RTX 5090

```json
"model": {
  "device": "cuda",
  "precision": "bfloat16",
  "compile": true,
  "cache_max_seq_len": 768
}
```

Рекомендации:

- `compile: true` — лучше performance, но первый запуск может быть долгим
- `compile: false` — быстрее старт, удобнее для разработки
- `cache_max_seq_len: 768` — безопасный стартовый budget
- `cleanup_after_request: false` — меньше overhead между запросами
- `empty_cache_per_stream_chunk: false` — важно для streaming latency

Для dev можно запускать без compile:

```bash
COMPILE=0 make dev-up
```

---

## Proxy commit policy

Настройки proxy находятся в:

```json
"proxy": {
  "commit": {
    "first": {
      "min_chars": 40,
      "target_chars": 58,
      "max_chars": 84
    },
    "next": {
      "min_chars": 120,
      "target_chars": 160,
      "max_chars": 240
    }
  }
}
```

Идея:

- первый commit меньше — чтобы быстрее услышать первый звук
- следующие commit крупнее — чтобы меньше рвать интонацию
- короткий финальный хвост должен уходить через `/finish` или short sentence commit logic

---

## Frontend overrides

`frontend_overrides` ограничивает, какие поля WebUI может менять при открытии session.

Если WebUI возвращает `400 disallowed paths`, лучше добавить нужные поля в:

```json
"frontend_overrides": {
  "allowed_paths": []
}
```

На время разработки можно отключить проверку:

```json
"frontend_overrides": {
  "enabled": false
}
```

Но для стабильного режима лучше оставить `enabled: true`.

---

# Makefile команды

## Production

```bash
make prod-up
make prod-down
make prod-logs
make prod-logs SERVICE=server
```

## Development

```bash
make dev-up
make dev-build-up
make dev-down
make dev-logs
make dev-logs SERVICE=proxy
```

## Работа с отдельными сервисами

```bash
make build SERVICE=proxy
make rebuild SERVICE=proxy
make restart SERVICE=proxy
make up SERVICE=proxy
make stop SERVICE=proxy
```

## Проверка

```bash
make ps
make health
```

---

# Docker Compose напрямую

Если не используете `Makefile`.

## Production

```bash
docker compose up -d --build
docker compose logs -f proxy
docker compose restart proxy
docker compose down
```

## Development

```bash
docker compose -f compose.yml -f compose.dev.yml up -d
docker compose -f compose.yml -f compose.dev.yml up -d --build proxy
docker compose -f compose.yml -f compose.dev.yml restart proxy
docker compose -f compose.yml -f compose.dev.yml logs -f proxy
docker compose -f compose.yml -f compose.dev.yml down
```

---

# Частые сценарии разработки

## Я поправил `fish_speech_server/proxy/pcm.py`

В dev-режиме обычно ничего делать не надо — `uvicorn --reload` перезапустится сам.

Если нужно руками:

```bash
make restart SERVICE=proxy
```

Не нужно:

```bash
docker compose build proxy
```

## Я поправил frontend

В dev-режиме Vite сам подхватит изменения.

Если нужно:

```bash
make restart SERVICE=webui
```

## Я поменял `requirements.txt` proxy

Нужна пересборка proxy:

```bash
make build SERVICE=proxy
make up SERVICE=proxy
```

## Я поменял `pyproject.toml`

Для server:

```bash
make build SERVICE=server
make up SERVICE=server
```

Для proxy пересборка нужна только если proxy действительно зависит от новых Python-пакетов.

## Я поменял `Dockerfile.proxy`

```bash
make build SERVICE=proxy
make up SERVICE=proxy
```

## Я поменял модель или reference

Если файлы лежат в `checkpoints/` или `references/`, пересборка не нужна, потому что они монтируются volume’ами.

Нужен restart server:

```bash
make restart SERVICE=server
```

---

# API proxy

## Открыть session

```bash
curl -s http://localhost:9000/session/open \
  -H "Content-Type: application/json" \
  -d '{"config_text":"{}"}'
```

Обычно WebUI отправляет полный config override.

## Append text

```bash
curl -s http://localhost:9000/session/<SESSION_ID>/append \
  -H "Content-Type: application/json" \
  -d '{"text":"Привет, это тест."}'
```

## Finish

```bash
curl -s http://localhost:9000/session/<SESSION_ID>/finish \
  -H "Content-Type: application/json" \
  -d '{"reason":"input_finished"}'
```

## Stream

```bash
curl -N http://localhost:9000/session/<SESSION_ID>/pcm-stream
```

События идут как NDJSON:

```json
{"type":"session_start"}
{"type":"commit_start","commit_seq":1}
{"type":"meta","sample_rate":44100,"channels":1,"sample_width":2}
{"type":"pcm","seq":1,"data":"...base64..."}
{"type":"commit_done","commit_seq":1}
{"type":"session_done"}
```

---

# Проверка полного сценария

1. Запустить dev:

```bash
make dev-up
```

2. Открыть WebUI:

```text
http://localhost:9001
```

3. Нажать `Open session`

4. Нажать `Start streaming`

5. Проверить логи proxy:

```bash
make dev-logs SERVICE=proxy
```

Ожидаемые признаки:

```text
session open id=...
commit queued session=... seq=1 reason=...
REQ ... commit_seq=1 upstream start ...
commit queued session=... seq=2 reason=...
REQ ... commit_seq=2 upstream start ...
session finish id=... committed=...
```

---

# Устранение типичных проблем

## 1. Каждый build proxy снова скачивает torch

Proxy не должен устанавливать `fish-speech[сpu]` и не должен скачивать `torch`.

Проверь `docker/Dockerfile.proxy`.

Неправильно:

```dockerfile
RUN pip install -e .[cpu]
```

Правильно:

```dockerfile
COPY fish_speech_server/proxy/requirements.txt /tmp/proxy-requirements.txt
RUN pip install -r /tmp/proxy-requirements.txt
```

После этого proxy rebuild будет быстрым.

---

## 2. Изменил Python-код, но ничего не поменялось

В prod-режиме исходники не монтируются. Нужно пересобрать образ:

```bash
make build SERVICE=proxy
make up SERVICE=proxy
```

В dev-режиме исходники монтируются. Достаточно:

```bash
make restart SERVICE=proxy
```

или дождаться `uvicorn --reload`.

---

## 3. Proxy падает с `ModuleNotFoundError`

Проверь, что в `Dockerfile.proxy` есть:

```dockerfile
ENV PYTHONPATH=/app
COPY fish_speech ./fish_speech
COPY fish_speech_server ./fish_speech_server
```

В dev-режиме проверь `compose.dev.yml`:

```yaml
volumes:
  - ./fish_speech:/app/fish_speech
  - ./fish_speech_server:/app/fish_speech_server
```

---

## 4. Server долго не становится healthy

Если включён `compile: true`, первый запуск может быть долгим.

Смотри логи:

```bash
make dev-logs SERVICE=server
```

Для разработки можно отключить compile:

```bash
COMPILE=0 make dev-up
```

---

## 5. WebUI возвращает 400 при открытии session

Причина обычно в `frontend_overrides.allowed_paths`.

Варианты:

1. Добавить недостающее поле в `allowed_paths`
2. Временно отключить проверку:

```json
"frontend_overrides": {
  "enabled": false
}
```

После изменения:

```bash
make restart SERVICE=proxy
```

---

## 6. Порты заняты

Проверить:

```bash
sudo lsof -i :8080
sudo lsof -i :9000
sudo lsof -i :9001
```

Или изменить port mapping в `compose.yml`.

---

# Рекомендуемая структура проекта

```text
.
├── compose.yml
├── compose.dev.yml
├── Makefile
├── .dockerignore
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.proxy
│   ├── Dockerfile.webui
│   └── Dockerfile.webui.dev
├── config/
│   └── runtime.json
├── checkpoints/
├── references/
├── fish_speech/
├── fish_speech_server/
│   └── proxy/
│       ├── pcm.py
│       └── requirements.txt
├── fish_speech_web_ui/
│   ├── server.py
│   └── ui/
├── tools/
└── scripts/
```

---

# Docker cache рекомендации

В Dockerfiles используется BuildKit cache:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip ...
RUN --mount=type=cache,target=/tmp/uv-cache ...
RUN --mount=type=cache,target=/root/.npm ...
```

Убедись, что BuildKit включён:

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

В `Makefile` эти переменные уже выставлены.

---

# Требования к оборудованию

Минимально:

- GPU NVIDIA с достаточным VRAM
- 16 GB RAM
- Docker с GPU support

Рекомендуется:

- RTX 5090 / 4090 / A-серия
- 24 GB VRAM
- 32–64 GB RAM
- `compile: true` для production
- `compile: false` для активной разработки

---

# Заключение

Рекомендуемый workflow:

## Для разработки

```bash
make dev-up
make dev-logs SERVICE=proxy
make restart SERVICE=proxy
```

## Для production

```bash
make prod-up
make prod-logs
```

Главные правила:

- не пересобирать весь проект после каждой правки
- proxy не должен скачивать torch
- модели и референсы хранить как mounted volumes
- для Python-правок использовать dev overlay
- для frontend-правок использовать Vite dev server
- пересобирать только тот сервис, где изменились зависимости или Dockerfile