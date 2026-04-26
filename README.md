# Fish Speech S2 Pro — стриминговая TTS на RTX 5090

Репозиторий содержит полный стек для запуска **Fish Speech** в стриминговом режиме:
- сервер синтеза речи (FastAPI + PyTorch)
- прокси для управления сессиями и PCM‑стриминга
- веб‑интерфейс для интерактивного тестирования

Всё работает через **Docker Compose** и оптимизировано для **NVIDIA RTX 5090** (но подойдёт для любых GPU с достаточным объёмом VRAM).

---

## Быстрый старт (если модели и референс уже подготовлены)

Если у вас уже есть:
- `checkpoints/fs-1.2-int8-s2-pro-int8/model.pth` (int8‑квантованная модель)
- `checkpoints/s2-pro/codec.pth`
- `references/voice/voice.codes.pt` и `references/voice/voice.lab`

то запуск одной командой:

```bash
git clone <url-репозитория>
cd fish-speech-s2pro-streeming
docker compose up -d
```

Через 1–2 минуты сервисы будут доступны:
- TTS сервер: `http://localhost:8080`
- Прокси: `http://localhost:9000`
- WebUI: `http://localhost:9001`

---

## Полная установка с нуля

### 1. Требования

- **Linux** (Ubuntu 22.04 / 24.04)
- **Docker** и **Docker Compose** (плагин)
- **NVIDIA Container Toolkit** (для GPU)
- **Python 3.10+** и **uv** (для вспомогательных скриптов)
- **ffmpeg** (для конвертации аудио)

### 2. Клонирование и локальные зависимости

```bash
git clone <url>
cd fish-speech-s2pro-streeming
uv sync --extra cu129   # или python -m venv .venv && source .venv/bin/activate && pip install -e .[cu129]
```

### 3. Сборка Docker‑образов

```bash
docker compose build
```

### 4. Скачивание исходной модели (s2-pro)

Модель **s2-pro** содержит кодек `codec.pth` и веса текстовой модели (которые мы потом квантуем). Скачайте через `huggingface-cli` (запуск из временного контейнера):

```bash
mkdir -p checkpoints/s2-pro
docker run --rm -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/venv/bin/huggingface-cli \
  fish-speech-server:latest \
  download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

### 5. Квантизация текстовой модели в int8

Для RTX 5090 настоятельно рекомендуется **int8 weight‑only** квантизация – она сильно снижает потребление VRAM.

```bash
docker run --rm --user root -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/venv/bin/python \
  fish-speech-server:latest \
  tools/llama/quantize.py \
    --checkpoint-path checkpoints/s2-pro \
    --mode int8 \
    --timestamp s2-pro-int8

sudo chown -R $USER:$USER checkpoints/fs-1.2-int8-s2-pro-int8
```

Результат: `checkpoints/fs-1.2-int8-s2-pro-int8/model.pth`.

### 6. Подготовка референсного голоса

Положите образец речи (5–15 секунд, моно, 44100 Гц, без шумов) в папку `input_ref` вместе с текстовой расшифровкой:

```bash
mkdir -p input_ref
# voice.wav и voice.lab
```

Если исходный файл не WAV или длиннее 15 секунд, приведите к формату:

```bash
ffmpeg -i my_voice.mp3 -ac 1 -ar 44100 -c:a pcm_s16le -ss 0 -t 15 input_ref/voice.wav
```

### 7. Предварительное кодирование референса (`.codes.pt`)

Чтобы сервер не тратил время на кодирование при каждом запуске:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace --user root \
  --entrypoint /app/venv/bin/python \
  fish-speech-server:latest \
  tools/preencode_references.py \
    -i input_ref \
    --ref-id voice \
    --checkpoint-path checkpoints/s2-pro/codec.pth \
    --device cpu
```

Появится папка `references/voice` с `voice.codes.pt` и `voice.lab`.

### 8. Настройка конфигурации

Отредактируйте `config/runtime.json`. **Важные параметры для RTX 5090**:

- `model.compile`: `true` – включает JIT‑компиляцию (ускоряет инференс, но первый запуск может длиться 10–15 минут).  
  `false` – быстрый старт, чуть ниже производительность.
- `model.precision = "bfloat16"`
- `model.cache_max_seq_len = 768`
- `playback.target_emit_bytes = 6144`, `playback.start_buffer_ms = 120`
- `frontend_overrides.enabled = false` – отключает проверку полей (избавляет от ошибок 400 при работе WebUI). Рекомендуется.

### 9. Запуск всех сервисов

Используйте **умный скрипт запуска**, который ждёт реальной готовности (особенно важно при `compile: true`):

```bash
chmod +x scripts/start_services.sh
./scripts/start_services.sh
```

Скрипт:
- запускает сервер, читает его логи и ждёт строк `Application startup complete` или `Warmup done`
- затем проверяет health endpoint
- после этого запускает прокси и WebUI

Если вы предпочитаете ручной запуск (без ожидания компиляции):

```bash
docker compose up -d
```

### 10. Проверка работоспособности

```bash
curl http://localhost:8080/v1/health        # {"status":"ok"}
curl http://localhost:9000/health           # {"ok":true,...}
curl http://localhost:9001/health           # {"status":"ok","service":"web-ui"}
```

Откройте в браузере `http://<IP-сервера>:9001` – появится интерфейс для ввода текста и потокового синтеза.

---

## Управление сервисами

- `docker compose down` – остановка всех
- `docker compose restart [service]` – перезапуск
- `docker compose logs -f [service]` – просмотр логов

**Изменение конфигурации:**  
Благодаря тому, что в `compose.yml` добавлен volume `./config:/app/config`, вы можете править `config/runtime.json` на хосте, а затем перезапустить контейнеры – изменения вступят в силу без пересборки образов.

```bash
docker compose restart server proxy
```

---

## Устранение типичных проблем

### 1. Сервер не становится healthy при `compile: true`

- Используйте `scripts/start_services.sh` – он ждёт завершения компиляции по логам, а не по таймауту healthcheck.
- Убедитесь, что контейнеру выделено достаточно памяти: в `docker-compose.override.yml` можно задать `mem_limit: 32g` и `shm_size: 8g`.
- Если компиляция падает с ошибками подпроцессов, добавьте `environment: TORCHINDUCTOR_COMPILE_THREADS: 1` (уже есть в `compose.yml`).

### 2. Прокси падает с `ModuleNotFoundError: No module named 'fish_speech'`

Образ прокси собирается с установкой основного пакета `fish-speech`. Проверьте, что `docker/Dockerfile.proxy` содержит строки:

```dockerfile
COPY pyproject.toml .
COPY fish_speech ./fish_speech
COPY fish_speech_server ./fish_speech_server
RUN pip install --no-cache-dir -e .[cpu]
```

Пересоберите прокси: `docker compose build --no-cache proxy`.

### 3. WebUI возвращает 400 при открытии сессии (disallowed paths)

Отключите проверку полей в `config/runtime.json`:

```json
"frontend_overrides": {
  "enabled": false,
  ...
}
```

Затем перезапустите прокси: `docker compose restart proxy`.

### 4. Порты 8080/9000/9001 уже заняты

Найдите и убейте процесс, занимающий порт, либо измените маппинг в `compose.yml` (например, `"8081:8080"`).

---

## Структура проекта (основные файлы)

```
.
├── compose.yml                     # оркестрация Docker
├── docker/
│   ├── Dockerfile.server           # образ сервера TTS
│   ├── Dockerfile.proxy            # образ прокси
│   └── Dockerfile.webui            # образ WebUI
├── config/
│   └── runtime.json                # единый конфиг (монтируется)
├── checkpoints/                    # модели (монтируются)
├── references/                     # референсные голоса (монтируются)
├── tools/                          # утилиты (квантизация, preencode)
├── scripts/
│   └── start_services.sh           # умный запуск с ожиданием
└── fish_speech_server/             # исходный код TTS сервера
```

---

## Требования к оборудованию

- **VRAM**: ~8–10 ГБ для int8‑квантованной модели + кодек. На RTX 5090 (24 ГБ) работает отлично.
- **RAM хоста**: 16 ГБ при `compile: false`, 32 ГБ рекомендуется при `compile: true`.
- **Драйверы NVIDIA**: версия 545+.
- **Docker**: с поддержкой GPU (NVIDIA Container Toolkit).

---

## Заключение

Проект полностью работоспособен. Рекомендации:

- **Используйте int8 квантизацию** – экономит VRAM.
- **Для быстрого старта отключайте компиляцию** (`compile: false`).
- **Для максимальной производительности** включайте компиляцию и запускайте через `start_services.sh`.
- **Монтирование `config`** позволяет легко менять параметры без пересборки.
- **WebUI** удобен для тестирования стриминга и сессий.

При возникновении проблем сверяйтесь с логами (`docker compose logs`) и проверяйте пути к моделям и референсам.