# Fish Speech S2 Pro — стриминговая TTS на RTX 5090

Репозиторий содержит полный стек для запуска **Fish Speech** в стриминговом режиме:
- сервер синтеза речи (FastAPI + PyTorch)
- прокси для управления сессиями и PCM‑стриминга
- веб‑интерфейс для интерактивного тестирования

Всё работает через **Docker Compose** и оптимизировано для **NVIDIA RTX 5090** (но подойдёт и для других GPU с достаточным объёмом VRAM).

---

## Быстрый старт (если уже скачаны модели и квантование)

Если у вас уже есть папка `checkpoints/fs-1.2-int8-s2-pro-int8` (int8‑квантованная модель) и `checkpoints/s2-pro/codec.pth`, а также подготовлен референсный голос в `references/voice/voice.codes.pt` и `references/voice/voice.lab`, то достаточно выполнить:

```bash
git clone <url-репозитория>
cd fish-speech-s2pro-streeming
docker compose up -d
```

Через минуту сервер будет доступен на `http://localhost:8080`, прокси на `http://localhost:9000`, WebUI на `http://localhost:9001`.

---

## Полная установка с нуля (подготовка моделей и референса)

### 1. Требования

- **Linux** (Ubuntu 22.04 / 24.04)
- **Docker** и **Docker Compose** (плагин)
- **NVIDIA Container Toolkit** (для GPU)
- **Python 3.10+** и **uv** (для вспомогательных скриптов)
- **ffmpeg** (для конвертации аудио)

### 2. Клонирование и установка локальных зависимостей

```bash
git clone <url>
cd fish-speech-s2pro-streeming
uv sync --extra cu129   # или просто python -m venv .venv && source .venv/bin/activate && pip install -e .[cu129]
```

### 3. Сборка Docker‑образов

В проекте три сервиса: `server`, `proxy`, `webui`. Для удобства они собираются одной командой:

```bash
docker compose build
```

### 4. Скачивание исходной модели (s2-pro)

Модель **s2-pro** нужна для получения кодека `codec.pth` и весов текстовой модели (которую мы потом квантуем). Скачайте её через `huggingface-cli` (можно запустить из временного контейнера):

```bash
mkdir -p checkpoints/s2-pro
docker run --rm -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/venv/bin/huggingface-cli \
  fish-speech-server:latest \
  download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

### 5. Квантизация текстовой модели в int8

Для RTX 5090 настоятельно рекомендуется использовать **int8 weight‑only** квантизацию. Она заметно снижает потребление VRAM. Запустите скрипт `tools/llama/quantize.py` внутри контейнера:

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

После этого в `checkpoints/fs-1.2-int8-s2-pro-int8` появится `model.pth` (квантованные веса).

### 6. Подготовка референсного голоса

Поместите ваш образец речи (один голос, 5–15 секунд, без шумов, моно 44100 Гц) в папку `input_ref` вместе с текстовой расшифровкой:

```bash
mkdir -p input_ref
# положите voice.wav и voice.lab (текст)
```

Если исходный файл не WAV или длиннее 15 секунд, приведите к нужному формату через `ffmpeg`:

```bash
ffmpeg -i my_voice.mp3 -ac 1 -ar 44100 -c:a pcm_s16le -ss 0 -t 15 input_ref/voice.wav
```

### 7. Предварительное кодирование референса в `.codes.pt`

Чтобы сервер не кодировал референс при каждом запуске, переведите его в готовые токены:

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

В результате появится папка `references/voice` с файлами `voice.codes.pt` и `voice.lab`.

### 8. Настройка конфигурации

Отредактируйте `config/runtime.json` под свои предпочтения. **Важные параметры для RTX 5090**:

- `model.compile = false` (для стабильности и быстрого старта; если очень нужна компиляция, включите, но увеличьте `healthcheck.start_period` до 600 с)
- `model.precision = "bfloat16"`
- `model.cache_max_seq_len = 768`
- `playback.target_emit_bytes = 6144`, `start_buffer_ms = 120`
- `commit.first.*` и `commit.next.*` – настройки, уменьшающие рваную интонацию

**Прокси** читает `frontend_overrides.allowed_paths`. Чтобы избежать ошибок 400 при работе WebUI, либо добавьте в `allowed_paths` все поля из конфига WebUI, либо просто отключите проверку: `"enabled": false`. Мы рекомендуем отключить проверку для простоты.

### 9. Запуск всех сервисов

```bash
docker compose up -d
```

Скрипт ожидания готовности (опционально) можно использовать из `scripts/start_services.sh` (он последовательно проверяет health каждого сервиса). Убедитесь, что файл исполняемый:

```bash
chmod +x scripts/start_services.sh
./scripts/start_services.sh
```

### 10. Проверка

```bash
curl http://localhost:8080/v1/health        # должен вернуть {"status":"ok"}
curl http://localhost:9000/health           # {"ok":true,...}
curl http://localhost:9001/health           # {"status":"ok","service":"web-ui"}
```

Откройте в браузере `http://ваш-сервер:9001`. Там можно открыть сессию, ввести текст, запустить стриминг и услышать синтезированную речь.

---

## Управление сервисами

- Останов: `docker compose down`
- Перезапуск: `docker compose restart`
- Просмотр логов: `docker compose logs -f [service]`

Если вы меняете `config/runtime.json` на хосте, перезапустите прокси и сервер, чтобы изменения вступили в силу (благодаря монтированию тома `./config:/app/config`).

---

## Устранение типичных проблем

### 1. Сервер не отвечает, контейнер не healthy

- Проверьте логи: `docker compose logs server --tail 50`
- Убедитесь, что `checkpoints/fs-1.2-int8-s2-pro-int8/model.pth` и `checkpoints/s2-pro/codec.pth` существуют.
- Если включена компиляция (`COMPILE=1`), увеличьте `start_period` в `healthcheck` до 600 с или отключите компиляцию.

### 2. Прокси падает с `ModuleNotFoundError: No module named 'fish_speech'`

Образ прокси должен включать установку основного пакета `fish-speech`. Исправленный `docker/Dockerfile.proxy` приведён в репозитории. Пересоберите его:

```bash
docker compose build --no-cache proxy
```

### 3. WebUI возвращает 400 при открытии сессии (disallowed paths)

Либо отключите проверку в `config/runtime.json`: `"frontend_overrides": { "enabled": false }`, либо добавьте недостающие пути в `allowed_paths`. После изменения перезапустите прокси.

### 4. Порты 9000 или 8080 уже заняты

Найдите и убейте процесс, занимающий порт, или измените маппинг портов в `compose.yml`.

---

## Структура проекта (важные файлы)

- `compose.yml` – основной файл оркестрации Docker.
- `docker/Dockerfile.server`, `Dockerfile.proxy`, `Dockerfile.webui` – образы сервисов.
- `config/runtime.json` – единая конфигурация для сервера и прокси (монтируется в контейнеры).
- `checkpoints/` – модели (монтируются).
- `references/` – референсные голоса (монтируются).
- `tools/` – утилиты для квантизации, предкодирования, тестовый клиент.
- `scripts/start_services.sh` – пошаговый запуск с ожиданием health.

---

## Требования к RTX 5090 (и другим GPU)

- **VRAM**: ~8–10 ГБ для int8‑квантованной модели + кодек. При 24 ГБ (5090) работает без проблем.
- **RAM хоста**: 32 ГБ рекомендуется для компиляции (если она включена). При `COMPILE=0` достаточно 16 ГБ.
- **Драйверы NVIDIA**: версия 545+.

---

## Заключение

Проект полностью работоспособен и готов к использованию. Основные рекомендации:

- Используйте **int8** квантизацию.
- Отключайте компиляцию (`COMPILE=0`), если не нужна максимальная скорость после прогрева.
- Монтируйте `config` для быстрой смены параметров.
- Пользуйтесь WebUI для удобного тестирования стриминга.

При возникновении проблем сверяйтесь с логами и проверяйте пути к моделям и референсам.