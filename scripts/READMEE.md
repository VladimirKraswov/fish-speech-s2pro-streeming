Ниже готовый `README.md` для папки `scripts` и всего контура запуска.

````md
# Fish Speech: запуск стриминга и session mode

Этот набор скриптов поднимает полный runtime для стриминговой озвучки и работы в режиме сессий.

## Что поднимается

Стек состоит из трех уровней:

1. **Основной TTS API**  
   Запускается в Docker-контейнере.  
   Отвечает за генерацию аудио и декодирование.  
   Обычно слушает `http://127.0.0.1:8080`.

2. **Proxy слой**  
   Удобная тонкая прослойка для PCM/streaming сценариев.  
   Обычно слушает `http://127.0.0.1:9000`.

3. **Session Mode**  
   Менеджер realtime-сессий:
   - принимает поток текстовых дельт,
   - буферизует текст,
   - режет его на чанки,
   - отправляет чанки в основной TTS backend,
   - отдает события и бинарные аудио чанки по WebSocket.  
   Обычно слушает `http://127.0.0.1:8765`.

---

## Быстрый старт

### Поднять всё сразу

```bash
bash scripts/up_5090.sh
````

Это:

* при необходимости соберет Docker image,
* остановит старые процессы,
* поднимет контейнер с моделью,
* дождется `health`,
* выполнит дополнительный warmup,
* поднимет proxy,
* поднимет session mode.

После этого будут доступны:

* Model API: `http://127.0.0.1:8080/v1/health`
* Proxy: `http://127.0.0.1:9000/health`
* Session mode: `http://127.0.0.1:8765/health`

### Остановить всё

```bash
bash scripts/down_5090.sh
```

### Перезапустить всё

```bash
bash scripts/restart_5090.sh
```

### Смотреть логи

```bash
bash scripts/logs_5090.sh
```

---

## Основные сценарии запуска

## 1. Только модельный API

Если нужен только основной TTS backend:

```bash
bash scripts/run_server_32gb.sh
```

По умолчанию:

* порт: `8080`
* контейнер: `fish-speech`
* image: `fish-speech-webui:cu129`

### Полезные env-переменные

```bash
PORT=8080
CONTAINER=fish-speech
IMAGE=fish-speech-webui:cu129
COMPILE=1
CHECKPOINTS_DIR=checkpoints/s2-pro
FISH_CACHE_MAX_SEQ_LEN=320
FISH_MAX_NEW_TOKENS_CAP=64
```

Пример:

```bash
PORT=8080 COMPILE=1 bash scripts/run_server_32gb.sh
```

---

## 2. Только proxy

```bash
bash scripts/run_proxy.sh
```

По умолчанию:

* host: `0.0.0.0`
* port: `9000`

### Настройки

```bash
PROXY_HOST=0.0.0.0
PROXY_PORT=9000
PROXY_LOG_LEVEL=info
PYTHON_BIN=python3
```

Пример:

```bash
PROXY_PORT=9000 bash scripts/run_proxy.sh
```

Health:

```bash
curl http://127.0.0.1:9000/health
```

---

## 3. Только session mode

```bash
bash scripts/run_session_mode.sh
```

По умолчанию:

* host: `0.0.0.0`
* port: `8765`

### Настройки

```bash
SESSION_HOST=0.0.0.0
SESSION_PORT=8765
SESSION_LOG_LEVEL=info
PYTHON_BIN=python3
```

Пример:

```bash
SESSION_PORT=8765 bash scripts/run_session_mode.sh
```

Health:

```bash
curl http://127.0.0.1:8765/health
```

---

## 4. Полный стек одной командой

Рекомендуемый режим для локальной разработки и e2e-тестов:

```bash
bash scripts/up_5090.sh
```

### Что делает `up_5090.sh`

1. Проверяет наличие checkpoint'ов.
2. При необходимости собирает Docker image.
3. Останавливает старый контейнер, proxy и session mode.
4. Поднимает контейнер с моделью.
5. Ждет `http://127.0.0.1:8080/v1/health`.
6. Делает дополнительный warmup стриминга.
7. Поднимает proxy.
8. Поднимает session mode.
9. Проверяет health всех сервисов.

### Ключевые env-переменные

```bash
IMAGE=fish-speech-webui:cu129
CONTAINER=fish-speech
PORT=8080
PROXY_PORT=9000
SESSION_PORT=8765

COMPILE=1
BUILD_IMAGE=0
START_PROXY=1
START_SESSION=1
EXTRA_WARMUP=1

CHECKPOINTS_DIR=checkpoints/s2-pro
FISH_CACHE_MAX_SEQ_LEN=320
FISH_MAX_NEW_TOKENS_CAP=64
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HEALTH_TIMEOUT=1800
DOCKER_USE_SUDO=0
```

Пример:

```bash
COMPILE=1 START_PROXY=1 START_SESSION=1 bash scripts/up_5090.sh
```

---

## Warmup

Дополнительный прогрев полезен, если нужен более стабильный первый ответ после запуска.

```bash
bash scripts/warmup_5090.sh
```

По умолчанию скрипт:

* ждет готовности `v1/health`,
* отправляет один streaming TTS запрос,
* сохраняет результат в `logs/warmup_stream.wav`.

### Настройки

```bash
PORT=8080
BASE_URL=http://127.0.0.1:8080
WARMUP_TIMEOUT=1800
WARMUP_TEXT="Привет. Это дополнительный прогрев."
OUT_FILE=logs/warmup_stream.wav
```

---

## Проверка сервиса

## Быстрая e2e smoke-проверка

```bash
bash scripts/e2e_smoke.sh
```

Проверяет:

* готовность `/v1/health`,
* streaming TTS,
* oneshot TTS,
* `/v1/debug/memory`.

Результаты пишет в:

```bash
results/e2e
```

## Проверка памяти между запросами

```bash
bash scripts/e2e_memory.sh
```

Полезно для контроля:

* освобождения VRAM между запросами,
* отсутствия OOM на втором запросе.

---

## Измерение TTFA

Скрипт `ttfa_smoke.py` делает один TTS запрос и печатает:

* `ttfa_s` — время до первого байта,
* `ttfa_audio_s` — время до первого аудио-байта,
* `total_s` — полное время запроса.

### Streaming

```bash
python3 scripts/ttfa_smoke.py \
  --url http://127.0.0.1:8080 \
  --output out.wav
```

### Oneshot

```bash
python3 scripts/ttfa_smoke.py \
  --url http://127.0.0.1:8080 \
  --output out.wav \
  --oneshot
```

### С reference_id

```bash
python3 scripts/ttfa_smoke.py \
  --url http://127.0.0.1:8080 \
  --output out.wav \
  --reference-id my_voice
```

---

## Работа с reference voice

## Pre-encode reference файлов

Если есть папка с reference WAV и текстами, можно заранее закодировать references:

```bash
bash scripts/preencode.sh
```

По умолчанию:

* input: `data/voice_references`
* output: `references`
* checkpoint: `checkpoints/s2-pro/codec.pth`

### Настройки

```bash
PREENCODE_INPUT_DIR=data/voice_references
PREENCODE_OUTPUT_DIR=references
PREENCODE_CHECKPOINT=checkpoints/s2-pro/codec.pth
PREENCODE_REF_ID=
UPLOAD=0
SERVER_URL=http://127.0.0.1:8080
```

Пример с немедленной загрузкой на сервер:

```bash
UPLOAD=1 bash scripts/preencode.sh
```

## Upload уже подготовленных references

```bash
bash scripts/upload_references.sh
```

Настройки:

```bash
PREENCODE_OUTPUT_DIR=references
SERVER_URL=http://127.0.0.1:8080
```

---

## Health endpoints

## Model API

```bash
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/debug/memory
```

## Proxy

```bash
curl http://127.0.0.1:9000/health
```

Пример стриминга через proxy:

```bash
curl "http://127.0.0.1:9000/pcm-stream?text=Привет"
```

## Session mode

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/sessions
```

---

## Где лежат логи и pid-файлы

### Логи

```bash
logs/proxy.log
logs/session_mode.log
```

Логи контейнера модели смотрятся через:

```bash
docker logs -f fish-speech
```

или через:

```bash
bash scripts/logs_5090.sh
```

### PID-файлы

```bash
run/proxy.pid
run/session_mode.pid
```

---

## Типичный рабочий цикл

## Вариант A: всё поднять и работать

```bash
bash scripts/up_5090.sh
bash scripts/logs_5090.sh
```

## Вариант B: модель отдельно, session отдельно

```bash
bash scripts/run_server_32gb.sh
bash scripts/run_session_mode.sh
```

## Вариант C: прогрев + тест

```bash
bash scripts/warmup_5090.sh
bash scripts/e2e_smoke.sh
```

---

## Рекомендованные значения для 32GB / 5090

Это не жесткие требования, а хорошие стартовые значения:

```bash
COMPILE=1
FISH_CACHE_MAX_SEQ_LEN=320
FISH_MAX_NEW_TOKENS_CAP=64
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA_WARMUP=1
```

Если упирается в память:

* уменьшить `FISH_CACHE_MAX_SEQ_LEN`,
* уменьшить `FISH_MAX_NEW_TOKENS_CAP`,
* временно выключить `COMPILE=0` для диагностики.

---

## Troubleshooting

## Session mode не стартует

Проверь:

```bash
cat logs/session_mode.log
```

Убедись, что:

* активен нужный Python,
* установлены зависимости,
* модуль `session_mode.app:app` импортируется без ошибок.

## Proxy не стартует

Проверь:

```bash
cat logs/proxy.log
```

Также проверь порт:

```bash
lsof -i :9000
```

## Контейнер модели падает при старте

Смотри:

```bash
docker logs fish-speech
```

Частые причины:

* отсутствуют checkpoints,
* нехватка VRAM,
* неправильный CUDA runtime,
* слишком агрессивные параметры compile/cache.

## Health долго не поднимается

Это нормально при:

* первом запуске после сборки image,
* `COMPILE=1`,
* холодном запуске с прогревом.

В таком случае просто увеличь timeout:

```bash
HEALTH_TIMEOUT=1800 bash scripts/up_5090.sh
```

---

## Минимальные команды

### Старт всего

```bash
bash scripts/up_5090.sh
```

### Стоп всего

```bash
bash scripts/down_5090.sh
```

### Рестарт всего

```bash
bash scripts/restart_5090.sh
```

### Логи

```bash
bash scripts/logs_5090.sh
```

### Только session mode

```bash
bash scripts/run_session_mode.sh
```

### Только proxy

```bash
bash scripts/run_proxy.sh
```

### Только model API

```bash
bash scripts/run_server_32gb.sh
```

### Smoke test

```bash
bash scripts/e2e_smoke.sh
```

### Memory test

```bash
bash scripts/e2e_memory.sh
```

---

## Примечание

Эти скрипты рассчитаны на локальную разработку и ручной dev/runtime orchestration.
Если позже понадобится production-режим, лучше вынести:

* общие env в `.env`,
* сервисы в `docker compose` или process manager,
* health/warmup/checks в отдельный orchestration слой.

```

Могу сразу дать еще и короткий `README-session-mode.md` только для WebSocket/session API.
```
