# Разворачивание и запуск проекта

Проект рассчитан на запуск **Fish Speech S2 Pro** в стриминговом режиме на **RTX 5090**.
Рекомендуемый сценарий такой:

1. клонируем репозиторий
2. ставим зависимости
3. собираем Docker-образ
4. скачиваем исходную модель
5. квантуем текстовую модель в `int8`
6. готовим референсный голос
7. при необходимости конвертируем референс в `wav` и подрезаем до **15 секунд**
8. предкодируем референс в `voice.codes.pt`
9. проверяем `config/runtime.json`
10. только после этого запускаем сервер

---

## Требования

Нужны:

* Linux-машина с **NVIDIA GPU**
* **RTX 5090** или другая совместимая карта
* установленный **Docker**
* установленный **NVIDIA Container Toolkit**
* установленный **Python** и **uv**
* достаточно места под:

  * исходную модель `s2-pro`
  * квантизованную модель `int8`
  * Docker-образ

---

## 1. Клонирование репозитория

```bash
git clone <URL_ВАШЕГО_РЕПО>
cd fish-speech-s2pro-streeming
```

---

## 2. Установка локальных зависимостей

```bash
uv sync --extra cu129
```

Это полезно для локальной работы с репозиторием и вспомогательными скриптами.
Основной сервер при этом всё равно запускается через Docker.

---

## 3. Сборка Docker-образа

Для квантизации и предкодирования референсов удобнее сначала собрать образ вручную:

```bash
docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  --build-arg BACKEND=cuda \
  --build-arg CUDA_VER=12.9.0 \
  --build-arg UV_EXTRA=cu129 \
  --target webui \
  -t fish-speech-webui:cu129 .
```

> `scripts/up_5090.sh` тоже умеет собрать образ автоматически, но для шагов квантизации и `preencode` образ нужен заранее.

---

## 4. Скачивание исходной модели `s2-pro`

Исходную модель нужно положить в:

```text
checkpoints/s2-pro
```

Удобный способ скачать её через уже собранный Docker-образ:

```bash
mkdir -p checkpoints/s2-pro

docker run --rm \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/.venv/bin/huggingface-cli \
  fish-speech-webui:cu129 \
  download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

После этого проверьте, что у вас есть как минимум:

```text
checkpoints/s2-pro/config.json
checkpoints/s2-pro/tokenizer*
checkpoints/s2-pro/model*
checkpoints/s2-pro/codec.pth
```

---

## 5. Квантизация модели в `int8`

Для RTX 5090 безопаснее использовать **int8 weight-only checkpoint**: он заметно снижает VRAM-нагрузку и даёт запас под `compile + warmup + streaming`.

Запуск квантизации:

```bash
docker run --rm --user root \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/.venv/bin/python \
  fish-speech-webui:cu129 \
  tools/llama/quantize.py \
    --checkpoint-path checkpoints/s2-pro \
    --mode int8 \
    --timestamp s2-pro-int8
```

Результат появится в папке:

```text
checkpoints/fs-1.2-int8-s2-pro-int8
```

Если запускали с `sudo` или `--user root`, сразу выровняйте права:

```bash
sudo chown -R $USER:$USER checkpoints/fs-1.2-int8-s2-pro-int8
```

### Важно

Квантизация затрагивает только **текстовую модель**.
Файл `codec.pth` нужно продолжать брать из исходной папки:

```text
checkpoints/s2-pro/codec.pth
```

---

## 6. Подготовка референса

Нужен один качественный референсный голос:

* **один диктор**
* без музыки, эха и сильного шума
* лучше всего **10–15 секунд**
* обязательна точная расшифровка в `.lab`

Создайте временную папку:

```bash
mkdir -p input_ref
```

---

## 7. Конвертация референса в `wav` и ограничение до 15 секунд

Если исходник не в `wav`, или он длиннее 15 секунд, сразу приведите его к нужному виду.

### Если у вас MP3 / M4A / OGG

```bash
ffmpeg -i my_voice.mp3 \
  -ac 1 \
  -ar 44100 \
  -c:a pcm_s16le \
  -ss 0 \
  -t 15 \
  input_ref/voice.wav
```

### Если у вас уже WAV, но его нужно подрезать и нормализовать

```bash
ffmpeg -i my_voice.wav \
  -ac 1 \
  -ar 44100 \
  -c:a pcm_s16le \
  -ss 0 \
  -t 15 \
  input_ref/voice.wav
```

### Расшифровка

Создайте файл:

```text
input_ref/voice.lab
```

Пример:

```text
В очередной раз захожу в кабинет ректора, готовая уже отчислиться из аспирантуры.
```

### Важно

Имена должны совпадать по stem:

```text
input_ref/voice.wav
input_ref/voice.lab
```

---

## 8. Предкодирование референса в `voice.codes.pt`

Чтобы сервер не кодировал референс заново при работе, его лучше заранее перевести в `.codes.pt`.

Команда:

```bash
docker run --rm \
  -v "$PWD":/workspace -w /workspace \
  --user root \
  --entrypoint /app/.venv/bin/python \
  fish-speech-webui:cu129 \
  tools/preencode_references.py \
    -i input_ref \
    --ref-id voice \
    --checkpoint-path checkpoints/s2-pro/codec.pth \
    --device cpu
```

После этого должна появиться папка:

```text
references/voice
```

И внутри:

```text
references/voice/voice.codes.pt
references/voice/voice.lab
```

Проверьте:

```bash
ls -la references/voice
```

---

## 9. Проверка `config/runtime.json`

Перед запуском обязательно проверьте `config/runtime.json`.

### Что важно проверить

#### Пути

```json
"paths": {
  "llama_checkpoint_path": "checkpoints/s2-pro",
  "decoder_checkpoint_path": "checkpoints/s2-pro/codec.pth",
  "references_dir": "references"
}
```

> Даже если потом вы передадите путь к `int8` модели через переменную окружения для запуска, файл всё равно должен быть консистентным.

#### Дефолтный референс

```json
"warmup": {
  "reference_id": "voice"
}
```

```json
"proxy": {
  "default_reference_id": "voice",
  "tts": {
    "reference_id": "voice"
  }
}
```

#### Рекомендуемый профиль под 5090

Рекомендуемые значения:

* `precision = "bfloat16"`
* `compile = true`
* `cache_max_seq_len = 768`
* `max_new_tokens_cap = 160`
* `cleanup_every_n_requests = 24`
* `initial_stream_chunk_size = 10`
* `stream_chunk_size = 8`
* `target_emit_bytes = 6144`
* `start_buffer_ms = 120`

### Почему именно такие параметры

* `int8` checkpoint по умолчанию сильно снижает VRAM-давление и даёт запас под `compile + warmup + streaming`
* `bfloat16` даёт хороший баланс скорости и стабильности на 5090
* `cache_max_seq_len = 768` — KV cache не слишком большой, но и не тесный для типового референса и рабочего чанка
* `max_new_tokens_cap = 160` — меньше риск обрыва фразы, чем на `128`, но без лишнего раздувания
* `cleanup_every_n_requests = 24` — помогает периодически чистить фрагментацию без чистки после каждого запроса
* `initial_stream_chunk_size = 10`, `stream_chunk_size = 8` — первый звук приходит быстро, но старт не слишком дробный
* `target_emit_bytes = 6144` и `start_buffer_ms = 120` — заметно меньше шанс на рывки на старте
* более аккуратные `commit.first` и `commit.next` уменьшают рваную интонацию и проглатывание кусков текста

> Важно: фактические runtime-параметры читаются из `config/runtime.json`. Если вы меняете только `FISH_*` переменные в shell-скрипте, этого недостаточно, пока код явно не подхватывает такие env-переменные. Поэтому перед запуском ориентируйтесь прежде всего на сам `config/runtime.json`.

---

## 10. Опционально: принудительный дефолтный референс для прямого `/v1/tts`

Если вы работаете **через proxy**, обычно достаточно `proxy.default_reference_id = "voice"` и `proxy.tts.reference_id = "voice"`.

Если же вы хотите, чтобы **прямой вызов** `/v1/tts` тоже автоматически подставлял `voice`, можно добавить это в `tools/server/views.py` внутри функции `tts`:

```python
if req.reference_id is None:
    req.reference_id = "voice"
```

Если не хотите править код, просто всегда передавайте в запросах:

```json
{
  "reference_id": "voice"
}
```

---

## 11. Запуск

После того как:

* скачана исходная модель
* сделана `int8` квантизация
* подготовлен референс
* создан `voice.codes.pt`
* проверен `runtime.json`

можно запускать сервер.

### Рекомендуемый запуск

```bash
LLAMA_CHECKPOINTS_DIR=checkpoints/fs-1.2-int8-s2-pro-int8 \
DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
DEFAULT_REFERENCE_ID=voice \
FISH_WARMUP_REFERENCE_ID=voice \
bash scripts/up_5090.sh
```

### Минимальный запуск

```bash
uv sync --extra cu129
bash scripts/up_5090.sh
```

или
```
DOCKER_USE_SUDO=1 bash scripts/up_5090.sh
```

---

## 12. Управление сервисом

### Остановить

```bash
bash scripts/down_5090.sh
```

### Перезапустить

```bash
bash scripts/restart_5090.sh
```

### Посмотреть логи

```bash
bash scripts/logs_5090.sh
```

---

## 13. Быстрая проверка после запуска

### Health-check модели

```bash
curl -s http://127.0.0.1:8080/v1/health
```

### Проверка proxy

```bash
curl -s http://127.0.0.1:9000/health
```

### Тестовый TTS-запрос

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Привет, это тестовый синтез.","reference_id":"voice","streaming":true}' \
  --output test.wav
```

---

## 14. Типовые проблемы

### Нет модели в `checkpoints/s2-pro`

`scripts/up_5090.sh` не скачивает её сам.
Сначала нужно руками скачать исходный checkpoint.

### Нет `voice.codes.pt`

Сервер сможет работать и без него, но будет тратить время на повторное кодирование референса. Для стабильного старта лучше всегда делать `preencode`.

### Референс длинный или грязный

Лучше держать референс **до 15 секунд**, с чистой дикцией и точной расшифровкой. Это уменьшает latency и делает голос стабильнее.

### После квантизации не хватает прав

Исправьте:

```bash
sudo chown -R $USER:$USER checkpoints references
```

---

## 15. Краткий рабочий сценарий

Если совсем кратко, рабочий флоу такой:

```bash
git clone <URL_ВАШЕГО_РЕПО>
cd fish-speech-s2pro-streeming

uv sync --extra cu129

docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  --build-arg BACKEND=cuda \
  --build-arg CUDA_VER=12.9.0 \
  --build-arg UV_EXTRA=cu129 \
  --target webui \
  -t fish-speech-webui:cu129 .

docker run --rm \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/.venv/bin/huggingface-cli \
  fish-speech-webui:cu129 \
  download fishaudio/s2-pro --local-dir checkpoints/s2-pro

docker run --rm --user root \
  -v "$PWD":/workspace -w /workspace \
  --entrypoint /app/.venv/bin/python \
  fish-speech-webui:cu129 \
  tools/llama/quantize.py \
    --checkpoint-path checkpoints/s2-pro \
    --mode int8 \
    --timestamp s2-pro-int8

ffmpeg -i my_voice.mp3 -ac 1 -ar 44100 -c:a pcm_s16le -ss 0 -t 15 input_ref/voice.wav

docker run --rm \
  -v "$PWD":/workspace -w /workspace \
  --user root \
  --entrypoint /app/.venv/bin/python \
  fish-speech-webui:cu129 \
  tools/preencode_references.py \
    -i input_ref \
    --ref-id voice \
    --checkpoint-path checkpoints/s2-pro/codec.pth \
    --device cpu

LLAMA_CHECKPOINTS_DIR=checkpoints/fs-1.2-int8-s2-pro-int8 \
DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
DEFAULT_REFERENCE_ID=voice \
FISH_WARMUP_REFERENCE_ID=voice \
bash scripts/up_5090.sh
```

Отдельно отмечу: в присланном бандле я не увидел `scripts/status_5090.sh`, поэтому не включал его как обязательную команду. Если он у тебя уже есть локально, можно просто добавить ещё один раздел:

```bash
bash scripts/status_5090.sh
```