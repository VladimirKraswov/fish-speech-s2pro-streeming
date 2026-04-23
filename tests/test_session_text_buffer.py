from session_mode.buffer import StreamingTextBuffer


def test_does_not_emit_first_chunk_before_safe_boundary():
    buf = StreamingTextBuffer(min_words=3, soft_limit_chars=80, hard_limit_chars=120)

    assert buf.push("Привет это час") == []
    emits = buf.push("тичный тест.")

    assert len(emits) == 1
    assert emits[0].text == "Привет это частичный тест."
    assert emits[0].reason == "punct"


def test_hard_limit_prefers_whitespace_boundary():
    buf = StreamingTextBuffer(min_words=3, soft_limit_chars=40, hard_limit_chars=58)
    text = "Один два три четыре пять шесть семь восемь девять десять одиннадцать."

    emits = buf.push(text)

    assert emits
    assert emits[0].reason in {"force", "hard_limit"}
    assert emits[0].text == "Один два три четыре пять шесть семь восемь девять десять"
    assert buf.text == "одиннадцать."


def test_force_flush_keeps_partial_tail_buffered():
    buf = StreamingTextBuffer(min_words=3, soft_limit_chars=80, hard_limit_chars=120)
    buf.push("один два три незав")

    emits = buf.flush(final=False)

    assert len(emits) == 1
    assert emits[0].text == "один два три"
    assert buf.text == "незав"


def test_soft_limit_does_not_force_unpunctuated_phrase():
    buf = StreamingTextBuffer(min_words=3, soft_limit_chars=30, hard_limit_chars=120)

    emits = buf.push("Текст должен делиться только на ")

    assert emits == []
    assert buf.text == "Текст должен делиться только на"


def test_final_flush_returns_remaining_tail():
    buf = StreamingTextBuffer(min_words=3, soft_limit_chars=80, hard_limit_chars=120)
    buf.push("один два три незав")

    emits = buf.flush(final=True)

    assert len(emits) == 1
    assert emits[0].text == "один два три незав"
    assert buf.empty()
