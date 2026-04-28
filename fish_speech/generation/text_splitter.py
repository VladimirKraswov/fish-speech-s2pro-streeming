import re


def split_long_text(
    text: str,
    *,
    target_chars: int = 220,
    max_chars: int = 320,
) -> list[str]:
    """
    Split a long text into smaller speech segments.
    """

    if not text:
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # 1. Split into sentences by strong punctuation
    # . ! ? … 。 ！ ？ ;
    pieces = _split_by_delimiters(text, r"([.!?…。！？;]+)")

    segments = []
    current_segment = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if len(piece) > max_chars:
            # Flush current segment
            if current_segment:
                segments.append(current_segment)
                current_segment = ""

            # Split oversized piece recursively
            segments.extend(
                _split_oversized_piece(piece, target_chars=target_chars, max_chars=max_chars)
            )
            continue

        # Check if adding this piece exceeds max_chars
        sep = " " if current_segment else ""
        if len(current_segment) + len(sep) + len(piece) > max_chars:
            if current_segment:
                segments.append(current_segment)
            current_segment = piece
        else:
            current_segment = current_segment + sep + piece

        # If we reached target_chars, we can close this segment
        if len(current_segment) >= target_chars:
            segments.append(current_segment)
            current_segment = ""

    if current_segment:
        segments.append(current_segment)

    return [s.strip() for s in segments if s.strip()]


def _split_by_delimiters(text: str, pattern: str) -> list[str]:
    """
    Split text by pattern but keep the delimiters.
    """
    parts = re.split(pattern, text)
    result = []
    for i in range(0, len(parts) - 1, 2):
        combined = parts[i] + parts[i + 1]
        if combined:
            result.append(combined)

    if len(parts) % 2 == 1 and parts[-1]:
        result.append(parts[-1])

    return result


def _split_oversized_piece(text: str, target_chars: int, max_chars: int) -> list[str]:
    """
    Split a piece that is longer than max_chars using medium punctuation or spaces.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Try medium punctuation: , : — – -
    pieces = _split_by_delimiters(text, r"([,:—–-]+)")
    if len(pieces) > 1:
        return _group_pieces(pieces, target_chars, max_chars, separator=" ")

    # Try spaces
    pieces = text.split(" ")
    if len(pieces) > 1:
        return _group_pieces(pieces, target_chars, max_chars, separator=" ")

    # Last resort: hard cut
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _group_pieces(
    pieces: list[str], target_chars: int, max_chars: int, separator: str
) -> list[str]:
    """
    Group pieces into segments according to character limits.
    """
    segments = []
    current_segment = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if len(piece) > max_chars:
            if current_segment:
                segments.append(current_segment)
                current_segment = ""
            segments.extend(_split_oversized_piece(piece, target_chars, max_chars))
            continue

        sep = separator if current_segment else ""
        if len(current_segment) + len(sep) + len(piece) > max_chars:
            if current_segment:
                segments.append(current_segment)
            current_segment = piece
        else:
            current_segment = current_segment + sep + piece

        if len(current_segment) >= target_chars:
            segments.append(current_segment)
            current_segment = ""

    if current_segment:
        segments.append(current_segment)

    return segments
