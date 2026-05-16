import re


def hms_to_seconds(time_str: str) -> float:
    """
    Convert HH:MM:SS.mmm or HH:MM:SS,mmm to seconds.

    Examples:
        "00:01:23.456" -> 83.456
        "00:01:23,456" -> 83.456
    """
    if time_str is None:
        raise ValueError("time_str is None")

    time_str = str(time_str).strip()
    time_str = time_str.replace(",", ".")

    pattern = r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$"
    match = re.match(pattern, time_str)

    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = match.group(4)

    total = hours * 3600 + minutes * 60 + seconds

    if milliseconds:
        total += float("0." + milliseconds)

    return total


def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")

    hours = int(seconds // 3600)
    remaining = seconds % 3600
    minutes = int(remaining // 60)
    sec = remaining % 60

    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def seconds_to_frame(seconds: float, fps: float) -> int:
    """
    Convert seconds to frame index.
    """
    return int(round(seconds * fps))


def frame_to_seconds(frame: int, fps: float) -> float:
    """
    Convert frame index to seconds.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    return frame / fps


def duration_seconds(start: float, end: float) -> float:
    """
    Compute duration from start and end seconds.
    """
    return max(0.0, end - start)