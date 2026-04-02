def to_hours_minutes_seconds(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h:{int(m):02d}m:{int(s):02d}s"