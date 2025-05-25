from datetime import datetime


def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")
