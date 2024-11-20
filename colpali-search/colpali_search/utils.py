from datetime import datetime, timezone
from uuid import uuid4


def get_now():
    return datetime.now(timezone.utc).isoformat()


def generate_uuid() -> str:
    return str(uuid4())
