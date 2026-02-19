from threading import Lock
from typing import Optional

_session_collections = {}
_lock = Lock()


def set_session_collection(session_id: str, collection_name: str) -> None:
    with _lock:
        _session_collections[session_id] = collection_name


def get_session_collection(session_id: str) -> Optional[str]:
    with _lock:
        return _session_collections.get(session_id)


def clear_session_collection(session_id: str) -> None:
    with _lock:
        _session_collections.pop(session_id, None)
