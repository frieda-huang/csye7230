from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List

from searchagent.utils import get_now
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


class FileChangeType(Enum):
    MOVED = "moved"
    MODIFIED = "modified"
    DELETED = "deleted"
    CREATED = "created"


@dataclass
class FileChangeEvent:
    filepath: str
    change_type: FileChangeType
    timestamp: datetime


class Watcher:
    def __init__(self, path, event_list: List[FileChangeEvent]) -> None:
        self.observer = Observer()
        self.path = path
        self.event_list = event_list

    def run(self):
        event_handler = EventHandler(self.event_list)
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()


class EventHandler(FileSystemEventHandler):
    def __init__(self, event_list: List[FileChangeEvent]):
        self.event_list = event_list
        self.last_modified = datetime.now()

    def create_event(
        self, event: DirCreatedEvent | FileCreatedEvent, change_type: FileChangeType
    ) -> FileChangeEvent:
        return FileChangeEvent(
            filepath=event.src_path, change_type=change_type, timestamp=get_now()
        )

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        if not event.is_directory:
            self.event_list.append(self.create_event(event, FileChangeType.CREATED))

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        if not event.is_directory:
            self.event_list.append(self.create_event(event, FileChangeType.DELETED))

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        if not event.is_directory:
            if datetime.now() - self.last_modified < timedelta(seconds=1):
                return
            else:
                self.last_modified = datetime.now()
                self.event_list.append(
                    self.create_event(event, FileChangeType.MODIFIED)
                )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if not event.is_directory:
            self.event_list.append(self.create_event(event, FileChangeType.MOVED))
