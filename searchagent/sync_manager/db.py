import sqlite3

from searchagent.sync_manager.file_change_detector import FileChangeEvent, List


def insert_file_changes(event_list: List[FileChangeEvent]):
    DB_NAME = "file_changes"
    con = sqlite3.connect(f"{DB_NAME}.db")
    cur = con.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB_NAME} (
            filepath TEXT PRIMARY KEY,
            change_type TEXT,
            timestamp TEXT
        )
        """
    )
    for event in event_list:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {DB_NAME} (filepath, change_type, timestamp)
            VALUES (?, ?, ?)
            """,
            (event.filepath, event.change_type.value, event.timestamp),
        )
    con.commit()
    con.close()
