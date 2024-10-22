import threading
import time

from searchagent.sync_manager.db import insert_file_changes
from searchagent.sync_manager.file_change_detector import Watcher

# TODO: Periodically delete old entries
INTERVAL = 300  # 5 MINS
event_list = []


def periodic_insert():
    while True:
        time.sleep(INTERVAL)
        if event_list:
            insert_file_changes(event_list)
            event_list.clear()


if __name__ == "__main__":
    w = Watcher(".", event_list)
    w.run()

    try:
        # Start the periodic insertion in a separate thread
        insert_thread = threading.Thread(target=periodic_insert)
        insert_thread.daemon = True  # Ensure thread closes when the main program exits
        insert_thread.start()

        # Keep the main program running
        while True:
            time.sleep(1)  # This keeps the main thread alive
    finally:
        w.observer.stop()
        w.observer.join()
