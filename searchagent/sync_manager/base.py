import threading
import time

from searchagent.sync_manager.db import insert_data
from searchagent.sync_manager.file_change_detector import Watcher, FileChangeEventList

INTERVAL = 300  # 5 MINS


def periodic_insert(event_list: FileChangeEventList):
    while True:
        time.sleep(INTERVAL)
        if event_list:
            insert_data(event_list)
            event_list.clear()


def monitor(input_dir: str):

    # TODO: Periodically delete old entries
    # TODO: Allow more than one dir
    event_list: FileChangeEventList = []

    w = Watcher(input_dir, event_list)
    w.run()

    try:
        # Start the periodic insertion in a separate thread
        insert_thread = threading.Thread(target=periodic_insert(event_list))
        insert_thread.daemon = True  # Ensure thread closes when the main program exits
        insert_thread.start()

        # Keep the main program running
        while True:
            time.sleep(1)  # This keeps the main thread alive
    finally:
        w.observer.stop()
        w.observer.join()
