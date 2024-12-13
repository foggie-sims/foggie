import time, datetime, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def monitor_for_file_watchdog(file_to_monitor):
    print('\nThis is automatic mode. Will keep monitoring for', file_to_monitor)
    class ExampleHandler(FileSystemEventHandler):
        def on_created(self, event):  # when file is created
            print('Deb:', event.src_path)
            if event.src_path == file_to_monitor:
                observer.stop()
                print('Found it')
            elif event.is_directory:
                print(event.src_path, 'created at', datetime.datetime.now().strftime("%H:%M:%S"))

    foundfile = False
    observer = Observer()
    event_handler = ExampleHandler()
    observer.schedule(event_handler, path=os.path.split(file_to_monitor)[0]+'/', recursive=True)
    observer.start()

    # sleep until keyboard interrupt, then stop + rejoin the observer
    try:
        while observer.isAlive():
            time.sleep(1)
        foundfile = True
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

    return foundfile

if __name__ == '__main__':
    file_to_monitor = '/Users/acharyya/Desktop/newdir/targetfile.txt'
    foundfile2 = monitor_for_file_watchdog(file_to_monitor)
    if foundfile2: print('Found', file_to_monitor, 'at', datetime.datetime.now().strftime("%H:%M:%S"))