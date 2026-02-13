# ServiceManager.py
import CONSTANTS
from DataRoutingEngine import DataRoutingEngine
from SimpleUI import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSignal, QObject, QThread
import time
import sys

class ServiceWorker(QObject):
    # Update the signal to send three items: status, image identifier, and detection data.
    updateImageSignal = pyqtSignal(str, object, object)

    def __init__(self, imgDirectory):
        super().__init__()
        self.running = False
        self.paused = False
        self.DataEngine = DataRoutingEngine(imgDirectory)
            
    def run(self):
        try:
            self.running = True
            sentCount = 0
            self.DataEngine.reset()

            while self.running:
                if not self.paused:
                    sentCount += 1
                    if sentCount % 20 == 0:
                        self.DataEngine.logEntry("Service running...")

                    result = self.DataEngine.sendNextToClassifier()
                    
                    # Check for None values.
                    if result is None or result[0] is None:
                        print("⚠️ Warning: No image classified, skipping iteration...")
                        time.sleep(1)
                        continue 

                    # Unpack the result.
                    detectionData, annotated_filename = result
                    
                    # Emit the status text, annotated filename, and detection data.
                    self.updateImageSignal.emit("Time: " + str(sentCount), annotated_filename, detectionData)
                
                time.sleep(1)

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"❌ Error in ServiceWorker: {e}")
            self.DataEngine.logEntry(f"❌ Error in ServiceWorker: {e}")
            self.stop()

    def pause(self):
        self.paused = True
        self.DataEngine.logEntry("Service paused")

    def resume(self):
        self.paused = False
        self.DataEngine.logEntry("Service resumed")

    def stop(self):
        if self.running:
            self.running = False
            self.DataEngine.logEntry("Stopping service...")

    def restart(self):
        self.running = False
        self.DataEngine.logEntry("Restarting service...")

    def logEntry(self, msg):
        with open("service_log.txt", "a") as log_file:
            log_file.write(msg + f" at {time.ctime()}\n")


class ServiceManager:
    def __init__(self, imgDirectory):
        self.app = QApplication(sys.argv)
        self.mainWindow = MainWindow(self.restart, self.stop, self.pause, self.resume)
        self.mainWindow.show()

        self.worker = ServiceWorker(imgDirectory)
        self.workerThread = QThread()

        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(self.worker.run)
        self.worker.updateImageSignal.connect(self.mainWindow.updateLabelAndImage)

        self.workerThread.start()

    def pause(self):
        self.worker.pause()

    def resume(self):
        self.worker.resume()

    def restart(self):
        self.worker.stop()
        self.workerThread.quit()
        self.workerThread.wait()
        self.workerThread.start()

    def stop(self):
        self.worker.stop()
        self.workerThread.quit()
        self.workerThread.wait()
        self.app.quit()
        sys.exit(0)


def main():
    # The 'images' folder is used to load the list of files.
    service = ServiceManager('images')
    sys.exit(service.app.exec())


if __name__ == "__main__":
    main()
