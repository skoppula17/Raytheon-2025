# DataRoutingEngine.py
import CONSTANTS
from runModelOnImage import modelAPI
from collections import deque
import time
import cv2
import os

class DataRoutingEngine:
     
    def __init__(self, inputDirectory = None):
        self.classifiedFiles = set()
        self.inputSpectrograms = deque()
        self.inputFolder = inputDirectory
        self.modelAPI = modelAPI()
        self.running = False
        self.paused = False
        
        if inputDirectory:
            try:
                self.loadStatus(inputDirectory)
            except:
                self.logEntry("ERROR: Input directory is inaccessible / nonexistent")
                self.stop()
        
    def loadStatus(self, plotsDirectory, clearExisting = True):
        if clearExisting:
            self.inputSpectrograms.clear()
            self.classifiedFiles.clear()
        self.inputSpectrograms.extend(sorted(os.listdir(plotsDirectory), key = lambda p: (len(p), p)))
        return len(self.inputSpectrograms)

    def sendNextToClassifier(self):
        while self.inputSpectrograms and self.inputSpectrograms[0] in self.classifiedFiles:
            self.logEntry("WARNING: " + self.inputSpectrograms[0] + " already classified")
            self.inputSpectrograms.popleft()

        if not self.inputSpectrograms:
            successfulReset = self.resetFileTracking()
            if not successfulReset:
                self.logEntry("ERROR: no spectrogram left to classify")
                return None, None

        nextClassification = self.inputSpectrograms.popleft()

        try:
            classifiedData = self.modelAPI.classify(self.inputFolder + "/" + nextClassification)
            if len(classifiedData) == 2 and classifiedData[0] == CONSTANTS.FAILURE:
                self.logEntry("ERROR: " + classifiedData[1])
                return None, None
        except Exception as e:
            self.logEntry(f"ERROR SENDING FILE {nextClassification} to classifier: {e}")
            return None, None

        self.classifiedFiles.add(nextClassification)
        
        # Return the detection data and the filename.
        annotated_filename = nextClassification
        return classifiedData, annotated_filename

    def resetFileTracking(self):
        filesToUnclassify = sorted(list(self.classifiedFiles), key = lambda p: (len(p), p))
        if not filesToUnclassify: return False
        self.inputSpectrograms.extend(filesToUnclassify)
        self.classifiedFiles.clear()
        return True

    def start(self):
        if not self.running:
            print("Service starting...")
            self.running = True
        else:
            print("Service is already running.")

    def stop(self):
        if self.running:
            print("Stopping service...")
            self.running = False
        else:
            print("Service is not running.")

    def logEntry(self, msg):
        with open("service_log.txt", "a") as log_file:
            log_file.write(msg + f" at {time.ctime()}\n")

    def pause(self):
        self.paused = True
        self.logEntry("Service paused")
        print("Service paused.")

    def resume(self):
        self.paused = False
        self.logEntry("Service resumed")
        print("Service resumed.")

    def reset(self):
        self.paused = True
        self.resetFileTracking()
        self.logEntry("System reset")
        print("Create a script to reset MongoDB collecting analysis metrics...")
        self.paused = False

    def status(self):
        for file in self.classifiedFiles:
            print(file + ": classified")
        for file in self.inputSpectrograms:
            print(file + ": unclassified")

    def run(self):
        while self.running:
            if not self.paused:
                self.logEntry("Service running")
                time.sleep(10)

    def commandListener(self):
        while self.running:
            command = input("Enter command (pause, resume, status, reset, stop): ").strip().lower()
            if command == "pause":
                self.pause()
            elif command == "resume":
                self.resume()
            elif command == "status":
                self.status()
            elif command == "reset":
                self.reset()
            elif command == "stop":
                self.stop()
                break
            else:
                print(f"Unknown command: {command}")

if __name__ == "__main__":
    service = DataRoutingEngine('images')
    try:
        service.start()
        while service.running:
            time.sleep(2)
            if not service.paused:
                service.sendNextToClassifier()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping service...")
        service.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
        service.stop()
