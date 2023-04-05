#!/usr/bin/env python
import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DockerBuildHandler:
    build_process: subprocess.Popen = None
    service_process: subprocess.Popen = None

    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.last_event_time = time.time() - 0.1
        self.pending_build = True

    def handle_event(self, event):
        if event.src_path.endswith('~'):
            return
        now = time.time()
        if now - self.last_event_time >= 0.5:
            self.last_event_time = now
            self.pending_build = True


    def cancel_build(self):
        if self.build_process is not None and self.build_process.poll() is None:
            self.build_process.terminate()
            self.build_process.wait()

    def cancel_service(self):
        print("Stopping Docker service...")

    def iterate(self):
        if self.pending_build:
            self.pending_build = False
            self.cancel_build()
            self.start_build()
        else:
            r = self.build_process.poll()
            if self.pending_service_restart and r is not None:
                self.pending_service_restart = False
                if r == 0:
                    self.cancel_service()
                    self.start_service()
                else:
                    print("\n\nDocker build failed, not restarting service.\n\n")

    def start_build(self):
        print("Starting Docker build...")
        self.build_process = subprocess.Popen("docker build -t thesis . --progress=plain && docker run --rm --entrypoint cat thesis /app/title-page-ai.pdf > ../title-page-ai.pdf && docker run --rm --entrypoint cat thesis /app/neurips_2022.pdf > ../neurips_2022.pdf",
                                              shell=True)
        # ['docker', 'build', '-t', 'thesis', self.path])
        self.pending_service_restart = True

    def start_service(self):
        print("Starting Docker service...")


class MyHandler(FileSystemEventHandler):
    def __init__(self, docker_handler):
        self.docker_handler = docker_handler

    def on_created(self, event):
        self.docker_handler.handle_event(event)

    def on_modified(self, event):
        self.docker_handler.handle_event(event)

    def on_deleted(self, event):
        self.docker_handler.handle_event(event)


if __name__ == "__main__":
    path = '.'
    config = sys.argv[1] if len(sys.argv) > 1 else 'devel'
    docker_handler = DockerBuildHandler(path, config)
    event_handler = MyHandler(docker_handler)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(0.1)
            docker_handler.iterate()
    finally:
        observer.stop()
        observer.join()
