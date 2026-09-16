"""Operational sampling only; metric formulas live in fleetrl.metrics."""

import os
import time
from threading import Event, Thread

import psutil


class ResourceSampler:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self.stop_event = Event()
        self.thread = None

    def sample(self):
        try:
            process = psutil.Process(os.getpid())
            processes = [process] + process.children(recursive=True)
            rss = 0
            cpu = 0.0
            for item in processes:
                try:
                    rss += item.memory_info().rss
                    t = item.cpu_times()
                    cpu += t.user + t.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples.append(
                {
                    "time": time.perf_counter(),
                    "tree_rss_bytes": rss,
                    "tree_cpu_s": cpu,
                    "processes": len(processes),
                }
            )
        except psutil.Error:
            pass

    def start(self):
        self.sample()

        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.sample()
        return self.samples
