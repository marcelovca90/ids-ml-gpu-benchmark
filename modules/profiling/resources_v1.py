import psutil
import time
import threading
import os
from modules.logging.logger import log_print

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.stats = {
            "cpu_peak": 0.0,
            "cpu_avg": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "duration_sec": 0.0
        }
        self._thread = None
        self.start_time = 0.0

    def _monitor(self):
        process = psutil.Process(os.getpid())
        cpu_samples = []
        ram_samples = []
        
        while not self.stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)
            
            cpu_samples.append(cpu)
            ram_samples.append(ram_mb)
            
            if cpu > self.stats["cpu_peak"]: self.stats["cpu_peak"] = round(cpu, 2)
            if ram_mb > self.stats["ram_peak_mb"]: self.stats["ram_peak_mb"] = round(ram_mb, 2)
            
            time.sleep(self.interval)
            
        if cpu_samples:
            self.stats["cpu_avg"] = round(sum(cpu_samples) / len(cpu_samples), 2)
            self.stats["ram_avg_mb"] = round(sum(ram_samples) / len(ram_samples), 2)

    # --- FLAT USAGE METHODS ---
    def start(self):
        """Starts monitoring (non-blocking)."""
        self.start_time = time.time()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stops monitoring and calculates final duration."""
        if not self._thread: return self.stats
        
        self.stop_event.set()
        self._thread.join()
        self.stats["duration_sec"] = round(time.time() - self.start_time, 2)
        return self.stats

    # --- CONTEXT MANAGER SUPPORT ---
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()