import psutil
import time
import threading
import os

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

# Try to import log_print from modules
try:
    from modules.logging.logger import log_print
except ImportError:
    def log_print(x):
        print(x)

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.history = {}

        # Get CPU Core Count once
        self.cpu_count = psutil.cpu_count(logical=True) or 1

        # Initialize Accumulators
        self._reset_accumulators()

        self._thread = None
        self.gpu_handle = None

        # Setup NVML
        if pynvml_available:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def _reset_accumulators(self):
        self.start_time = time.time()

        # Peaks
        self.cpu_raw_peak = 0.0   # Can go above 100%
        self.cpu_norm_peak = 0.0  # Max 100%
        self.ram_peak_mb = 0.0
        self.vram_peak_mb = 0.0
        self.gpu_util_peak = 0.0

        # Samples
        self.cpu_raw_samples = []
        self.cpu_norm_samples = []
        self.ram_samples = []
        self.vram_samples = []
        self.gpu_util_samples = []

    def _monitor(self):
        process = psutil.Process(os.getpid())

        # PRIME THE COUNTER: The first call always returns 0.0.
        # We call it here so the first loop iteration gets a real delta.
        process.cpu_percent(interval=None)

        while not self.stop_event.is_set():
            # 1. Metrics
            raw_cpu = process.cpu_percent(interval=None)
            norm_cpu = raw_cpu / self.cpu_count

            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)

            vram_mb = 0.0
            gpu_util = 0.0
            
            if self.gpu_handle:
                try:
                    # Memory Usage
                    mem_stats = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_mb = mem_stats.used / (1024 * 1024)
                    
                    # Compute Utilization (core load)
                    # Returns object with .gpu and .memory
                    util_stats = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_util = util_stats.gpu 
                except:
                    pass

            # 2. Update Accumulators
            with self.lock:
                self.cpu_raw_samples.append(raw_cpu)
                self.cpu_norm_samples.append(norm_cpu)
                self.ram_samples.append(ram_mb)
                self.vram_samples.append(vram_mb)
                self.gpu_util_samples.append(gpu_util)

                if raw_cpu > self.cpu_raw_peak: self.cpu_raw_peak = round(raw_cpu, 2)
                if norm_cpu > self.cpu_norm_peak: self.cpu_norm_peak = round(norm_cpu, 2)
                if ram_mb > self.ram_peak_mb: self.ram_peak_mb = round(ram_mb, 2)
                if vram_mb > self.vram_peak_mb: self.vram_peak_mb = round(vram_mb, 2)
                if gpu_util > self.gpu_util_peak: self.gpu_util_peak = round(gpu_util, 2)

            time.sleep(self.interval)

    def _snapshot(self):
        duration = round(time.time() - self.start_time, 2)

        if self.cpu_raw_samples:
            n = len(self.cpu_raw_samples)
            cpu_raw_avg = round(sum(self.cpu_raw_samples) / n, 2)
            cpu_norm_avg = round(sum(self.cpu_norm_samples) / n, 2)
            ram_avg = round(sum(self.ram_samples) / n, 2)
            vram_avg = round(sum(self.vram_samples) / n, 2)
            gpu_util_avg = round(sum(self.gpu_util_samples) / n, 2)
        else:
            cpu_raw_avg, cpu_norm_avg, ram_avg, vram_avg, gpu_util_avg = 0.0, 0.0, 0.0, 0.0, 0.0

        return {
            "duration_sec": duration,
            "cpu_raw_peak": self.cpu_raw_peak,
            "cpu_raw_avg": cpu_raw_avg,
            "cpu_norm_peak": self.cpu_norm_peak,
            "cpu_norm_avg": cpu_norm_avg,
            "ram_peak_mb": self.ram_peak_mb,
            "ram_avg_mb": ram_avg,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_avg_mb": vram_avg,
            "gpu_util_peak": self.gpu_util_peak,
            "gpu_util_avg": gpu_util_avg
        }

    # --- PUBLIC API ---

    def start(self):
        self._reset_accumulators()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def checkpoint(self, stage_name: str):
        with self.lock:
            stats = self._snapshot()
            self.history[stage_name] = stats
            log_print(f"[Monitor] Checkpoint '{stage_name}' ({stats['duration_sec']}s).")
            self._reset_accumulators()

    def stop(self):
        if not self._thread: return self.history

        self.stop_event.set()
        self._thread.join()

        with self.lock:
            final_stats = self._snapshot()
            if final_stats['duration_sec'] > 0.1:
                self.history["teardown"] = final_stats

        if pynvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

        return self.history

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()