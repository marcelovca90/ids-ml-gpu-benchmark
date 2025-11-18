import psutil
import time
import threading
import os
from modules.logging.logger import log_print

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # Safety for resetting variables
        
        # Global History (Stores finished stages)
        self.history = {} 
        
        # Current Stage Accumulators
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
        """Resets the counters for a new measurement phase."""
        self.start_time = time.time()
        self.cpu_peak = 0.0
        self.ram_peak_mb = 0.0
        self.vram_peak_mb = 0.0
        
        # Samples for averaging
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []

    def _monitor(self):
        process = psutil.Process(os.getpid())
        
        while not self.stop_event.is_set():
            # 1. Capture Metrics
            cpu = psutil.cpu_percent(interval=None)
            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)
            
            vram_mb = 0.0
            if self.gpu_handle:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_mb = info.used / (1024 * 1024)
                except:
                    pass

            # 2. Update Accumulators (Thread-Safe)
            with self.lock:
                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram_mb)
                self.vram_samples.append(vram_mb)
                
                if cpu > self.cpu_peak: self.cpu_peak = round(cpu, 2)
                if ram_mb > self.ram_peak_mb: self.ram_peak_mb = round(ram_mb, 2)
                if vram_mb > self.vram_peak_mb: self.vram_peak_mb = round(vram_mb, 2)
            
            time.sleep(self.interval)

    def _snapshot(self):
        """Calculates averages and duration for the CURRENT phase."""
        duration = round(time.time() - self.start_time, 2)
        
        # Avoid division by zero if no samples collected yet
        if self.cpu_samples:
            cpu_avg = round(sum(self.cpu_samples) / len(self.cpu_samples), 2)
            ram_avg = round(sum(self.ram_samples) / len(self.ram_samples), 2)
            vram_avg = round(sum(self.vram_samples) / len(self.vram_samples), 2)
        else:
            cpu_avg, ram_avg, vram_avg = 0.0, 0.0, 0.0

        return {
            "duration_sec": duration,
            "cpu_peak": self.cpu_peak,
            "cpu_avg": cpu_avg,
            "ram_peak_mb": self.ram_peak_mb,
            "ram_avg_mb": ram_avg,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_avg_mb": vram_avg
        }

    # --- PUBLIC API ---

    def start(self):
        self._reset_accumulators()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def checkpoint(self, stage_name: str):
        """
        Saves current stats under 'stage_name', resets counters, and continues monitoring.
        """
        with self.lock:
            # 1. Save current state
            stats = self._snapshot()
            self.history[stage_name] = stats
            log_print(f"[Monitor] Checkpoint '{stage_name}' recorded ({stats['duration_sec']}s).")
            
            # 2. Reset for next stage
            self._reset_accumulators()

    def stop(self):
        """
        Stops monitoring, saves the final/remaining period, and returns the full history.
        """
        if not self._thread: return self.history
        
        self.stop_event.set()
        self._thread.join()
        
        # Save the final chunk (whatever happened since the last checkpoint)
        with self.lock:
            final_stats = self._snapshot()
            # Only add if meaningful time passed
            if final_stats['duration_sec'] > 0.1:
                self.history["teardown"] = final_stats

        # Cleanup NVML
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