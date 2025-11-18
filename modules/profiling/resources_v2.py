import psutil
import time
import threading
import os
from modules.logging.logger import log_print

# Try to import pynvml, handle failure gracefully (e.g. on non-GPU machines)
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        
        # Initialize Stats
        self.stats = {
            "cpu_peak": 0.0,
            "cpu_avg": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "vram_peak_mb": 0.0,  # <--- NEW
            "vram_avg_mb": 0.0,   # <--- NEW
            "duration_sec": 0.0
        }
        self._thread = None
        self.start_time = 0.0
        
        # Setup GPU handle
        self.gpu_handle = None
        if pynvml_available:
            try:
                pynvml.nvmlInit()
                # We monitor GPU 0. If you have multiple, you might want to loop/sum them.
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                # Drivers missing or GPU not found
                self.gpu_handle = None

    def _monitor(self):
        process = psutil.Process(os.getpid())
        
        # Lists to store samples
        cpu_samples = []
        ram_samples = []
        vram_samples = [] # <--- NEW
        
        while not self.stop_event.is_set():
            # 1. CPU & RAM
            cpu = psutil.cpu_percent(interval=None)
            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)
            
            # 2. VRAM (If GPU available)
            vram_mb = 0.0
            if self.gpu_handle:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_mb = info.used / (1024 * 1024)
                except:
                    pass
            
            # 3. Store Samples
            cpu_samples.append(cpu)
            ram_samples.append(ram_mb)
            vram_samples.append(vram_mb)
            
            # 4. Update Peaks
            if cpu > self.stats["cpu_peak"]: 
                self.stats["cpu_peak"] = round(cpu, 2)
            if ram_mb > self.stats["ram_peak_mb"]: 
                self.stats["ram_peak_mb"] = round(ram_mb, 2)
            if vram_mb > self.stats["vram_peak_mb"]: 
                self.stats["vram_peak_mb"] = round(vram_mb, 2)
            
            time.sleep(self.interval)
            
        # Calculate Averages
        if cpu_samples:
            self.stats["cpu_avg"] = round(sum(cpu_samples) / len(cpu_samples), 2)
            self.stats["ram_avg_mb"] = round(sum(ram_samples) / len(ram_samples), 2)
            self.stats["vram_avg_mb"] = round(sum(vram_samples) / len(vram_samples), 2)

    # --- FLAT USAGE METHODS ---
    def start(self):
        self.start_time = time.time()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if not self._thread: return self.stats
        
        self.stop_event.set()
        self._thread.join()
        self.stats["duration_sec"] = round(time.time() - self.start_time, 2)
        
        # Cleanup NVML
        if pynvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
                
        return self.stats

    # --- CONTEXT MANAGER SUPPORT ---
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()