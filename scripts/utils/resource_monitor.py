from __future__ import annotations
import threading, time, random
from collections import defaultdict
from typing import TypedDict, Optional
from .system_resource import CPU, GPU

class ResourceUsageT(TypedDict):
    cpu_usage: dict[int, float]             # cpu_id: usage in 0-1
    memory_total: int                       # total memory in bytes
    memory_percent: float                   # memory usage in percent in 0-100
    gpu_usage: dict[int, float]             # gpu_id: util in 0-1
    gpu_memory_total: dict[int, int]        # gpu_id: total_memory in bytes
    gpu_memory_percent: dict[int, float]    # gpu_id: memoryUtil in 0-100

class ResourceMonitor:
    """
    Check CPU, GPU utilization periodically
    """
    ETA = 0.75
    INTERVAL = 0.5
    def __init__(self, aval_gpu_ids: list[int]):
        self._aval_gpu_ids = aval_gpu_ids
        self.__cpu = CPU()
        self.__gpu = GPU()

        # usage refers to power usage
        self._gpu_usage: dict[int, Optional[float]] = defaultdict(lambda: None)
        self._cpu_percent: list[float] | None = None
        self.__monitor_thread = None

        self.update()
    
    @property
    def cpu(self):
        return self.__cpu
    
    @property
    def gpu(self):
        return self.__gpu
    
    @property
    def gpu_usage(self):
        return self._gpu_usage
    
    @property
    def aval_gpu_ids(self):
        return self._aval_gpu_ids
    
    def update(self):
        for gpu_device in self.gpu.all_devices():
            if not gpu_device.idx in self._aval_gpu_ids:
                continue
            gpu_load = gpu_device.util              # 0-1, or nan

            # check if is nan
            if gpu_load == float('nan'):
                gpu_load = -1.
                
            if self._gpu_usage[gpu_device.idx] is None:
                self._gpu_usage[gpu_device.idx] = gpu_load
            else:
                __prev = self._gpu_usage[gpu_device.idx]
                assert __prev is not None
                self._gpu_usage[gpu_device.idx] = self.ETA * __prev + (1-self.ETA) * gpu_load
        
        if self._cpu_percent is None:
            self._cpu_percent = self.__cpu.cpu_percent()
        else:
            __prev = self._cpu_percent
            assert __prev is not None
            self._cpu_percent = [
                self.ETA * __prev[i] + (1-self.ETA) * x 
                for i, x in enumerate(self.__cpu.cpu_percent())
                ]
    
    def _monitor(self):
        while True:
            self.update()
            time.sleep(self.INTERVAL)
    
    def get_gpu_usage(self, gpu_id: int) -> float:
        """Get GPU power usage in percentage, 0-1, return None if the GPU is not available"""
        assert gpu_id in self._aval_gpu_ids, f'GPU {gpu_id} is not available, available GPUs: {self._aval_gpu_ids}'
        ret = self._gpu_usage[gpu_id]
        assert ret is not None, f'GPU {gpu_id} usage is None, please update the monitor first'
        return ret
    
    def get_cpu_usage(self) -> list[float]:
        """ Get CPU usage in percentage, 0-1, return a list of float for each core """
        assert self._cpu_percent is not None, 'Please update the monitor first'
        return [x/100 for x in self._cpu_percent]
    
    def start_monitor_daemon(self):
        self.__monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.__monitor_thread.start()

    def resource_usage(self) -> ResourceUsageT:
        gpu_mem_percent = {}
        gpu_usage = {}
        gpu_memory_total = {}
        for g in self.gpu.all_devices():
            if g.idx in self.aval_gpu_ids:
                gpu_mem_percent[g.idx] = g.memory_util() * 100
                gpu_usage[g.idx] = self.get_gpu_usage(g.idx)
                gpu_memory_total[g.idx] = g.memory_total * 1024     # convert to bytes
        memory = self.cpu.memory()

        return {
            'cpu_usage': {i: u for i, u in enumerate(self.get_cpu_usage())},
            'memory_total': memory.total,
            'memory_percent': memory.percent,
            'gpu_usage': gpu_usage, 
            'gpu_memory_total': gpu_memory_total, 
            'gpu_memory_percent': gpu_mem_percent,
        }