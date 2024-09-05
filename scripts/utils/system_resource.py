import os, psutil, dataclasses
from typing import Optional
import pynvml

class CPU:

    def num_threads(self) -> int:
        assert (cpu_count := os.cpu_count()) is not None, 'CPU count is not available'
        return cpu_count
    
    def cpu_percent(self) -> list[float]: 
        """ The CPU usage of the system, in percent. """
        return psutil.cpu_percent(percpu=True)
    
    def memory(self):
        """ The memory usage of the system, in bytes. """
        @dataclasses.dataclass(frozen=True)
        class Memory:
            total: int
            available: int
            percent: float
            used: int
            free: int
        return Memory(*psutil.virtual_memory()[:5])


@dataclasses.dataclass()
class GPUDevice:
    idx: int
    uid: str
    temperature: float
    memory_used: int
    memory_total: int
    power_usage: float
    power_limit: float
    util: float

    def memory_util(self) -> float:
        return self.memory_used / self.memory_total
    
    def power_util(self) -> float:
        return self.power_usage / self.power_limit

class GPU:
    def __init__(self):
        pynvml.nvmlInit()
    
    def __del__(self):
        pynvml.nvmlShutdown()
    
    def device_count(self) -> int:
        return pynvml.nvmlDeviceGetCount()
    
    def all_devices(self) -> list[GPUDevice]:
        return [gpu_device for i in range(self.device_count()) if (gpu_device := self.query_device(i)) is not None]
    
    def query_device(self, idx: int) -> Optional[GPUDevice]:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return GPUDevice(
                idx=idx,
                uid=pynvml.nvmlDeviceGetUUID(handle),
                memory_used=info.used,      # type: ignore
                memory_total=info.total,    # type: ignore
                power_usage=power / 1000,
                power_limit=limit / 1000,
                temperature=temperature, 
                util=util.gpu               # type: ignore
            )
        except pynvml.NVMLError:
            return None
        







