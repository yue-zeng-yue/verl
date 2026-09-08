"""Sample only the two GPU UUIDs selected by the CUDA preflight."""

import csv
import threading
import time

import pynvml


class GPUMonitor:
    def __init__(self, path, uuids):
        self.path, self.uuids = path, uuids
        self.stop = threading.Event()
        self.error = None

    def __enter__(self):
        pynvml.nvmlInit()
        try:
            self.handles = [
                pynvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in self.uuids
            ]
        except Exception:
            pynvml.nvmlShutdown()
            raise
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        try:
            with self.path.open("w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "time_unix",
                        "gpu_uuid",
                        "memory_used_bytes",
                        "memory_total_bytes",
                        "gpu_utilization_percent",
                    ]
                )
                while not self.stop.is_set():
                    for uuid, handle in zip(self.uuids, self.handles, strict=True):
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        writer.writerow(
                            [time.time(), uuid, memory.used, memory.total, util.gpu]
                        )
                    file.flush()
                    self.stop.wait(0.1)
        except Exception as error:  # noqa: BLE001 - propagate thread failures to the caller.
            self.error = error

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        pynvml.nvmlShutdown()
        if self.error is not None and exc is None:
            raise self.error
