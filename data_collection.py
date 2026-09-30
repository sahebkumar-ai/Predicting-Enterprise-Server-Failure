import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
from ping3 import ping

CSV_PATH = Path(__file__).with_name("system_working_performance_dataset.csv")
COLUMNS = [
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "memory_availabale_gb",
    "disk_usage",
    "disk_read_mb",
    "disk_right_md",
    "network_set_mb",
    "network_recive_mb",
    "ping_ms",
    "battery_percent",
    "running_processes",
    "failure",
]
def ensure_dataset():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        pd.DataFrame(columns=COLUMNS).to_csv(CSV_PATH, index=False)


def collect_sample(previous_disk, previous_network):
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    system_drive = os.environ.get("SystemDrive")
    disk = psutil.disk_usage(f"{system_drive}{os.sep}" if system_drive else "/")
    current_disk = psutil.disk_io_counters()
    current_network = psutil.net_io_counters()

    disk_read = 0.0 if current_disk is None or previous_disk is None else max(0.0, (current_disk.read_bytes - previous_disk.read_bytes) / (1024 ** 2))
    disk_write = 0.0 if current_disk is None or previous_disk is None else max(0.0, (current_disk.write_bytes - previous_disk.write_bytes) / (1024 ** 2))
    network_sent = 0.0 if current_network is None or previous_network is None else max(0.0, (current_network.bytes_sent - previous_network.bytes_sent) / (1024 ** 2))
    network_received = 0.0 if current_network is None or previous_network is None else max(0.0, (current_network.bytes_recv - previous_network.bytes_recv) / (1024 ** 2))

    try:
        latency = ping("8.8.8.8", timeout=2)
        ping_ms = round(latency * 1000, 2) if latency is not None else None
    except Exception:
        ping_ms = None

    try:
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery is not None else None
    except (OSError, NotImplementedError):
        battery_percent = None
    failure = int(
        cpu_usage > 90
        or memory.percent > 95
        or disk.percent > 95
        or (ping_ms is not None and ping_ms > 300)
    )
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage": cpu_usage,
        "memory_usage": memory.percent,
        "memory_availabale_gb": round(memory.available / (1024 ** 3), 2),
        "disk_usage": disk.percent,
        "disk_read_mb": round(disk_read, 2),
        "disk_right_md": round(disk_write, 2),
        "network_set_mb": round(network_sent, 2),
        "network_recive_mb": round(network_received, 2),
        "ping_ms": ping_ms,
        "battery_percent": battery_percent,
        "running_processes": len(psutil.pids()),
        "failure": failure,
    }
    return row, current_disk, current_network

def main():
    ensure_dataset()
    previous_disk = psutil.disk_io_counters()
    previous_network = psutil.net_io_counters()
    print(f"Collecting server metrics into {CSV_PATH}. Press Ctrl+C to stop.")
    try:
        while True:
            row, previous_disk, previous_network = collect_sample(previous_disk, previous_network)
            pd.DataFrame([row], columns=COLUMNS).to_csv(CSV_PATH, mode="a", header=False, index=False)
            print(row)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Data collection stopped.")


if __name__ == "__main__":
    main()



