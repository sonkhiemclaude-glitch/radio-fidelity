"""Read-only inventory for preparing the reviewer-revision experiments.

Run on a workstation or EC2: python3 code/preflight_revision.py
No third-party packages are required; no data are changed.
"""
from pathlib import Path
import importlib.util
import json
import os
import platform
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "disk_free_gib": round(disk.free / 1024**3, 1),
        "packages": {name: importlib.util.find_spec(name) is not None
                     for name in ("numpy", "scipy", "pandas", "matplotlib", "torch")},
        "processed_data": {p.name: p.stat().st_size for p in sorted((root / "data").glob("*.npz"))},
        "external_dependencies": {
            name: (root.parent / relative).exists()
            for name, relative in {
                "statsx": "common/pyutils/statsx.py",
                "plotstyle": "common/pyutils/plotstyle.py",
                "latex_preamble": "common/tex/elsarticle-preamble.tex",
                "latex_authors": "common/tex/authors.tex",
                "raw_data_directory": "common/data/rf",
            }.items()
        },
    }
    if platform.system() == "Linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            report["memory"] = [line for line in meminfo.read_text().splitlines()
                                if line.startswith(("MemTotal:", "MemAvailable:"))]
    elif platform.system() == "Darwin":
        value = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if value.returncode == 0:
            report["ram_gib"] = round(int(value.stdout) / 1024**3, 1)
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        report["nvidia_gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
