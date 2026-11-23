"""Verify the released bytes and reconstruct the paper's primary audit results."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CHECKS = ("verify.py", "check_estimands.py", "check_alignment.py", "check_seed_map.py", "check_tables.py")


def verify_manifest():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        file = ROOT / name
        if sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"File digest differs: {name}")


if __name__ == "__main__":
    verify_manifest()
    for check in CHECKS:
        subprocess.run([sys.executable, "-B", str(ROOT / "artifact" / check)], check=True)
