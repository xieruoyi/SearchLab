"""Download the registered dataset; validate bytes; never extract test data."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))["dataset"]
target = ROOT / "data/raw/nfcorpus.zip"
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest() != config["sha256"]:
        raise RuntimeError("Existing archive checksum differs; preserve it and investigate")
    print("Registered archive already present and verified")
else:
    temporary = target.with_suffix(".zip.partial")
    with urllib.request.urlopen(config["url"], timeout=60) as response, temporary.open("wb") as f:
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != config["sha256"]:
        raise RuntimeError("Download checksum mismatch; archive not promoted")
    os.replace(temporary, target)
    print("Downloaded and verified", target)
