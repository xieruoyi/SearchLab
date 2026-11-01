"""Download only pinned PyTorch model assets; record their SHA256 hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    for entry in (cfg["retrieval"]["dense"], cfg["reranker"]):
        repo, revision = entry["model"], entry["revision"]
        target = ROOT / ".cache/models" / repo.replace("/", "--") / revision
        target.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}", timeout=60) as response:
            info = json.load(response)
        assert info["sha"] == revision
        names = [x["rfilename"] for x in info["siblings"]]
        assert "model.safetensors" in names
        selected = [n for n in names if ("/" not in n or n == "1_Pooling/config.json") and (n.endswith(".json") or n in ("vocab.txt", "model.safetensors", "README.md"))]
        hashes = {}
        for name in selected:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                print(f"Downloading {repo}@{revision[:8]}/{name}", flush=True)
                temp = dest.with_suffix(dest.suffix + ".partial")
                with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120) as response, temp.open("wb") as output:
                    shutil.copyfileobj(response, output)
                temp.replace(dest)
            hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        (target / "download_manifest.json").write_text(json.dumps({"repo": repo, "revision": revision, "sha256": hashes}, indent=2) + "\n")
        print(f"Verified pinned assets: {repo}", flush=True)


if __name__ == "__main__":
    main()
