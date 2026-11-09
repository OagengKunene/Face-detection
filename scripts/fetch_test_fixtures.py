"""Explicitly fetch two public sample portraits for opt-in integration tests."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

FIXTURES = (
    (
        "portrait_a.png",
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.24.0/skimage/data/astronaut.png",
        "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    ),
    (
        "portrait_b.jpg",
        "https://raw.githubusercontent.com/matplotlib/matplotlib/"
        "v3.10.7/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
        "a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130",
    ),
)


def fetch(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in FIXTURES:
        target = destination / filename
        raw = target.read_bytes() if target.is_file() else None
        if raw is None or hashlib.sha256(raw).hexdigest() != expected:
            with urllib.request.urlopen(url, timeout=45) as stream:
                raw = stream.read(2_000_000)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Fixture checksum failed: {filename}")
            target.write_bytes(raw)
        print(f"Verified {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("test-fixtures"))
    fetch(parser.parse_args().destination)
