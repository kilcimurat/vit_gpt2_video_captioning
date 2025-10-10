from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from ..config import DatasetConfig


def _download(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1 << 20  # 1 MiB
        with open(target_path, "wb") as sink, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading MSVD",
        ) as progress:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                sink.write(chunk)
                progress.update(len(chunk))


def _safe_extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    destination = destination.resolve()
    member_path = (destination / member.name).resolve()
    if not str(member_path).startswith(str(destination)):
        raise RuntimeError(
            f"Blocked path traversal attempt in tar file for member: {member.name}"
        )
    archive.extract(member, destination)


def download_dataset(cfg: DatasetConfig) -> Path:
    """Download the MSVD dataset archive if it is not present."""

    archive_path = cfg.raw_dir / "YouTubeClips.tar"
    marker = cfg.raw_dir / "YouTubeClips"

    if marker.exists():
        return marker

    if not archive_path.exists():
        _download(cfg.download_url, archive_path)

    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting MSVD", unit="file"):
            _safe_extract_member(tar, member, cfg.raw_dir)

    return marker
