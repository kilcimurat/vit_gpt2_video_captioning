from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

from tqdm.auto import tqdm

from ..config import DatasetConfig


def _download_bytes(url: str, description: str) -> bytes:
    with urlopen(url) as response:  # nosec - trusted source list provided via config
        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1 << 20
        buffer = io.BytesIO()
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        ) as progress:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                progress.update(len(chunk))
        return buffer.getvalue()


def _extract_csv_from_zip(data: bytes, filename_hint: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError("No CSV file found in annotation archive")
        if filename_hint in candidates:
            target = filename_hint
        else:
            target = candidates[0]
        with archive.open(target) as handle:
            return handle.read()


def _convert_txt_annotations(data: bytes) -> bytes:
    text = data.decode("utf-8")
    output = io.StringIO()
    fieldnames = ["VideoID", "YouTubeID", "Start", "End", "Sentence"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            clip_id, sentence = line.split(" ", 1)
        except ValueError:
            # Line without a sentence – skip to match original corpus behaviour
            continue

        try:
            youtube_id, start, end = clip_id.rsplit("_", 2)
        except ValueError:
            youtube_id, start, end = clip_id, "", ""

        writer.writerow(
            {
                "VideoID": clip_id,
                "YouTubeID": youtube_id,
                "Start": start,
                "End": end,
                "Sentence": sentence.strip(),
            }
        )

    return output.getvalue().encode("utf-8")


def ensure_captions_file(cfg: DatasetConfig) -> Path:
    target = cfg.raw_dir / cfg.captions_filename
    if target.exists():
        return target

    last_error: Exception | None = None
    for url in cfg.captions_urls:
        try:
            description = f"Downloading annotations ({url.split('/')[-1]})"
            payload = _download_bytes(url, description=description)
            if url.lower().endswith(".zip"):
                payload = _extract_csv_from_zip(payload, cfg.captions_filename)
            elif url.lower().endswith(".txt"):
                payload = _convert_txt_annotations(payload)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as sink:
                sink.write(payload)
            return target
        except (URLError, FileNotFoundError, zipfile.BadZipFile) as exc:
            last_error = exc
            continue

    raise FileNotFoundError(
        "Unable to retrieve MSVD captions. Tried URLs: "
        + ", ".join(cfg.captions_urls)
        + f". Place {cfg.captions_filename} in {cfg.raw_dir} to proceed."
    ) from last_error
