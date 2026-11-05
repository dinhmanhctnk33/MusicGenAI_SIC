"""FMA data pipeline khong phu thuoc model pretrained."""
from __future__ import annotations

import os
import random
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from config import (CODEC_CHECKPOINT_DIR, CODEC_CONFIG, DATA_CONFIG,
                    MODEL_CONFIG, TRAIN_CONFIG)
from audio_codec import load_codec_checkpoint
from scratch_components import ByteTokenizer


GENRE_VI = {
    "Blues": "Blues",
    "Classical": "cổ điển",
    "Country": "đồng quê",
    "Easy Listening": "nhẹ nhàng dễ nghe",
    "Electronic": "điện tử",
    "Experimental": "thử nghiệm",
    "Folk": "dân gian",
    "Hip-Hop": "Hip-Hop",
    "Instrumental": "hòa tấu",
    "International": "quốc tế",
    "Jazz": "Jazz",
    "Old-Time / Historic": "cổ xưa và lịch sử",
    "Pop": "Pop",
    "Rock": "Rock",
    "Soul-RnB": "Soul và R&B",
    "Spoken": "đọc nói",
}


class InvalidTrackError(RuntimeError):
    pass


def download_and_extract_fma(dataset_dir: str, subset: str = DATA_CONFIG["subset"]):
    if subset != "small":
        raise ValueError("Trinh tai tu dong hien chi ho tro FMA Small.")
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    urls = {
        "fma_metadata": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "fma_small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    }
    for name, url in urls.items():
        target = root / name
        if target.exists():
            continue
        archive = root / f"{name}.zip"
        print(f"Dang tai {name}...")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(root)
        archive.unlink()


class FMAAudioTextDataset(Dataset):
    def __init__(self, fma_metadata_dir: str, fma_audio_dir: str,
                 sample_rate=DATA_CONFIG["sample_rate"],
                 audio_duration=DATA_CONFIG["audio_duration"],
                 subset=DATA_CONFIG["subset"], device="cpu", cache_dir=None,
                 use_cache=True):
        self.fma_audio_dir = Path(fma_audio_dir)
        self.dataset_root = self.fma_audio_dir.parent
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.target_samples = round(sample_rate * audio_duration)
        self.segments_per_track = DATA_CONFIG["segments_per_track"]
        self.expected_audio_steps = round(audio_duration * DATA_CONFIG["audio_token_rate"])
        self.use_cache = use_cache
        self.device = torch.device(device)
        self.tokenizer = ByteTokenizer()
        self.codec = None
        self.codec_checkpoint = CODEC_CHECKPOINT_DIR / "best_codec.pt"
        duration_ms = round(audio_duration * 1000)
        self.cache_dir = Path(cache_dir or self.dataset_root / f"encoded_cache_{DATA_CONFIG['cache_version']}" / f"sr{sample_rate}_duration{duration_ms}ms")
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        path = Path(fma_metadata_dir) / "tracks.csv"
        tracks = pd.read_csv(path, index_col=0, header=[0, 1])
        order = ["small", "medium", "large", "full"]
        tracks = tracks[tracks[("set", "subset")].isin(order[:order.index(subset) + 1])]
        meta = pd.DataFrame({
            "track_id": tracks.index, "title": tracks[("track", "title")],
            "genre": tracks[("track", "genre_top")], "artist": tracks[("artist", "name")],
        }).reset_index(drop=True)
        meta["genre"] = meta["genre"].fillna("Unknown").astype(str)
        meta["title"] = meta["title"].fillna("Untitled").astype(str)
        meta["artist"] = meta["artist"].fillna("Unknown artist").astype(str)
        meta = meta[meta["track_id"].map(lambda value: self._audio_path(int(value)).is_file())]
        self.meta = meta.reset_index(drop=True)
        self.track_ids = self.meta["track_id"].astype(int).tolist()
        self.samples = [(i, s) for i in range(len(self.meta)) for s in range(self.segments_per_track)]
        if not self.samples:
            raise RuntimeError("Khong tim thay track FMA hop le.")
        print(f"Da nap {len(self.track_ids)} track, {len(self.samples)} sample.")

    def _audio_path(self, track_id: int) -> Path:
        value = f"{track_id:06d}"
        return self.fma_audio_dir / value[:3] / f"{value}.mp3"

    def _cache_path(self, track_id, segment):
        return self.cache_dir / f"{track_id:06d}_segment{segment:02d}.pt"

    def _signature(self):
        return {"version": DATA_CONFIG["cache_version"], "sample_rate": self.sample_rate,
                "duration": self.audio_duration, "codebooks": MODEL_CONFIG["num_codebooks"],
                "codebook_size": MODEL_CONFIG["codebook_size"],
                "codec": "neural_codec_v1"}

    def _ensure_codec(self):
        if self.codec is not None:
            return
        if not self.codec_checkpoint.is_file():
            raise FileNotFoundError(
                f"Chua co neural codec: {self.codec_checkpoint}. "
                "Hay chay 'uv run train_codec.py' truoc khi train MusicGen."
            )
        print(f"Nap neural codec tu: {self.codec_checkpoint}")
        self.codec = load_codec_checkpoint(
            self.codec_checkpoint, self.device, CODEC_CONFIG
        )

    def _prompt(self, row, segment):
        genre_en = str(row.genre)[:200]
        genre_vi = GENRE_VI.get(genre_en, genre_en)
        values = {
            "genre_en": genre_en,
            "genre_vi": genre_vi,
            "artist": str(row.artist)[:200],
            "title": str(row.title)[:200],
        }
        # Moi track co 3 prompt Anh + 3 prompt Viet. Hai ngon ngu cung mo ta
        # mot track, giup encoder tu hoc lien ket ngu nghia ma khong can pretrained.
        templates = (
            "{genre_en} music.",
            "Nhạc {genre_vi}.",
            "A {genre_en} track by {artist}.",
            "Một bản nhạc {genre_vi} của {artist}.",
            "A {genre_en} piece titled {title}.",
            "Một tác phẩm {genre_vi} có tên {title}.",
        )
        return templates[segment % len(templates)].format(**values)

    def _read_audio(self, path: Path):
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise InvalidTrackError(str(exc)) from exc
        waveform = torch.from_numpy(audio.T).mean(0, keepdim=True)
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise InvalidTrackError("Audio rong hoac khong huu han.")
        waveform -= waveform.mean()
        if waveform.square().mean().sqrt() < DATA_CONFIG["min_audio_rms"]:
            raise InvalidTrackError("Audio qua im lang.")
        peak = waveform.abs().max()
        if peak > 1:
            waveform /= peak
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    def _crop_waveform(self, waveform, segment: int, random_crop: bool):
        if waveform.size(1) >= self.target_samples:
            maximum = waveform.size(1) - self.target_samples
            start = random.randint(0, maximum) if random_crop else round(maximum * segment / max(1, self.segments_per_track - 1))
            waveform = waveform[:, start:start + self.target_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_samples - waveform.size(1)))
        return waveform

    def _load_waveform(self, path: Path, segment: int, random_crop: bool):
        return self._crop_waveform(self._read_audio(path), segment, random_crop)

    def _item_from_waveform(self, row, segment, waveform):
        self._ensure_codec()
        track_id = int(row.track_id)
        prompt = self._prompt(row, segment)
        audio_tokens = self.codec.encode(
            waveform.unsqueeze(0).to(self.device)
        )[0].cpu()
        text_ids, mask = self.tokenizer.encode(prompt, DATA_CONFIG["text_max_length"])
        return {"audio_tokens": audio_tokens, "text_token_ids": text_ids,
                "text_attention_mask": mask, "prompt": prompt, "track_id": track_id,
                "segment_index": segment, "genre": row.genre}

    def _build_item(self, index, random_crop=False):
        meta_index, segment = self.samples[index]
        row = self.meta.iloc[meta_index]
        track_id = int(row.track_id)
        waveform = self._load_waveform(self._audio_path(track_id), segment, random_crop)
        return self._item_from_waveform(row, segment, waveform)

    def _load_cache(self, index):
        meta_index, segment = self.samples[index]
        track_id = self.track_ids[meta_index]
        path = self._cache_path(track_id, segment)
        if not self.use_cache or not path.is_file():
            return None
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
            return value["item"] if value.get("signature") == self._signature() else None
        except Exception:
            return None

    def populate_cache(self):
        if not self.use_cache:
            return
        completed = 0
        invalid_tracks = []
        for meta_index, row in self.meta.iterrows():
            track_id = int(row.track_id)
            missing_segments = []
            for segment in range(self.segments_per_track):
                index = meta_index * self.segments_per_track + segment
                if self._load_cache(index) is None:
                    missing_segments.append(segment)
            if not missing_segments:
                completed += self.segments_per_track
                continue
            # Doc/decode/resample moi MP3 dung mot lan. Track hong, rong, NaN/Inf
            # hoac qua im lang bi loai hoan toan thay vi lam dung toan bo pipeline.
            try:
                source = self._read_audio(self._audio_path(track_id))
            except (InvalidTrackError, OSError, RuntimeError, ValueError) as exc:
                invalid_tracks.append({
                    "track_id": track_id,
                    "audio_path": str(self._audio_path(track_id)),
                    "error": str(exc),
                })
                completed += self.segments_per_track
                print(f"[LOAI TRACK] {track_id}: {exc}")
                continue
            for segment in missing_segments:
                waveform = self._crop_waveform(source, segment, False)
                item = self._item_from_waveform(row, segment, waveform)
                compact = dict(item)
                compact["audio_tokens"] = compact["audio_tokens"].to(torch.int16)
                compact["text_token_ids"] = compact["text_token_ids"].to(torch.int16)
                compact["text_attention_mask"] = compact["text_attention_mask"].to(torch.uint8)
                torch.save({"signature": self._signature(), "item": compact},
                           self._cache_path(item["track_id"], segment))
            completed += self.segments_per_track
            if completed % 600 == 0 or completed == len(self):
                print(f"Cache: {completed}/{len(self)}")

        if invalid_tracks:
            invalid_ids = {item["track_id"] for item in invalid_tracks}
            self.meta = self.meta[
                ~self.meta["track_id"].astype(int).isin(invalid_ids)
            ].reset_index(drop=True)
            self.track_ids = self.meta["track_id"].astype(int).tolist()
            self.samples = [
                (meta_index, segment)
                for meta_index in range(len(self.meta))
                for segment in range(self.segments_per_track)
            ]
            report_path = self.dataset_root / "invalid_tracks.csv"
            pd.DataFrame(invalid_tracks).to_csv(
                report_path, index=False, encoding="utf-8-sig"
            )
            print(
                f"Da loai hoan toan {len(invalid_ids)} track khong hop le. "
                f"Con {len(self.track_ids)} track / {len(self.samples)} sample."
            )
            print(f"Bao cao track loi: {report_path}")
            if not self.samples:
                raise RuntimeError("Khong con track hop le sau khi lam sach du lieu.")

    def release_encoders(self):
        self.codec = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_item(self, index, random_crop=True):
        cached = self._load_cache(index) if not random_crop else None
        item = cached or self._build_item(index, random_crop)
        item["audio_tokens"] = item["audio_tokens"].long()
        item["text_token_ids"] = item["text_token_ids"].long()
        item["text_attention_mask"] = item["text_attention_mask"].long()
        return item

    def __getitem__(self, index):
        return self.get_item(index, False)

    def __len__(self):
        return len(self.samples)

    def sample_track_id(self, index):
        return self.track_ids[self.samples[index][0]]

    def sample_genre(self, index):
        return str(self.meta.iloc[self.samples[index][0]].genre)


def create_dataloader(batch_size=TRAIN_CONFIG["batch_size"],
                      audio_duration=DATA_CONFIG["audio_duration"], shuffle=True,
                      auto_download=True, num_workers=0, device="cpu", use_cache=True,
                      dataset_dir=DATA_CONFIG["dataset_dir"]):
    if auto_download:
        download_and_extract_fma(dataset_dir, DATA_CONFIG["subset"])
    dataset = FMAAudioTextDataset(
        str(Path(dataset_dir) / "fma_metadata"),
        str(Path(dataset_dir) / f"fma_{DATA_CONFIG['subset']}"),
        audio_duration=audio_duration, subset=DATA_CONFIG["subset"],
        device=device, use_cache=use_cache,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
