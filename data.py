import os
import re
import sys
import time
import zipfile
import urllib.request
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import T5Tokenizer, T5EncoderModel
from transformers import EncodecModel, AutoProcessor

from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG


MIN_AUDIO_RMS = DATA_CONFIG["min_audio_rms"]
MAX_PROMPT_FIELD_LENGTH = DATA_CONFIG["max_prompt_field_length"]


class InvalidTrackError(RuntimeError):
    """Lỗi dữ liệu của một track; có thể loại track và tiếp tục."""


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def download_and_extract_fma(
        dataset_dir: str, subset: str = DATA_CONFIG["subset"]
):
    """
    Tự động kiểm tra, tải và giải nén FMA dataset (metadata + audio) nếu chưa tồn tại.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    supported_subsets = {"small"}
    if subset not in supported_subsets:
        raise ValueError(
            f"Subset '{subset}' chưa được hỗ trợ. Các subset hợp lệ: {sorted(supported_subsets)}"
        )

    urls = {
        "fma_metadata": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        f"fma_{subset}": f"https://os.unil.cloud.switch.ch/fma/fma_{subset}.zip",
    }

    for name, url in urls.items():
        target_folder = os.path.join(dataset_dir, name)
        temp_zip_path = os.path.join(dataset_dir, f"{name}_temp.zip")

        # Kiểm tra nếu thư mục dữ liệu chưa tồn tại
        if not os.path.exists(target_folder):
            print(f"-> [TỰ ĐỘNG TẢI] Đang tải {name} từ server FMA...")

            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\r   Tiến độ tải {name}: {percent}%", end="")

            try:
                urllib.request.urlretrieve(url, temp_zip_path, reporthook=_progress)
                print(f"\n-> [GIẢI NÉN] Đang giải nén {name}_temp.zip vào thư mục dataset...")

                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)

                os.remove(temp_zip_path)  # Xóa file temp zip sau khi giải nén
                print(f"-> Hoàn tất chuẩn bị dữ liệu: {name}!\n")
            except Exception as e:
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
                raise RuntimeError(f"Lỗi trong quá trình tải/giải nén {name}: {e}")
        else:
            print(f"-> Thư mục '{name}' đã sẵn sàng tại: {target_folder}")


class FMAAudioTextDataset(Dataset):
    def __init__(
            self,
            fma_metadata_dir: str,
            fma_audio_dir: str,
            sample_rate: int = DATA_CONFIG["sample_rate"],
            audio_duration: float = DATA_CONFIG["audio_duration"],
            subset: str = DATA_CONFIG["subset"],
            t5_model_name: str = DATA_CONFIG["t5_model_name"],
            encodec_model_name: str = DATA_CONFIG["encodec_model_name"],
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            cache_dir: str = None,
            use_cache: bool = True,
    ):
        super().__init__()
        self.fma_audio_dir = fma_audio_dir
        self.dataset_root = os.path.dirname(fma_audio_dir)
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.segments_per_track = DATA_CONFIG["segments_per_track"]
        self.target_samples = int(sample_rate * audio_duration)
        self.expected_audio_steps = round(
            audio_duration * DATA_CONFIG["audio_token_rate"]
        )
        self.device = device
        self.t5_model_name = t5_model_name
        self.encodec_model_name = encodec_model_name
        self.use_cache = use_cache
        if cache_dir is None:
            duration_ms = round(audio_duration * 1000)
            cache_dir = os.path.join(
                self.dataset_root,
                f"encoded_cache_{DATA_CONFIG['cache_version']}",
                f"sr{sample_rate}_duration{duration_ms}ms",
            )
        self.cache_dir = cache_dir
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Đọc Metadata FMA (tracks.csv)
        tracks_path = os.path.join(fma_metadata_dir, "tracks.csv")
        if not os.path.exists(tracks_path):
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {tracks_path}")

        print(f"Đang đọc dữ liệu metadata từ: {tracks_path}")
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

        # Chỉ giữ các track thuộc đúng subset audio đã tải xuống.
        subset_order = ["small", "medium", "large", "full"]
        if subset not in subset_order:
            raise ValueError(
                f"Subset '{subset}' không hợp lệ. Chọn một trong: {subset_order}"
            )

        max_subset_index = subset_order.index(subset)
        allowed_subsets = subset_order[:max_subset_index + 1]
        tracks = tracks[tracks[("set", "subset")].isin(allowed_subsets)]

        # Làm sạch metadata trước khi chia train/test. Việc loại file lỗi
        # ngay tại đây tránh __getitem__ tự ý trả về một track khác, vốn có
        # thể làm lẫn dữ liệu giữa train và test.
        self.meta = pd.DataFrame({
            "track_id": tracks.index,
            "title": tracks[('track', 'title')],
            "genre": tracks[('track', 'genre_top')],
            "artist": tracks[('artist', 'name')],
        })
        self.meta = self._clean_metadata(self.meta)

        self.track_ids = self.meta["track_id"].tolist()
        if not self.track_ids:
            raise RuntimeError(
                f"Không tìm thấy file audio nào của subset '{subset}' trong: {fma_audio_dir}"
            )
        self.samples = [
            (meta_idx, segment_index)
            for meta_idx in range(len(self.meta))
            for segment_index in range(self.segments_per_track)
        ]
        print(
            f"Đã nạp {len(self.track_ids)} bài hát, "
            f"tạo {len(self.samples)} sample ({self.segments_per_track} đoạn/bài)."
        )

        # Lazy-load encoder: nếu cache đã đầy thì lần chạy sau không
        # cần nạp T5/EnCodec vào RAM/VRAM.
        self.t5_tokenizer = None
        self.t5_encoder = None
        self.encodec_processor = None
        self.encodec_model = None
        if self.use_cache:
            print(f"Cache encoded tensors: {self.cache_dir}")

    def _ensure_encoders_loaded(self):
        if self.t5_encoder is not None and self.encodec_model is not None:
            return
        print(f"Đang nạp T5 và EnCodec trên {self.device} cho cache miss...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(
            self.t5_model_name
        ).to(self.device).eval()
        self.encodec_processor = AutoProcessor.from_pretrained(
            self.encodec_model_name
        )
        self.encodec_model = EncodecModel.from_pretrained(
            self.encodec_model_name
        ).to(self.device).eval()
        # FP16 giảm gần một nửa VRAM và tăng tốc inference trên GPU
        # 4 GB; cache trên đĩa vốn đã lưu text embedding dạng FP16.
        if self.device == "cuda":
            self.t5_encoder.half()
            self.encodec_model.half()
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        for param in self.encodec_model.parameters():
            param.requires_grad = False

    def _cache_path(self, track_id: int, segment_index: int) -> str:
        return os.path.join(
            self.cache_dir, f"{track_id:06d}_segment{segment_index:02d}.pt"
        )

    def _cache_signature(self):
        return {
            "cache_version": DATA_CONFIG["cache_version"],
            "t5_model_name": self.t5_model_name,
            "encodec_model_name": self.encodec_model_name,
            "sample_rate": self.sample_rate,
            "audio_duration": self.audio_duration,
            "segments_per_track": self.segments_per_track,
            "text_max_length": DATA_CONFIG["text_max_length"],
            "codebook_size": MODEL_CONFIG["codebook_size"],
            "num_codebooks": MODEL_CONFIG["num_codebooks"],
        }

    def _validate_item(self, item):
        audio_tokens = item["audio_tokens"]
        expected_audio_shape = (
            MODEL_CONFIG["num_codebooks"],
            self.expected_audio_steps,
        )
        if tuple(audio_tokens.shape) != expected_audio_shape:
            raise InvalidTrackError(
                f"Audio token shape {tuple(audio_tokens.shape)}; "
                f"cần {expected_audio_shape}."
            )
        if audio_tokens.numel() == 0:
            raise InvalidTrackError("Audio token rỗng.")
        token_min = int(audio_tokens.min())
        token_max = int(audio_tokens.max())
        if token_min < 0 or token_max >= MODEL_CONFIG["codebook_size"]:
            raise InvalidTrackError(
                f"Audio token ngoài miền [0, {MODEL_CONFIG['codebook_size'] - 1}]: "
                f"min={token_min}, max={token_max}."
            )

        text_embeddings = item["text_embeddings"]
        expected_text_shape = (
            DATA_CONFIG["text_max_length"],
            MODEL_CONFIG["text_dim"],
        )
        if tuple(text_embeddings.shape) != expected_text_shape:
            raise InvalidTrackError(
                f"Text embedding shape {tuple(text_embeddings.shape)}; "
                f"cần {expected_text_shape}."
            )
        if not torch.isfinite(text_embeddings).all():
            raise InvalidTrackError("Text embedding chứa NaN/Inf.")

        attention_mask = item["text_attention_mask"]
        if tuple(attention_mask.shape) != (DATA_CONFIG["text_max_length"],):
            raise InvalidTrackError("Text attention mask sai shape.")
        if not isinstance(item["prompt"], str) or not item["prompt"].strip():
            raise InvalidTrackError("Prompt rỗng hoặc không hợp lệ.")

    def _load_cached_item(self, track_id: int, segment_index: int):
        if not self.use_cache:
            return None
        path = self._cache_path(track_id, segment_index)
        if not os.path.isfile(path):
            return None
        try:
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached.get("cache_signature") != self._cache_signature():
                return None
            item = {
                "audio_tokens": cached["audio_tokens"].long(),
                "text_embeddings": cached["text_embeddings"].float(),
                "text_attention_mask": cached["text_attention_mask"].long(),
                "prompt": cached["prompt"],
                "track_id": track_id,
                "segment_index": segment_index,
                "genre": cached["genre"],
            }
            self._validate_item(item)
            return item
        except (OSError, RuntimeError, KeyError, TypeError, AttributeError):
            return None

    def _save_cached_item(self, item):
        if not self.use_cache:
            return
        self._validate_item(item)
        path = self._cache_path(item["track_id"], item["segment_index"])
        temp_path = f"{path}.{os.getpid()}.tmp"
        compact = {
            "cache_signature": self._cache_signature(),
            "audio_tokens": item["audio_tokens"].to(torch.int16),
            "text_embeddings": item["text_embeddings"].to(torch.float16),
            "text_attention_mask": item["text_attention_mask"].to(torch.uint8),
            "prompt": item["prompt"],
            "genre": item["genre"],
        }
        torch.save(compact, temp_path)
        os.replace(temp_path, path)

    def populate_cache(self):
        """Mã hóa các track còn thiếu một lần trước khi train."""
        if not self.use_cache:
            return
        missing_indices = []
        for idx, (meta_idx, segment_index) in enumerate(self.samples):
            track_id = self.track_ids[meta_idx]
            if self._load_cached_item(track_id, segment_index) is None:
                missing_indices.append(idx)
        if not missing_indices:
            print(f"Cache đã đầy: {len(self.samples)} sample.")
            return
        print(
            f"Đang tạo cache cho {len(missing_indices)}/{len(self.samples)} sample..."
        )
        invalid_tracks = []
        started_at = time.perf_counter()
        cache_batch_size = DATA_CONFIG["cache_batch_size"]
        for batch_start in range(0, len(missing_indices), cache_batch_size):
            batch_indices = missing_indices[
                batch_start:batch_start + cache_batch_size
            ]
            invalid_tracks.extend(self._encode_sample_batch(batch_indices))
            position = min(batch_start + len(batch_indices), len(missing_indices))
            if position % 20 == 0 or position == len(missing_indices):
                elapsed = time.perf_counter() - started_at
                tracks_per_second = position / max(elapsed, 1e-6)
                remaining = (len(missing_indices) - position) / tracks_per_second
                print(
                    f"   Cache [{position}/{len(missing_indices)}] | "
                    f"{tracks_per_second:.2f} track/s | ETA {remaining / 60:.1f} phút"
                )

        if invalid_tracks:
            invalid_keys = {
                (item["track_id"], item["segment_index"])
                for item in invalid_tracks
            }
            self.samples = [
                (meta_idx, segment_index)
                for meta_idx, segment_index in self.samples
                if (self.track_ids[meta_idx], segment_index) not in invalid_keys
            ]
            report_path = os.path.join(self.dataset_root, "invalid_tracks.csv")
            pd.DataFrame(invalid_tracks).to_csv(
                report_path, index=False, encoding="utf-8-sig"
            )
            print(
                f"Đã loại {len(invalid_tracks)} track lỗi; "
                f"báo cáo: {report_path}"
            )
            if not self.samples:
                raise RuntimeError("Không còn sample hợp lệ sau khi tạo cache.")

    def _encode_sample_batch(self, sample_indices):
        """Mã hóa nhiều crop cùng lúc để tận dụng GPU khi tạo cache."""
        self._ensure_encoders_loaded()
        prepared = []
        invalid = []
        for idx in sample_indices:
            meta_idx, segment_index = self.samples[idx]
            track_id = self.track_ids[meta_idx]
            audio_path = self._get_audio_path(track_id)
            try:
                waveform = self._load_and_process_audio(
                    audio_path,
                    segment_index=segment_index,
                    random_crop=False,
                )
                row = self.meta.iloc[meta_idx]
                prepared.append(
                    (track_id, segment_index, row, waveform, audio_path)
                )
            except InvalidTrackError as exc:
                invalid.append({
                    "track_id": track_id,
                    "segment_index": segment_index,
                    "audio_path": audio_path,
                    "error": str(exc),
                })
                print(
                    f"   [BỎ QUA] Track {track_id}, segment {segment_index}: {exc}"
                )

        if not prepared:
            return invalid

        raw_audio = [item[3].squeeze(0).numpy() for item in prepared]
        inputs = self.encodec_processor(
            raw_audio=raw_audio,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        prompts = [
            self._generate_text_prompt(item[2], item[1]) for item in prepared
        ]
        text_inputs = self.t5_tokenizer(
            prompts,
            padding="max_length",
            max_length=DATA_CONFIG["text_max_length"],
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            audio_inputs = inputs["input_values"].to(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            padding_mask = inputs.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            audio_codes = self.encodec_model.encode(
                audio_inputs, padding_mask
            ).audio_codes

            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            text_embeddings = self.t5_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.cpu()

        for batch_idx, (track_id, segment_index, row, _, _) in enumerate(prepared):
            item = {
                "audio_tokens": audio_codes[0, batch_idx].cpu(),
                "text_embeddings": text_embeddings[batch_idx],
                "text_attention_mask": text_inputs.attention_mask[batch_idx].cpu(),
                "prompt": prompts[batch_idx],
                "track_id": track_id,
                "segment_index": segment_index,
                "genre": row["genre"],
            }
            self._validate_item(item)
            self._save_cached_item(item)
        return invalid

    def release_encoders(self):
        """Giải phóng encoder sau khi cache đầy để nhường RAM/VRAM."""
        self.t5_tokenizer = None
        self.t5_encoder = None
        self.encodec_processor = None
        self.encodec_model = None
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_audio_path(self, track_id: int) -> str:
        tid_str = f"{track_id:06d}"
        folder = tid_str[:3]
        return os.path.join(self.fma_audio_dir, folder, f"{tid_str}.mp3")

    @staticmethod
    def _clean_text(value, fallback: str) -> str:
        """Chuẩn hóa text dùng trong prompt và loại ký tự điều khiển."""
        if pd.isna(value):
            return fallback
        value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value))
        value = re.sub(r"\s+", " ", value).strip()
        return value[:MAX_PROMPT_FIELD_LENGTH] or fallback

    def _audio_file_is_readable(self, track_id: int) -> bool:
        audio_path = self._get_audio_path(track_id)
        if not os.path.isfile(audio_path):
            return False
        try:
            info = sf.info(audio_path)
            return (
                info.frames > 0
                and info.samplerate > 0
                and info.channels > 0
                and info.duration >= DATA_CONFIG["minimum_audio_duration"]
            )
        except (RuntimeError, TypeError, ValueError):
            return False

    def _clean_metadata(self, meta: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(meta)
        meta = meta.copy()
        meta["track_id"] = pd.to_numeric(meta["track_id"], errors="coerce")
        meta = meta.dropna(subset=["track_id", "genre"])
        meta["track_id"] = meta["track_id"].astype("int64")
        meta = meta[meta["track_id"] >= 0]
        meta = meta.drop_duplicates(subset=["track_id"], keep="first")

        meta["genre"] = meta["genre"].map(lambda x: self._clean_text(x, ""))
        meta["title"] = meta["title"].map(
            lambda x: self._clean_text(x, "Unknown Title")
        )
        meta["artist"] = meta["artist"].map(
            lambda x: self._clean_text(x, "Unknown Artist")
        )
        meta = meta[meta["genre"] != ""]

        readable_mask = meta["track_id"].map(self._audio_file_is_readable)
        unreadable_count = int((~readable_mask).sum())
        meta = meta[readable_mask].reset_index(drop=True)
        print(
            "Làm sạch dataset: "
            f"giữ {len(meta)}/{initial_count} track; "
            f"loại {unreadable_count} file thiếu/hỏng."
        )
        return meta

    def _generate_text_prompt(self, row: pd.Series, segment_index: int) -> str:
        genre = row["genre"]
        templates = (
            "{genre} music.",
            "A {genre} music track.",
            "An instrumental track in the {genre} genre.",
            "A characteristic {genre} piece with a clear musical structure.",
        )
        return templates[segment_index % len(templates)].format(genre=genre)

    def _load_and_process_audio(
            self, audio_path: str, segment_index: int, random_crop: bool = False
    ) -> torch.Tensor:
        # soundfile đọc MP3 trực tiếp qua libsndfile, tránh phụ thuộc TorchCodec/FFmpeg
        # của torchaudio.load trên các phiên bản torchaudio mới.
        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidTrackError(f"Không decode được audio: {exc}") from exc
        waveform = torch.from_numpy(audio.T)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise InvalidTrackError("Audio rỗng hoặc chứa NaN/Inf.")

        # Loại DC offset và các file gần như im lặng. Chỉ scale khi
        # biên độ vượt [-1, 1], không làm thay đổi loudness bình thường.
        waveform = waveform - waveform.mean(dim=1, keepdim=True)
        rms = waveform.square().mean().sqrt().item()
        if rms < MIN_AUDIO_RMS:
            raise InvalidTrackError(f"Audio quá im lặng (RMS={rms:.2e}).")
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak
        waveform = waveform.clamp(-1.0, 1.0)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        num_samples = waveform.shape[1]
        if num_samples > self.target_samples:
            max_start = num_samples - self.target_samples
            if random_crop:
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                denominator = max(self.segments_per_track - 1, 1)
                start = round(max_start * segment_index / denominator)
            waveform = waveform[:, start:start + self.target_samples]
        elif num_samples < self.target_samples:
            pad_len = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __len__(self):
        return len(self.samples)

    def sample_track_id(self, idx: int) -> int:
        meta_idx, _ = self.samples[idx]
        return self.track_ids[meta_idx]

    def sample_genre(self, idx: int) -> str:
        meta_idx, _ = self.samples[idx]
        return str(self.meta.iloc[meta_idx]["genre"])

    def __getitem__(self, idx: int):
        return self.get_item(idx, random_crop=True)

    def get_item(self, idx: int, random_crop: bool = True):
        """Lấy sample với crop ngẫu nhiên cho train hoặc crop giữa cho eval."""
        if not self.samples:
            raise RuntimeError("Dataset không có bài hát hợp lệ để xử lý.")

        meta_idx, segment_index = self.samples[idx]
        track_id = self.track_ids[meta_idx]
        audio_path = self._get_audio_path(track_id)
        row = self.meta.iloc[meta_idx]

        cached_item = self._load_cached_item(track_id, segment_index)
        if cached_item is not None:
            return cached_item

        try:
            self._ensure_encoders_loaded()
            # 1. Mã hóa Audio -> Acoustic Tokens
            waveform = self._load_and_process_audio(
                audio_path,
                segment_index=segment_index,
                random_crop=random_crop,
            )
            inputs = self.encodec_processor(
                raw_audio=waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )

            with torch.inference_mode():
                audio_inputs = inputs["input_values"].to(
                    self.device,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                padding_mask = inputs.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)
                encodec_out = self.encodec_model.encode(audio_inputs, padding_mask)
                audio_tokens = encodec_out.audio_codes[0].squeeze(0).squeeze(0).cpu()

            # 2. Mã hóa Text Prompt -> Text Embeddings
            text_prompt = self._generate_text_prompt(row, segment_index)
            text_inputs = self.t5_tokenizer(
                text_prompt,
                padding="max_length",
                max_length=DATA_CONFIG["text_max_length"],
                truncation=True,
                return_tensors="pt"
            )

            with torch.inference_mode():
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)
                t5_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = t5_outputs.last_hidden_state.squeeze(0).cpu()

            item = {
                "audio_tokens": audio_tokens,
                "text_embeddings": text_embeddings,
                "text_attention_mask": text_inputs.attention_mask.squeeze(0).cpu(),
                "prompt": text_prompt,
                "track_id": track_id,
                "segment_index": segment_index,
                "genre": row["genre"],
            }
            self._validate_item(item)
            self._save_cached_item(item)
            return item

        except InvalidTrackError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Không thể xử lý track {track_id} tại '{audio_path}': {exc}"
            ) from exc


def create_dataloader(
        dataset_dir: str = None,
        batch_size: int = TRAIN_CONFIG["batch_size"],
        shuffle: bool = True,
        auto_download: bool = True,
        num_workers: int = 0,
        **dataset_kwargs
) -> DataLoader:
    if num_workers != 0:
        raise ValueError(
            "num_workers phải bằng 0 vì Dataset chứa trực tiếp các model T5 và EnCodec."
        )

    if dataset_dir is None:
        dataset_dir = DATA_CONFIG["dataset_dir"]

    if auto_download:
        download_and_extract_fma(
            dataset_dir=dataset_dir, subset=DATA_CONFIG["subset"]
        )

    fma_metadata_dir = os.path.join(dataset_dir, "fma_metadata")
    fma_audio_dir = os.path.join(dataset_dir, "fma_small")

    dataset = FMAAudioTextDataset(
        fma_metadata_dir=fma_metadata_dir,
        fma_audio_dir=fma_audio_dir,
        **dataset_kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == "__main__":
    print("=== KIỂM TRA PIPELINE DỮ LIỆU DATA.PY ===")
    dataloader = create_dataloader(
        batch_size=1,
        audio_duration=DATA_CONFIG["audio_duration"],
        auto_download=True,
    )

    for batch in dataloader:
        print("\n--- TEST NẠP BATCH THÀNH CÔNG ---")
        print("Audio Tokens Shape :", batch["audio_tokens"].shape)
        print("Text Embeddings Shape:", batch["text_embeddings"].shape)
        print("Sample Prompt       :", batch["prompt"][0])
        break
