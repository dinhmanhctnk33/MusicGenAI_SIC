import os
import re
import sys
import zipfile
import urllib.request
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import T5Tokenizer, T5EncoderModel
from transformers import EncodecModel, AutoProcessor


MIN_AUDIO_RMS = 1e-4
MAX_PROMPT_FIELD_LENGTH = 200


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def download_and_extract_fma(dataset_dir: str, subset: str = "small"):
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
            sample_rate: int = 32000,
            audio_duration: float = 5.0,
            subset: str = "small",
            t5_model_name: str = "t5-base",
            encodec_model_name: str = "facebook/encodec_32khz",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.fma_audio_dir = fma_audio_dir
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * audio_duration)
        self.device = device

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
        print(f"Đã nạp thành công {len(self.track_ids)} bài hát hợp lệ.")

        # 2. Khởi tạo Tokenizers và Pre-trained Encoders
        print("Đang tải/khởi tạo T5 Encoder và EnCodec Processor...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name).to(self.device).eval()

        self.encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
        self.encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(self.device).eval()

        # Đóng đóng băng trọng số Encoder
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        for param in self.encodec_model.parameters():
            param.requires_grad = False

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
            return info.frames > 0 and info.samplerate > 0 and info.channels > 0
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

    def _generate_text_prompt(self, row: pd.Series) -> str:
        title = row["title"]
        artist = row["artist"]
        genre = row["genre"]
        return f"A {genre} music track titled '{title}' by {artist}."

    def _load_and_process_audio(
            self, audio_path: str, random_crop: bool = True
    ) -> torch.Tensor:
        # soundfile đọc MP3 trực tiếp qua libsndfile, tránh phụ thuộc TorchCodec/FFmpeg
        # của torchaudio.load trên các phiên bản torchaudio mới.
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError("Audio rỗng hoặc chứa NaN/Inf.")

        # Loại DC offset và các file gần như im lặng. Chỉ scale khi
        # biên độ vượt [-1, 1], không làm thay đổi loudness bình thường.
        waveform = waveform - waveform.mean(dim=1, keepdim=True)
        rms = waveform.square().mean().sqrt().item()
        if rms < MIN_AUDIO_RMS:
            raise ValueError(f"Audio quá im lặng (RMS={rms:.2e}).")
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
                start = max_start // 2
            waveform = waveform[:, start:start + self.target_samples]
        elif num_samples < self.target_samples:
            pad_len = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        return self.get_item(idx, random_crop=True)

    def get_item(self, idx: int, random_crop: bool = True):
        """Lấy sample với crop ngẫu nhiên cho train hoặc crop giữa cho eval."""
        if not self.track_ids:
            raise RuntimeError("Dataset không có bài hát hợp lệ để xử lý.")

        track_id = self.track_ids[idx]
        audio_path = self._get_audio_path(track_id)
        row = self.meta.iloc[idx]

        try:
            # 1. Mã hóa Audio -> Acoustic Tokens
            waveform = self._load_and_process_audio(
                audio_path, random_crop=random_crop
            )
            inputs = self.encodec_processor(
                raw_audio=waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )

            with torch.no_grad():
                audio_inputs = inputs["input_values"].to(self.device)
                padding_mask = inputs.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)
                encodec_out = self.encodec_model.encode(audio_inputs, padding_mask)
                audio_tokens = encodec_out.audio_codes[0].squeeze(0).squeeze(0).cpu()

            # 2. Mã hóa Text Prompt -> Text Embeddings
            text_prompt = self._generate_text_prompt(row)
            text_inputs = self.t5_tokenizer(
                text_prompt,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)
                t5_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = t5_outputs.last_hidden_state.squeeze(0).cpu()

            return {
                "audio_tokens": audio_tokens,
                "text_embeddings": text_embeddings,
                "text_attention_mask": text_inputs.attention_mask.squeeze(0).cpu(),
                "prompt": text_prompt,
                "track_id": track_id
            }

        except Exception as exc:
            raise RuntimeError(
                f"Không thể xử lý track {track_id} tại '{audio_path}': {exc}"
            ) from exc


def create_dataloader(
        dataset_dir: str = None,
        batch_size: int = 8,
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "dataset")

    if auto_download:
        download_and_extract_fma(dataset_dir=dataset_dir, subset="small")

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
    dataloader = create_dataloader(batch_size=1, audio_duration=5.0, auto_download=True)

    for batch in dataloader:
        print("\n--- TEST NẠP BATCH THÀNH CÔNG ---")
        print("Audio Tokens Shape :", batch["audio_tokens"].shape)
        print("Text Embeddings Shape:", batch["text_embeddings"].shape)
        print("Sample Prompt       :", batch["prompt"][0])
        break
