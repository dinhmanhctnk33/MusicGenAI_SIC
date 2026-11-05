# MusicGenAI — huấn luyện hoàn toàn từ đầu

MusicGenAI là hệ thống text-to-music thử nghiệm trên FMA Small. Toàn bộ tokenizer,
text encoder, neural audio codec và Transformer sinh nhạc đều được định nghĩa và
huấn luyện trong dự án; không tải T5, EnCodec hay bất kỳ trọng số pretrained nào.

## Kiến trúc

```text
Prompt Việt/Anh
  → UTF-8 ByteTokenizer
  → Transformer text encoder (4 lớp)
  → Transformer decoder tự hồi quy (6 lớp, CFG)
  → 8 luồng acoustic token
  → RVQ + CNN audio decoder
  → WAV 32 kHz

Audio huấn luyện
  → CNN audio encoder
  → Residual Vector Quantizer (8 × 1.024 code)
  → acoustic token ở 50 frame/giây
```

Pipeline được huấn luyện theo hai giai đoạn:

1. Huấn luyện neural audio codec để tái tạo waveform.
2. Đóng băng codec, tạo cache acoustic token và huấn luyện Music Transformer.

## Tính năng

- Không sử dụng model hoặc tokenizer pretrained.
- Prompt song ngữ Việt–Anh, mã hóa trực tiếp bằng UTF-8 byte.
- Neural codec CNN + Residual Vector Quantizer tự huấn luyện.
- 8 codebook × 1.024 code ở 50 frame/giây, bitrate xấp xỉ 4 kbps.
- Transformer decoder dùng delay pattern và classifier-free guidance.
- Chia train/validation/test theo track để tránh rò rỉ các đoạn cùng bài hát.
- Tự loại audio hỏng, rỗng, NaN/Inf hoặc quá im lặng.
- Cache token có thể tiếp tục sau khi gián đoạn.
- Hiển thị loss và token accuracy tổng/từng codebook.
- Lưu latest/best checkpoint và hỗ trợ resume.
- FastAPI frontend, lịch sử SQLite và xuất WAV/MP3.

## Yêu cầu

- Windows 10/11 và Python 3.12.
- [uv](https://docs.astral.sh/uv/) để quản lý môi trường.
- GPU NVIDIA hỗ trợ CUDA được khuyến nghị; cấu hình mặc định hướng tới GPU 4 GB.
- Internet chỉ cần khi tải FMA Small lần đầu.
- Dung lượng trống cho FMA, cache token, checkpoint và WAV.

## Cài đặt

```powershell
git clone https://github.com/dinhmanhctnk33/MusicGenAI_SIC.git
cd MusicGenAI_SIC
uv sync
```

Kiểm tra GPU:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Trình tự chạy chuẩn

### 1. Huấn luyện neural audio codec

Chạy mới từ đầu:

```powershell
$env:MUSICGEN_CODEC_RESUME="none"
uv run train_codec.py
```

Checkpoint và audio đối chiếu được lưu tại:

```text
checkpoints/neural_codec_v1/
├── latest_codec.pt
├── best_codec.pt
├── original_best.wav
└── reconstruction_best.wav
```

Nếu bị gián đoạn, tiếp tục từ `latest_codec.pt`:

```powershell
Remove-Item Env:MUSICGEN_CODEC_RESUME -ErrorAction SilentlyContinue
uv run train_codec.py
```

Phải nghe `original_best.wav` và `reconstruction_best.wav`. Chỉ chuyển sang giai
đoạn MusicGen khi reconstruction nghe rõ nội dung nhạc, không im lặng và không chỉ
là tiếng ù. Transformer không thể khôi phục thông tin mà codec làm mất.

### 2. Tạo cache và huấn luyện Music Transformer

```powershell
$env:MUSICGEN_RESUME="none"
uv run train.py
```

`train.py` sẽ tự động:

1. Nạp `best_codec.pt`.
2. Kiểm tra và loại track lỗi.
3. Tạo cache acoustic token còn thiếu.
4. Chia dữ liệu theo track.
5. Huấn luyện, validation, early stopping và lưu checkpoint.

Nếu bị gián đoạn, tiếp tục từ latest checkpoint:

```powershell
Remove-Item Env:MUSICGEN_RESUME -ErrorAction SilentlyContinue
uv run train.py
```

## Dữ liệu và cache

FMA Small được tải tự động vào `dataset/`. Mỗi track tạo 6 đoạn dài 5 giây,
tương đương gần 48.000 sample trước khi loại track lỗi.

Mỗi cache item gồm:

- `audio_tokens`: `[8, 250]`.
- `text_token_ids`: `[64]`.
- `text_attention_mask`: `[64]`.
- Prompt, genre, track ID và segment index.

Cache neural codec:

```text
dataset/encoded_cache_v5_neural_codec_bilingual/sr32000_duration5000ms/
```

Track lỗi được ghi tại:

```text
dataset/invalid_tracks.csv
```

Cache v1–v4 và checkpoint dùng T5, EnCodec hoặc codec DCT không tương thích với
pipeline hiện tại.

## Cấu hình mặc định

### Neural audio codec

| Thông số | Giá trị |
|---|---:|
| Sample rate | 32.000 Hz |
| Độ dài crop train | 2 giây |
| Token rate | 50 frame/giây |
| Codebook | 8 × 1.024 code |
| Latent dimension | 128 |
| Epoch tối đa | 50 |
| Early stopping patience | 7 |

Codec dùng waveform L1 loss, multi-resolution STFT loss và RVQ commitment loss.

### Music Transformer

| Thông số | Giá trị |
|---|---:|
| Audio mỗi sample | 5 giây |
| Text length | 64 byte token |
| `d_model` | 384 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 1.536 |
| Epoch tối đa | 30 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Early stopping patience | 5 |

Accuracy được tính theo tỷ lệ acoustic token top-1 trùng target. Đây chỉ là metric
phụ; validation loss, chất lượng codec reconstruction và audio sinh thực tế quan
trọng hơn.

## Đánh giá

```powershell
uv run test.py --mode evaluate `
  --checkpoint checkpoints/neural_codec_bilingual_v5/best_musicgen_model.pt
```

## Sinh nhạc

Prompt tiếng Việt:

```powershell
uv run test.py --mode generate `
  --checkpoint checkpoints/neural_codec_bilingual_v5/best_musicgen_model.pt `
  --prompt "Một bản nhạc Rock sôi động." `
  --output outputs/rock_vi.wav
```

Prompt tiếng Anh:

```powershell
uv run test.py --mode generate `
  --checkpoint checkpoints/neural_codec_bilingual_v5/best_musicgen_model.pt `
  --prompt "An energetic Rock track." `
  --output outputs/rock_en.wav
```

## Frontend

```powershell
uv run uvicorn frontend.app:app --host 127.0.0.1 --port 8000
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Cấu trúc dự án

```text
Music-GenAI/
├── audio_codec.py          # CNN encoder/decoder, RVQ và STFT loss
├── train_codec.py          # Huấn luyện neural audio codec
├── scratch_components.py  # ByteTokenizer và text encoder
├── data.py                 # FMA loader, làm sạch, crop và token cache
├── model.py                # Music Transformer và CFG generation
├── train.py                # Train/validation/test/checkpoint MusicGen
├── test.py                 # Đánh giá và sinh WAV/MP3
├── config.py               # Cấu hình tập trung
└── frontend/               # FastAPI và giao diện web
```

## Lưu ý

- Không train Music Transformer trước khi codec reconstruction đạt chất lượng chấp nhận được.
- Khi đổi kiến trúc codec hoặc số codebook, phải tạo cache và train MusicGen lại.
- Nếu CUDA hết bộ nhớ, giảm `CODEC_TRAIN_CONFIG["batch_size"]` xuống `1`.
- Dataset, cache, checkpoint, WAV và SQLite local không nên commit lên Git.
- FMA có giấy phép riêng cho metadata và từng track; người dùng phải tuân thủ giấy
  phép của [Free Music Archive](https://github.com/mdeff/fma).
