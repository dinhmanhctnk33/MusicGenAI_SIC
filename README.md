# MusicGenAI SIC

MusicGenAI SIC là dự án thử nghiệm sinh token âm nhạc có điều kiện từ mô tả văn bản. Pipeline sử dụng:

- **T5-base** để mã hóa tiêu đề, nghệ sĩ và thể loại thành text embeddings.
- **EnCodec 32 kHz** để chuyển âm thanh thành acoustic tokens.
- **Transformer Decoder** để học phân phối token âm thanh dựa trên điều kiện văn bản.
- **FMA Small** làm dữ liệu huấn luyện và kiểm thử.

## Cấu trúc dự án

```text
Music-GenAI/
├── config.py        # Hyperparameter tập trung cho data/model/train/generation
├── data.py          # Tải, chuẩn hóa và mã hóa dữ liệu FMA
├── model.py         # Mô hình MusicGen Transformer
├── train.py         # Huấn luyện, kiểm thử và lưu checkpoint
├── pyproject.toml   # Khai báo thư viện Python
├── uv.lock          # Khóa phiên bản thư viện
├── dataset/         # FMA metadata và audio (không lưu trực tiếp trong Git)
└── checkpoints/     # Checkpoint sinh ra khi huấn luyện
```

## Yêu cầu

- Windows 10/11
- Python 3.12
- GPU NVIDIA hỗ trợ CUDA; cấu hình hiện tại được tối ưu cho **RTX 2050 4 GB**
- Kết nối Internet cho lần đầu tải dataset và pretrained models
- Dung lượng trống đủ cho FMA Small và model cache

## Cài đặt

Dự án sử dụng [uv](https://docs.astral.sh/uv/) để quản lý môi trường và thư viện.

```cmd
git clone https://github.com/dinhmanhctnk33/MusicGenAI_SIC.git
cd MusicGenAI_SIC
uv sync
```

Kiểm tra PyTorch đã nhận GPU:

```cmd
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong đợi trên máy thử nghiệm:

```text
True
NVIDIA GeForce RTX 2050
```

## Chuẩn bị dữ liệu

Chạy:

```cmd
.\.venv\Scripts\python.exe data.py
```

Nếu chưa có dữ liệu, chương trình tự tải và giải nén:

- `fma_metadata`
- `fma_small`

Sau khi lọc, pipeline sử dụng 8.000 bài thuộc FMA Small. Mỗi mẫu gồm:

- `audio_tokens`: `[4, 250]`
- `text_embeddings`: `[64, 768]`
- `text_attention_mask`: `[64]`
- prompt và track ID

Audio được chuẩn hóa mono, 32 kHz và cắt/đệm về 5 giây. EnCodec sử dụng 4 codebook, mỗi codebook có 2.048 token.

## Kiểm tra mô hình

```cmd
.\.venv\Scripts\python.exe model.py
```

Cấu hình mặc định phù hợp GPU 4 GB:

| Tham số | Giá trị |
|---|---:|
| `d_model` | 256 |
| Số decoder layer | 4 |
| Attention heads | 8 |
| Feed-forward dimension | 1024 |
| Maximum sequence length | 512 |
| Codebook size | 2048 |
| Số codebook | 4 |

## Huấn luyện

Mọi hyperparameter được khai báo tập trung trong `config.py`. Nên chỉnh file này thay vì sửa trực tiếp `data.py`, `model.py`, `train.py` hay `test.py`.

```cmd
.\.venv\Scripts\python.exe train.py
```

Cấu hình huấn luyện hiện tại:

- Tối đa 30 epoch, early stopping sau 5 epoch không cải thiện
- Batch size 1
- Gradient accumulation 16 bước
- Automatic Mixed Precision FP16 trên CUDA
- Gradient clipping với `max_norm=1.0`
- AdamW và Cosine Annealing learning-rate scheduler
- Classifier-free guidance dropout 10%, guidance scale mặc định 2.5
- Chia dữ liệu theo tỷ lệ **90% train / 5% validation / 5% test**
- Mỗi track được chia thành 6 crop cố định, phủ từ đầu đến cuối bài
- T5 và EnCodec mã hóa theo batch 8 và lưu trong `dataset/encoded_cache_v2_multicrop`
- Train/validation/test chia theo track, không để crop của cùng bài rò sang tập khác
- Training dùng weighted sampler để cân bằng các genre
- GPU từ 4 GB trở lên tự dùng CUDA FP16 khi tạo cache

Validation set được đánh giá sau mỗi epoch để lưu best checkpoint và early stopping. Test set chỉ được đánh giá một lần sau khi train.

Checkpoint mới nhất được lưu tại:

```text
checkpoints/scratch_v2/latest_musicgen_model.pt
checkpoints/scratch_v2/best_musicgen_model.pt
```

Checkpoint chứa trạng thái model, optimizer, scheduler, AMP scaler, epoch, training loss và cấu hình.

Lần chạy đầu sẽ tạo cache và có thể mất nhiều thời gian. Các lần chạy và epoch sau sẽ đọc cache trực tiếp. Có thể ép thiết bị encoder bằng biến môi trường `MUSICGEN_ENCODER_DEVICE=cpu` hoặc `MUSICGEN_ENCODER_DEVICE=cuda`.

## Tiếp tục huấn luyện

Nếu `checkpoints/scratch_v2/latest_musicgen_model.pt` tồn tại, lệnh sau tự động
tiếp tục từ epoch đã lưu:

```powershell
uv run train.py
```

## Giao diện MusicGenAI Studio

Khởi động hệ thống local:

```powershell
uv run frontend/app.py
```

Trình duyệt sẽ mở `http://127.0.0.1:7860`. Hệ thống sử dụng
`frontend/index.html`, `frontend/style.css` và `frontend/app.js`, hỗ trợ sinh nhạc,
nghe WAV và lưu lịch sử hội thoại bằng SQLite.

Để cố ý train lại từ đầu trong PowerShell:

```powershell
$env:MUSICGEN_RESUME="none"
uv run train.py
```

## Lưu ý

- Dataset, môi trường `.venv` và checkpoint không nên commit trực tiếp vào Git thông thường vì có kích thước lớn.
- T5 và EnCodec hiện mã hóa dữ liệu trực tuyến nên mỗi epoch có thể mất nhiều thời gian.
- Khi gặp `CUDA device-side assert`, cần đóng tiến trình Python cũ trước khi chạy lại vì lỗi CUDA có thể còn giữ trạng thái trong tiến trình.
- Nếu thiếu VRAM, giữ `batch_size=1`, audio 5 giây và không tăng kích thước model.

## Giấy phép dữ liệu

FMA có giấy phép riêng cho metadata và từng track âm thanh. Người sử dụng dự án cần tuân thủ giấy phép của [Free Music Archive](https://github.com/mdeff/fma) và giấy phép tương ứng của từng bản nhạc.
