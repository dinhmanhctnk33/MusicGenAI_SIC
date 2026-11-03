import argparse
import os
import shutil
import subprocess
import tempfile

import soundfile as sf
import torch
from transformers import AutoProcessor, EncodecModel, T5EncoderModel, T5Tokenizer

from model import MusicGenTransformer


DEFAULT_CONFIG = {
    "codebook_size": 1024,
    "num_codebooks": 4,
    "text_dim": 768,
    "d_model": 256,
    "num_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1024,
    "max_seq_len": 512,
}


class MusicGenTester:
    """Generate audio from a text prompt using a trained checkpoint."""

    def __init__(self, checkpoint_path, t5_model_name="t5-base",
                 encodec_model_name="facebook/encodec_32khz", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        print(f"-> Device: {self.device}")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {os.path.abspath(checkpoint_path)}"
            )
        print(f"-> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")

        saved_config = checkpoint.get("config", {})
        model_config = {
            key: saved_config.get(key, default)
            for key, default in DEFAULT_CONFIG.items()
        }
        if not saved_config:
            print("-> Warning: checkpoint has no config; using training defaults.")

        print("-> Loading T5 tokenizer and encoder...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name).to(self.device).eval()

        print("-> Loading EnCodec processor and decoder...")
        self.encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
        self.encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(self.device).eval()

        print(f"-> Model config: {model_config}")
        self.model = MusicGenTransformer(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_seq_len = model_config["max_seq_len"]
        print("-> Checkpoint loaded successfully.")

    @torch.inference_mode()
    def generate_music(self, prompt, output_path="generated_sample.mp3",
                       duration_sec=5.0, temperature=1.0, top_k=250):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        if duration_sec <= 0:
            raise ValueError("duration_sec must be greater than zero.")

        # EnCodec 32 kHz produces approximately 50 frames per second.
        max_steps = round(duration_sec * 50)
        max_allowed_steps = self.max_seq_len - self.model.num_codebooks + 1
        if max_steps > max_allowed_steps:
            max_duration = max_allowed_steps / 50
            raise ValueError(
                f"Maximum duration is about {max_duration:.2f}s "
                f"for max_seq_len={self.max_seq_len}."
            )

        print(f"\n[1/3] Processing prompt: {prompt!r}")
        text_inputs = self.t5_tokenizer(
            prompt, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        text_embeddings = self.t5_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        print(f"[2/3] Generating {max_steps} token steps ({duration_sec:.2f}s)...")
        generated_tokens = self.model.generate(
            text_embeddings=text_embeddings,
            text_attention_mask=attention_mask,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
        )

        print("[3/3] Decoding tokens to waveform...")
        # EnCodec expects [num_frames, batch, num_codebooks, frame_length].
        audio_codes = generated_tokens.unsqueeze(0)
        waveform = self.encodec_model.decode(audio_codes, [None]).audio_values
        waveform = waveform[0].detach().float().cpu().transpose(0, 1).numpy()

        output_path = os.path.abspath(output_path)
        if not output_path.lower().endswith(".mp3"):
            raise ValueError("output_path must have the .mp3 extension.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg is required to create MP3 files. "
                "On Colab, install it with: !apt-get install -y ffmpeg"
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            sf.write(temp_path, waveform, 32000, subtype="PCM_16")
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    temp_path,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    output_path,
                ],
                check=True,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"-> Audio saved to: {output_path}")
        return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music from a MusicGen checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "latest_musicgen_model.pt"),
        help="Checkpoint .pt path (mounted Google Drive paths are supported).",
    )
    parser.add_argument(
        "--prompt",
        default="A joyful upbeat Rock track with electric guitar and energetic drums.",
    )
    parser.add_argument("--output", default="generated_sample.mp3")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def run_test():
    args = parse_args()
    tester = MusicGenTester(args.checkpoint, device=args.device)
    tester.generate_music(
        prompt=args.prompt,
        output_path=args.output,
        duration_sec=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    run_test()
