import librosa
import ffmpeg
import numpy as np
import pyaudio
import torch
from torch import nn
import torch.nn.functional as F
from transformers import HubertModel

from module.quantize import ResidualVectorQuantizer
from module.models import TextEncoder, ResidualCouplingBlock, Generator
from module.modules import MelStyleEncoder

from module.mel_processing import spectrogram_torch

device = "cuda"

# Load VITS model
sovits_path = "./GPT_SoVITS/pretrained_models/s2G488k.pth"
cnhubert_base_path = "./GPT_SoVITS/pretrained_models/chinese-hubert-base"
# vits = torch.load(sovits_path, map_location=device)

bert = HubertModel.from_pretrained(cnhubert_base_path).to(device)


def ori_wav_encoder(wav_path):
    wav16k, sr = librosa.load(wav_path, sr=16000)
    wav16k = torch.from_numpy(wav16k).to(device)
    wav_emb = bert(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
    return wav_emb


def prompt_wav_encoder(wav_path, alpha=1):
    out, _ = (
        ffmpeg.input(wav_path, threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=32000)  # hps.data.sampling_rate = 32000
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    wav = np.frombuffer(out, np.float32).flatten()
    audio = torch.FloatTensor(wav)
    audio_norm = audio.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        2048*alpha,  # hps.data.filter_length,
        32000,  # hps.data.sampling_rate,
        640*alpha,  # hps.data.hop_length,
        2048*alpha,  # hps.data.win_length,
        center=False,
    ).to(device)
    return spec


class VC(nn.Module):

    def __init__(self,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spec_channels,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        ssl_dim = 768
        self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )  # Rewrite this
        self.ref_enc = MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

    def forward(self, ori_wav_emb, text_phones, spec):
        # 原始音频处理
        ssl = self.ssl_proj(ori_wav_emb)  # 1x768x128
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)  # codes: 1x1x128
        codes = codes.transpose(0, 1)  # [n_q, B, T] -> [B, n_q, T] 1x1x128
        # prompt_semantic = codes[0, 0] # 128
        quantized = self.quantizer.decode(codes)  # [B, D, T] 1x768x128
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )  # 1x768x256

        # 参考音频处理
        ge = self.ref_enc(spec)  # [B, D, T/1] 1x512x1

        # y, y_lengths, text, text_lengths, ge
        q_lengths = torch.LongTensor([quantized.shape[-1]]).to(device)
        # text = torch.LongTensor(text_phones)[None].to(device)
        text = text_phones.transpose(1, 2).to(device)
        # text_lengths = torch.LongTensor([len(text_phones)]).to(device)
        text_lengths = torch.LongTensor([text_phones.shape[2]]).to(device)

        _, m_p, logs_p, y_mask = self.enc_p(quantized, q_lengths, text, text_lengths, ge)

        noise_scale = 0.0  # 去除噪声添加？
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)  # 1x92x256
        o = self.dec((z * y_mask)[:, :, :], g=ge)  # 1x1xlength
        return o


def gen_wav_encoder(tensor_wav):
    audio = tensor_wav.detach().cpu().numpy()[0, 0]
    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    return audio


if __name__ == "__main__":
    from test import get_cleaned_text_final
    ori_wav_emb = ori_wav_encoder("./testdata/ori2.wav")  # 1x768x256
    text_phones, word2ph, norm_text = get_cleaned_text_final("你们在聊什么呢？", "zh")
    text_phones = prompt_wav_encoder("./testdata/ori2.wav", alpha=8)  # 1x1025x257
    spec = prompt_wav_encoder("./testdata/prompt.wav")  # 1x1025x23

    vc_model = VC(
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0,
        spec_channels=1025,
        resblock='1',
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=512
    ).to(device)

    # torch.save(vc_model.state_dict(), "pretvc_model.pth")
    vc_model.load_state_dict(torch.load('vc_model.pth'))

    tensor_wav = vc_model(ori_wav_emb, text_phones, spec)
    audio = gen_wav_encoder(tensor_wav)

    print(audio)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=32000,
                    output=True)
    stream.write(audio.astype(np.float32).tobytes())
