"""
utils_sh.py

辅助函数模块：
- 音频读写
- CTC 贪心对齐(获取每帧标签 y_f)
- 攻击评估(成功判断、SNR 计算)

用于对齐标签、音频加载与保存
"""
import torch
import soundfile as sf
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    从文件加载音频，并转为单声道 float32 Tensor。
    用于和语音识别模型保持一致
    """
    audio, sr = sf.read(path)
    if sr != target_sr:
        raise ValueError(f"{path} 采样率 {sr} != {target_sr}")
    # 只保留第一通道
    if audio.ndim > 1:
        audio = audio[:, 0]
    return torch.from_numpy(audio.astype("float32"))


def save_audio(path: str, audio: torch.Tensor, sr: int = 16000) -> None:
    """
    保存 Tensor 音频到文件，若父目录不存在则创建。
    """
    # 1. 确保目录存在
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    # 2. 写文件
    # 确保父目录存在
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    data = audio.detach().cpu().numpy()
    sf.write(path, data, sr)

"""
margin loss 是逐帧的, 因此必须把原始转录对齐到每个时间步才能得到 labels y_f
"""
def get_alignment(audio: torch.Tensor,
                  processor: Wav2Vec2Processor,
                  model: Wav2Vec2ForCTC,
                  device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对输入 waveform 做 CTC 贪心对齐，返回 logits [T, C] 和每帧预测标签 y_f [T]。
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = processor(audio,
                           sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="pt",
                           padding=True)
        input_values = inputs.input_values.to(device)
        logits = model(input_values).logits.squeeze(0)  # [T, C]
    y_f = torch.argmax(logits, dim=-1)  # [T]
    return logits, y_f


def is_attack_success(orig_y: torch.Tensor, adv_y: torch.Tensor) -> bool:
    """
    判断 untargeted 攻击是否成功：任何一帧预测标签变化即认为成功。
    """
    return not torch.equal(orig_y, adv_y)


def compute_snr(orig: torch.Tensor, delta: torch.Tensor) -> float:
    """
    计算信噪比 SNR(dB): 10 * log10( sum(orig^2) / sum(delta^2) )
    """
    orig_power = torch.sum(orig ** 2)
    noise_power = torch.sum(delta ** 2)
    return 10 * torch.log10(orig_power / (noise_power + 1e-8)).item()
