import os
import glob
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

def batch_recognize(data_dir: str,
                    model_name: str = "facebook/wav2vec2-base-960h",
                    batch_size: int = 8,
                    beam_width: int = 50,
                    use_beam: bool = True,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    # 递归查找所有 wav 和 flac 文件，按照路径排序
    paths = sorted(glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True)
                   + glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    # 加载模型和 processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    # 准备 pyctcdecode decoder（可选 LM）
    # 从 processor 的 tokenizer 获取 token->id 映射
    # 按照 id 排序得到 id->token 列表
    vocab_dict = processor.tokenizer.get_vocab()
    vocab = [t for t, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
    # 构造不带 LM 的解码器
    decoder = build_ctcdecoder(vocab)

    # 存放识别结果
    results = {}
    # 按 batch 读取、预处理、推理、解码
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        audios = []
        for p in batch_paths:
            audio, sr = sf.read(p)
            # 检查采样率
            if sr != 16000:
                raise ValueError(f"{p} 采样率不是16kHz")
            # 如果是多声道，只取第一通道
            if audio.ndim > 1:
                audio = audio[:, 0]
            audios.append(audio.astype("float32"))
        # 调用 processor 对一批波形做特征提取、padding
        # 返回 PyTorch 张量
        inputs = processor(audios,
                           sampling_rate=16000,
                           return_tensors="pt",
                           padding=True)
        # 移动到指定设备
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits  # [B, T, C]
        # greedy 解码
        pred_ids = torch.argmax(logits, dim=-1)
        greedy_texts = processor.batch_decode(pred_ids)
        # beam search 解码
        beam_texts = []
        if use_beam:
            emissions = torch.softmax(logits, dim=-1).cpu().numpy()
            for em in emissions:
                beam_texts.append(decoder.decode(em, beam_width=beam_width))
        # 汇总结果
        for p, g, b in zip(batch_paths, greedy_texts,
                           beam_texts if use_beam else [""] * len(batch_paths)):
            results[p] = {
                "greedy": g,
                "beam": b
            }
    return results

if __name__ == "__main__":
    data_folder = "./data/LibriSpeech/test-clean/121/121726"  # 指向数据目录
    out = batch_recognize(data_folder, batch_size=4, device = "cpu")
    for path, txt in out.items():
        print(f"{os.path.basename(path)} → greedy: {txt['greedy']}    beam: {txt['beam']}")