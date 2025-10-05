"""
检查能够导入 Wav2vec 预训练模型, 以及能否预测或者能否得到对输入的梯度
"""

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder  # 用于 CTC beam-search 解码

def main():
    # 尝试加载预训练模型和处理器
    model_name = "facebook/wav2vec2-base-960h"

    # 下载并实例化音频预处理器，将原始波形转为模型输入格式
    # 用 transformers 里给的 Wav2Vec2Processor 已经做了特征提取、padding/截断等预处理
    # 完全不需要像 DeepSpeech 那样自己写一堆 STFT/MFCC 或者手动对齐
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # 下载并加载预训练的 Wav2Vec2-CTC 模型
    # 已经内置了一个 CTC head
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    # 构造 1s 静音输入
    sample_rate = 16000
    dummy_audio = np.zeros(sample_rate, dtype=np.float32)
    inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values  # [1, seq_len]
    # 启用梯度计算
    input_values.requires_grad_(True)

    # 前向推理
    logits = model(input_values).logits  # [B, T, C]
    print("Logits shape:", logits.shape)

    # 简单贪心解码来进行得到识别后的结果
    # 沿最后一位取最大值
    pred_ids = torch.argmax(logits, dim=-1)            # [B, T]
    # 调用内置的解码方法，将 token id 序列回字符(CTC 解码)
    transcripts = processor.batch_decode(pred_ids)     # List[str]
    print("Transcription (greedy):", transcripts)

    # CTC beam-search 解码（使用 pyctcdecode）
    # 构造 vocab 列表（按 id 顺序排序）
    vocab_dict = processor.tokenizer.get_vocab()
    vocab = [k for k, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
    decoder = build_ctcdecoder(vocab)
    # 转概率并转成 numpy
    emissions = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]
    # beam_width 可根据需求调整
    beam_result = decoder.decode(emissions, beam_width=50)
    print("Transcription (beam search):", beam_result)

    # 简单构造一个标量 loss，并反向以验证能否拿到梯度
    loss = logits.sum()
    loss.backward()
    print("Gradient shape:", input_values.grad.shape)
    print("Gradient sample:", input_values.grad[0, :10])

if __name__ == "__main__":
    main()