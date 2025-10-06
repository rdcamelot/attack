"""
attack_sh.py

针对基于 CTC 的 wav2vec2 模型实现 untargeted 序列级攻击（节省扰动的攻击）。

攻击思路：
1. 加载原始音频；
2. 获取原始对齐标签 y_f；
3. 初始化扰动 δ；
4. 迭代优化 δ 使得模型输出序列 ≠ 原始标签；
5. 输出并保存攻击后音频与指标。

主脚本, 加载模型、迭代更新扰动并输出指标
"""
import argparse
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from attack.loss_sh import attack_loss
from attack.utils_sh import load_audio, save_audio, get_alignment, is_attack_success, compute_snr


def main():
    parser = argparse.ArgumentParser(description="Untargeted sequence-level attack on wav2vec2-CTC model")
    parser.add_argument("--input", type=str, required=True, help="原始音频文件路径 (.wav/.flac)")
    parser.add_argument("--output", type=str, required=True, help="保存对抗样本音频文件路径")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base-960h", help="预训练模型名")
    parser.add_argument("--iterations", type=int, default=100, help="优化迭代次数")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam 学习率")
    parser.add_argument("--c", type=float, default=1.0, help="损失中序列项权重 c")
    parser.add_argument("--alpha", type=float, default=1.0, help="recognition score 的 α")
    parser.add_argument("--k", type=float, default=1.0, help="step 函数的 k")
    args = parser.parse_args()

    # 设备选择：优先 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型与 processor
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    # 2. 加载音频并获取原始 logits 和对齐标签 y_f
    x_orig = load_audio(args.input).to(device)  # [L]
    logits_orig, y_f = get_alignment(x_orig, processor, model, device)  # logits: [T, C], y_f: [T]

    # 3. 初始化扰动 δ，需梯度
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)

    # 优化器
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    success = False
    for it in range(1, args.iterations + 1):
        # 合成对抗样本并裁剪到 [-1,1]
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)

        # 前向，得到 logits_adv
        inputs = processor(x_adv,
                           sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="pt",
                           padding=True)
        input_values = inputs.input_values.to(device)
        logits_adv = model(input_values).logits.squeeze(0)  # [T, C]

        # 4. 计算攻击损失并反向更新 δ
        loss = attack_loss(logits_adv, y_f, delta, args.c, args.alpha, args.k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 10 次打印进度
        if it % 10 == 0 or it == 1:
            # 计算当前对齐
            y_adv = torch.argmax(logits_adv, dim=-1)
            succ = is_attack_success(y_f, y_adv)
            print(f"Iter {it}/{args.iterations}: loss={loss.item():.4f}, success={succ}")

        if is_attack_success(y_f, torch.argmax(logits_adv, dim=-1)):
            success = True
            print(f"[+] Attack succeeded at iteration {it}")
            break

    # 5. 结果输出：保存音频与指标
    x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
    save_audio(args.output, x_adv)

    # 最终预测文本
    logits_final = logits_adv.detach()
    preds = torch.argmax(logits_final, dim=-1)
    text_orig = processor.batch_decode(y_f.unsqueeze(0))[0]
    text_adv = processor.batch_decode(preds.unsqueeze(0))[0]

    snr = compute_snr(x_orig.detach(), (x_adv - x_orig).detach())
    print(f"Original transcription: {text_orig}")
    print(f"Adversarial transcription: {text_adv}")
    print(f"SNR(dB): {snr:.2f}")
    print(f"Attack {'Succeeded' if success else 'Failed'}")


if __name__ == "__main__":
    main()
