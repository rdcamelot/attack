"""
```bash
python .\attack\attack_debug.py `
  --input .\7729-102255-0005.flac `
  --iterations 100 `
  --lr 1e-2 `
  --c 1.0 `
  --alpha 1.0 `
  --k 1.0
```
"""

"""
用于分析攻击失败的原因

在之前的攻击代码的基础上添加了帧级margin和argmax token的调试输出

计算并统计：
- min_df / mean_df / max_df: 帧 margin 的最小/平均/最大值；
- num_changed: 与原始对齐标签不同的帧数；
- changed_idxs: 前 10 个变化帧的索引。
每 10 次迭代或到达最大迭代时打印这些信息。

利用这些输出来判断：
- 如果 num_changed 很小但是 adv_text 未变， 说明只是孤立帧被击破， CTC collapse 忽略了它；
- 如果 min_df 仍然很正或均值接近原始，说明大多数帧 margin 未被破坏，需要更强的整体扰动；

这样通过对比不同失败样本的变化分布, 看是否应该换用 collapse 后的 frame 针对更粗粒度做 margin 计算
"""

import argparse
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from attack.loss_sh import attack_loss
from attack.utils_sh import load_audio, save_audio, get_alignment, is_attack_success, compute_snr


def main():
    parser = argparse.ArgumentParser(description="Untargeted sequence-level attack on wav2vec2-CTC model")
    parser.add_argument("--input", type=str, required=True, help="原始音频文件路径 (.wav/.flac)")
    parser.add_argument("--output", type=str, default=None,
                        help="保存对抗样本音频路径，可选；默认在输入文件同目录，前缀 'attack_'")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base-960h", help="预训练模型名")
    parser.add_argument("--iterations", type=int, default=100, help="优化迭代次数")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam 学习率")
    parser.add_argument("--c", type=float, default=1.0, help="损失中序列项权重 c")
    parser.add_argument("--alpha", type=float, default=1.0, help="recognition score 的 α")
    parser.add_argument("--k", type=float, default=1.0, help="step 函数的 k")
    args = parser.parse_args()

    # 正常攻击流程
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    # 初始化 beam-search 解码器
    from pyctcdecode import build_ctcdecoder
    vocab_dict = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab_dict.items()}
    vocab = [id_to_token[i] for i in range(len(vocab_dict))]
    beam_decoder = build_ctcdecoder(vocab)

    x_orig = load_audio(args.input).to(device)  # [L]
    logits_orig, y_f = get_alignment(x_orig, processor, model, device)  # logits: [T, C], y_f: [T]
    """
    使用 贪心解码
    orig_ids = torch.argmax(logits_orig, dim=-1)
    orig_text = processor.batch_decode(orig_ids.unsqueeze(0))[0].upper().strip()
    """
    # 使用 beam-search 解码原始转录
    probs_orig = torch.softmax(logits_orig.detach(), dim=-1).cpu().numpy()
    orig_text = beam_decoder.decode(probs_orig, beam_width=10).upper().strip()

    print(f"[i] Original transcription: {orig_text}")

    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    success = False

    for it in range(1, args.iterations + 1):
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
        # 前向，直接使用 raw waveform 以保证梯度连通
        input_values = x_adv.unsqueeze(0).to(device)  # [1, L]
        logits_adv = model(input_values).logits.squeeze(0)  # [T, C]

        # Debug: 计算并打印帧级 margin 和 argmax token 变化情况
        frames = logits_adv.size(0)
        idx = torch.arange(frames, device=device)
        true_logits = logits_adv[idx, y_f]
        masked = logits_adv.clone()
        masked[idx, y_f] = float('-inf')
        other_max, _ = masked.max(dim=1)
        d_f = true_logits - other_max  # [T]
        argmax_tokens = torch.argmax(logits_adv, dim=-1)  # [T]
        # 统计信息
        min_df = d_f.min().item()
        mean_df = d_f.mean().item()
        max_df = d_f.max().item()
        num_changed = (argmax_tokens != y_f).sum().item()
        # 每 10 次或最后一次迭代打印帧级调试信息
        if it % 10 == 0 or it == args.iterations:
            print()
            changed_idxs = torch.nonzero(argmax_tokens != y_f).squeeze(1).tolist()
            print(f"Frame margin stats: min={min_df:.2f}, mean={mean_df:.2f}, max={max_df:.2f}")
            print(f"Changed frames: {num_changed}/{frames}, first indices {changed_idxs[:10]} ...")
        # === End ===

        loss = attack_loss(logits_adv, y_f, delta, args.c, args.alpha, args.k)
        optimizer.zero_grad()
        loss.backward()
        # if delta.grad is not None:
        #     grad_norm = delta.grad.norm().item()
        # else:
        #     grad_norm = float('nan')
        # print(f"Debug: iteration {it}, delta.grad.norm={grad_norm:.6f}")
        optimizer.step()
        # delta_norm = delta.norm().item()
        # print(f"Debug: iteration {it}, delta.norm={delta_norm:.6f}")

        # 每 10 次或首次打印当前迭代信息
        if it % 10 == 0 or it == 1:
            """
            # 贪心解码对抗样本 logits 得到当前转录
            adv_ids = torch.argmax(logits_adv, dim=-1)
            adv_text = processor.batch_decode(adv_ids.unsqueeze(0))[0].upper().strip()
            # 序列级判断：只要文本不同即视为攻击成功
            succ = (adv_text != orig_text)
            print(
                f"Iter {it}/{args.iterations}: loss={loss.item():.6e}, "
                f"adv_text={adv_text!r}, success={succ}"
            )
            """
            # Beam-search 解码对抗样本
            adv_text = beam_decoder.decode(
                torch.softmax(logits_adv.detach(), dim=-1).cpu().numpy(), beam_width=10
            ).upper().strip()
            # 序列级判断：只要文本不同即视为攻击成功
            succ = (adv_text != orig_text)
            print(
                 f"Iter {it}/{args.iterations}: loss={loss.item():.6e}, "
                 f"adv_text={adv_text!r}, success={succ}"
             )

        """
        贪心解码
        adv_ids = torch.argmax(logits_adv, dim=-1)
        adv_text = processor.batch_decode(adv_ids.unsqueeze(0))[0].upper().strip()
        """
        # 最后使用 beam-search 解码
        adv_text = beam_decoder.decode(
            torch.softmax(logits_adv.detach(), dim=-1).cpu().numpy(), beam_width=10
        ).upper().strip()
        if adv_text != orig_text:
            success = True
            print()
            print(f"[+] Attack succeeded at iteration {it}, adv_text={adv_text}\n")
            break

    # 结果输出：保存音频与指标
    x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
    # 保存对抗音频：若未指定输出路径，默认在原文件同目录，加前缀 'attack_'
    if args.output:
        output_path = args.output
    else:
        base = os.path.basename(args.input)
        folder = os.path.dirname(args.input)
        # 在输入文件同目录生成前缀 'attack_' 的文件名
        output_path = os.path.join(folder, f"attack_{base}")
    save_audio(output_path, x_adv)

    # 最终使用 beam-search 解码原始和对抗文本
    logits_final = logits_adv.detach()
    """
    使用贪心解码
    preds = torch.argmax(logits_final, dim=-1)
    text_orig = processor.batch_decode(y_f.unsqueeze(0))[0]
    text_adv = processor.batch_decode(preds.unsqueeze(0))[0]
    """

    probs_final = torch.softmax(logits_final, dim=-1).detach().cpu().numpy()
    final_res = beam_decoder.decode(probs_final, beam_width=10)
    text_adv = final_res
    # 原始文本已用 beam decode，直接复用 orig_text
    text_orig = orig_text

    snr = compute_snr(x_orig.detach(), (x_adv - x_orig).detach())
    print(f"已保存对抗样本: {output_path}")
    print(f"Original transcription:\n {text_orig}")
    print(f"Adversarial transcription:\n {text_adv}")
    print(f"SNR(dB): {snr:.2f}")
    print(f"Attack {'Succeeded' if success else 'Failed'}")


if __name__ == "__main__":
    main()
