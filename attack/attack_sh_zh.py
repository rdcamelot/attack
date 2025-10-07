import argparse
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import editdistance

from utils_sh import load_audio, save_audio, get_alignment, compute_snr
from loss_sh import attack_loss


def main():
    parser = argparse.ArgumentParser(description="中文语音对抗攻击脚本")
    parser.add_argument("--input", required=True, help="输入中文音频路径 (.wav/.flac)")
    parser.add_argument("--output", default=None, help="对抗样本保存路径")
    parser.add_argument("--model", default="voidful/wav2vec2-large-zh-cn-187k",
                        help="中文预训练 wav2vec2-CTC 模型名称或路径")
    parser.add_argument("--iterations", type=int, default=100, help="最大迭代次数")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--c", type=float, default=1.0, help="loss 中 margin 权重 c")
    parser.add_argument("--alpha", type=float, default=1.0, help="recognition_score alpha")
    parser.add_argument("--k", type=float, default=1.0, help="step_function k")
    parser.add_argument("--sr", type=int, default=16000, help="采样率，用于保存音频")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载中文模型及 processor
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    # 读取音频并对齐获取原始 logits 和 labels
    x_orig = load_audio(args.input).to(device)
    logits_orig, y_f = get_alignment(x_orig, processor, model, device)
    orig_ids = torch.argmax(logits_orig, dim=-1)
    orig_text = processor.batch_decode([orig_ids])[0].strip()
    print(f"[i] 原始转录: {orig_text}")

    # 初始化扰动
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    success = False
    adv_text = orig_text
    for it in range(1, args.iterations + 1):
        # 构造对抗样本并截断
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
        # 前向
        logits_adv = model(x_adv.unsqueeze(0)).logits.squeeze(0)
        # 计算 loss
        loss = attack_loss(logits_adv, y_f, delta, args.c, args.alpha, args.k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 解码并判断
        adv_ids = torch.argmax(logits_adv, dim=-1)
        adv_text = processor.batch_decode([adv_ids])[0].strip()
        if adv_text != orig_text:
            success = True
            print(f"[+] 对抗成功 at iter {it}, adv_text: {adv_text}")
            break
        if it % 10 == 0:
            print(f"Iter {it}/{args.iterations}: loss={loss.item():.6e}, text={adv_text}")

    # 保存对抗音频
    if args.output:
        out_path = args.output
    else:
        # 默认保存到 same folder with suffix
        base, ext = args.input.rsplit('.', 1)
        out_path = base + '_zh_adv.' + ext
    x_final = torch.clamp(x_orig + delta.detach(), -1.0, 1.0)
    save_audio(out_path, x_final, sr=args.sr)
    print(f"已保存对抗样本: {out_path}")

    # 计算 SNR
    snr = compute_snr(x_orig, delta.detach())
    # 计算 CER
    cer = editdistance.eval(orig_text, adv_text) / max(len(orig_text), 1)

    # 打印总结
    print(f"Adversarial 转录: {adv_text}")
    print(f"SNR(dB): {snr:.2f}")
    print(f"CER: {cer:.2%}")
    print(f"Attack {'Succeeded' if success else 'Failed'}")

if __name__ == '__main__':
    main()
