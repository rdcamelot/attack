import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils_sh import load_audio, save_audio, get_alignment, compute_snr
from loss_sh import attack_loss


def attack_single(x_orig, y_f, orig_text, processor, model, device, args):
    """
    对单段音频执行攻击，返回是否成功、迭代次数和最终 SNR
    """
    # 初始化扰动 delta（可学习）并开启梯度追踪
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    # 使用 Adam 优化器只更新 delta
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    success = False
    iters = args.iterations

    for it in range(1, args.iterations + 1):
        # 1. 构造对抗样本 x_adv，并裁剪到合法范围 [-1,1]
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
        # 2. 前向计算 logits_adv: [T, C]
        logits_adv = model(x_adv.unsqueeze(0)).logits.squeeze(0)
        # 3. 计算 attack loss，包括 L2、margin、confidence 和 step 项
        loss = attack_loss(logits_adv, y_f, delta, args.c, args.alpha, args.k)
        # 4. 梯度清零、反向传播并更新 delta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 5. 解码对抗样本文本并判断是否与原始转录不同
        adv_ids = torch.argmax(logits_adv, dim=-1)
        adv_text = processor.batch_decode([adv_ids])[0].upper().strip()
        if adv_text != orig_text:
            success = True
            iters = it
            break
    # 保存对抗音频并计算 SNR
    x_final = torch.clamp(x_orig + delta.detach(), -1.0, 1.0)
    snr = compute_snr(x_orig, delta.detach())
    return success, iters, snr, x_final


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="批量生成对抗语音样本")
    parser.add_argument("--input_dir", required=True, help="输入音频文件夹路径")
    parser.add_argument("--output_dir", required=True, help="输出对抗样本保存根目录")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h", help="预训练模型名称或路径")
    parser.add_argument("--iterations", type=int, default=100, help="最大迭代次数")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--c", type=float, default=1.0, help="loss 中 margin 权重 c")
    parser.add_argument("--alpha", type=float, default=1.0, help="recognition_score alpha")
    parser.add_argument("--k", type=float, default=1.0, help="step_function k")
    parser.add_argument("--sr", type=int, default=16000, help="采样率，用于保存音频")
    args = parser.parse_args()
    # 2. 选择运行设备：GPU 优先，若不可用则使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 3. 加载预训练模型和 processor，并设置为推理模式
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    # 4. 初始化统计指标
    total = 0
    success_count = 0
    fail_count = 0
    sum_iters = 0
    sum_snr = 0.0

    # 5. 递归遍历 input_dir 中的所有音频文件并保持目录结构
    for root, _, files in os.walk(args.input_dir):
        for fname in tqdm(files, desc="攻击中"):  # 进度条
            if not (fname.endswith(".wav") or fname.endswith(".flac")):
                continue
            total += 1
            in_path = os.path.join(root, fname)
            # 构造相对路径
            rel_path = os.path.relpath(in_path, args.input_dir)
            out_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 5.1 加载音频并使用 CTC greedy 对齐获取原始 logits 与 labels
            x_orig = load_audio(in_path).to(device)
            logits_orig, y_f = get_alignment(x_orig, processor, model, device)
            orig_ids = torch.argmax(logits_orig, dim=-1)
            orig_text = processor.batch_decode([orig_ids])[0].upper().strip()

            # 5.2 对单样本执行攻击
            success, iters, snr, x_adv = attack_single(
                x_orig, y_f, orig_text, processor, model, device, args
            )
            # 保存对抗音频
            save_audio(out_path, x_adv, sr=args.sr)

            # 5.3 更新全局统计结果
            if success:
                success_count += 1
                sum_iters += iters
                sum_snr += snr
            else:
                fail_count += 1

    # 6. 计算并打印批量攻击的汇总统计
    avg_iters = sum_iters / success_count if success_count > 0 else float('nan')
    avg_snr = sum_snr / success_count if success_count > 0 else float('nan')
    print("\n= 批量攻击汇总 =")
    print(f"总样本数: {total}")
    print(f"成功数: {success_count}")
    print(f"失败数: {fail_count}")
    print(f"成功率: {success_count/total*100:.2f}%")
    print(f"平均成功迭代轮次: {avg_iters:.2f}")
    print(f"平均 SNR(dB): {avg_snr:.2f}")


if __name__ == '__main__':
    main()
