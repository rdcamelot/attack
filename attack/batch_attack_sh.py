"""
这里每行命令末尾的空格不能省略, 要正确断行, 用反引号换行但不要把它留在参数值后面
否则就成为了参数值的一部分, 导致传给脚本的 --input_dir 值末尾多了一个 “\”
所以脚本去找的是一个并不存在的文件夹

```bash
python .\attack\batch_attack_sh.py `
  --input_dir .\data\LibriSpeech\test-clean\7729 `
  --model facebook/wav2vec2-base-960h `
  --iterations 100 `
  --lr 1e-2 `
  --c 1.0 `
  --alpha 1.0 `
  --k 1.0 `
  --sr 16000
```
"""
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
    攻击逻辑基本和 attack_sh.py 一致
    """
    # 初始化扰动 delta（可学习）并开启梯度追踪
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    # 使用 Adam 优化器只更新 delta
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    success = False
    iters = args.iterations

    for it in range(1, args.iterations + 1):
        # 构造对抗样本 x_adv，并裁剪到合法范围 [-1,1]
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0)
        # 前向计算 logits_adv: [T, C]
        logits_adv = model(x_adv.unsqueeze(0)).logits.squeeze(0)

        # 计算 attack loss，包括 L2、margin、confidence 和 step 项
        loss = attack_loss(logits_adv, y_f, delta, args.c, args.alpha, args.k)

        # 梯度清零、反向传播并更新 delta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 解码对抗样本文本并判断是否与原始转录不同
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
    parser = argparse.ArgumentParser(description="批量生成对抗语音样本")
    parser.add_argument("--input_dir", required=True, help="输入音频文件夹路径")
    parser.add_argument("--output_dir", help="输出对抗样本保存根目录，未指定则在当前目录新建 attack_<input_dir_basename> 文件夹")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h", help="预训练模型名称或路径")
    parser.add_argument("--iterations", type=int, default=100, help="最大迭代次数")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--c", type=float, default=1.0, help="loss 中 margin 权重 c")
    parser.add_argument("--alpha", type=float, default=1.0, help="recognition_score alpha")
    parser.add_argument("--k", type=float, default=1.0, help="step_function k")
    parser.add_argument("--sr", type=int, default=16000, help="采样率，用于保存音频")
    args = parser.parse_args()
    # 如果未指定输出目录，则根据输入目录名生成默认文件夹
    if not args.output_dir:
        base_name = os.path.basename(os.path.normpath(args.input_dir))
        args.output_dir = f"attack_{base_name}"
        print(f"[i] 未指定 --output_dir, 默认使用: {args.output_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()

    # 收集所有待攻击文件
    file_list = []
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if fname.endswith('.wav') or fname.endswith('.flac'):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(in_path, args.input_dir)
                file_list.append((in_path, rel_path))
    total = len(file_list)
    success_count = 0
    fail_count = 0
    sum_iters = 0
    sum_snr = 0.0
    failed_files = []

    # 批量攻击主循环，单个 tqdm 进度条
    for in_path, rel_path in tqdm(file_list, desc="攻击中"):
        out_path = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # 加载音频并对齐
        x_orig = load_audio(in_path).to(device)
        logits_orig, y_f = get_alignment(x_orig, processor, model, device)
        orig_ids = torch.argmax(logits_orig, dim=-1)
        orig_text = processor.batch_decode([orig_ids])[0].upper().strip()
        # 对单样本执行攻击
        success, iters, snr, x_adv = attack_single(
            x_orig, y_f, orig_text, processor, model, device, args
        )
        # 保存对抗音频
        save_audio(out_path, x_adv, sr=args.sr)
        # 更新统计并记录失败文件
        if success:
            success_count += 1
            sum_iters += iters
            sum_snr += snr
        else:
            fail_count += 1
            failed_files.append(rel_path)

    # 如果没有处理到任何文件，提前退出避免除零
    if total == 0:
        print(f"[Error] 在路径 {args.input_dir} 下未找到任何 .wav/.flac 文件。请检查路径是否正确。")
        return
    # 计算并打印批量攻击的汇总统计
    avg_iters = sum_iters / success_count if success_count > 0 else float('nan')
    avg_snr = sum_snr / success_count if success_count > 0 else float('nan')
    print("\n= 批量攻击汇总 =")
    print(f"总样本数:           {total}")
    print(f"成功数:             {success_count}")
    print(f"失败数:             {fail_count}")
    print(f"成功率:             {success_count/total*100:.2f}%")
    print(f"平均成功迭代轮次:   {avg_iters:.2f}")
    print(f"平均 SNR(dB):       {avg_snr:.2f}")
    # 列出攻击失败的文件
    if failed_files:
        print("\n以下文件攻击失败:")
        for f in failed_files:
            print(f"  {f}")


if __name__ == '__main__':
    main()
