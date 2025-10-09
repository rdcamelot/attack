"""
loss_test.py

当前损失函数设计：
    L_total = L2 + λ_ctc * CTC_NLL + λ_run * (S * H * run_margin)

各项含义：
 1. L2: ||δ||₂² 二范数正则，最小化扰动能量
 2. CTC_NLL: 对原始序列的 CTC 负对数似然，降低对原始转录的整体置信度
 3. run_margin: run-level margin，通过对连续同一符号区段（RLE）计算 hinge margin，保证符号级打破
 4. S (recognition_score): 全局置信度评分 exp(α·∑_f log p[y_f])，用于缩放 run_margin
 5. H (step_function): 平滑开关 sigmoid(k·min_f d_f)，基于最弱帧判断何时停止增大惩罚

设计动机与区别：
 - 原始损失只基于单帧 margin+S*H，对 CTC collapse 后的逻辑不敏感
 - 引入 CTC_NLL 提供全局序列级目标，直接影响转录概率
 - run_margin 针对 collapse 后的符号区段聚焦，使破坏某个符号 run 更可能改变最终输出
 - 通过混合这三部分，可提高攻击成功率并兼顾扰动最小化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算 margin loss:
    d_f = z_f[y_f] - max_{j≠y_f} z_f[j]
    L_margin = sum(max(d_f, 0))
    logits: [T, C]
    labels: [T]
    返回标量
    """
    # logits: 张量形状为 [T, C]，T 为时间步数，C 为类别数
    # 这里相当于在逐帧执行 true_logits[f] = logits[f, labels[f]]
    true_logits = logits[torch.arange(logits.size(0)), labels]
    
    # 将正确标签位置替换为 -inf，以排除在最大值计算中
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), labels] = float('-inf')

    # 对剩余类别取最大 logit，得到 other_max
    other_max, _ = masked.max(dim = 1)

    # 计算差值 d_f = z_f[y_f] - max_{j≠y_f} z_f[j]
    d = true_logits - other_max

    # 对每个差值取 ReLU 后累加，得到 margin loss
    loss_margin = F.relu(d).sum()
    return loss_margin

"""
度量模型对原始对齐路径的整体置信度,将其作为 margin loss 的放大系数
"""
def recognition_score(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    计算识别置信度分数 S = exp(alpha * sum_f log softmax(z_f)[y_f])
    logits: [T, C]
    labels: [T]
    """
    # logits: [T, C] -> 先计算每帧的 log softmax 概率
    log_probs = F.log_softmax(logits, dim=1)

    # 选取正确标签位置的 log_prob 并累加，得到 sum_f log p[y_f]
    # 在对数域上计算具有更好的数值稳定性
    sel = log_probs[torch.arange(log_probs.size(0)), labels].sum()

    # 计算识别置信度分数 S = exp(alpha * sum_f log_prob)
    score = torch.exp(alpha * sel)
    return score

"""
节约扰动的核心: H
取其中的最弱帧, 如果其等于 0, 说明已经有一帧被攻击成功了, 可以降低权重
但是在语音识别中可能没法直接使用, 因为在使用 CTC beam search 的时候, 最弱帧可能并不等同于一个符号被删除或替换
"""
def step_function(logits: torch.Tensor, labels: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    用 sigmoid(k * min_f d_f) 近似 step 函数
    d_f 同 margin 定义
    """
    # 同样获取正确标签 logit，计算其他类别的最大 logit
    true_logits = logits[torch.arange(logits.size(0)), labels]
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), labels] = float('-inf')
    other_max, _ = masked.max(dim=1)

    # 计算每帧差值 d_f
    d = true_logits - other_max

    # 取所有帧中最小的 d_f，用于近似最弱帧的判断
    min_d = d.min()

    # 以 sigmoid(k * min_d) 近似 step 函数输出 H
    h = torch.sigmoid(k * min_d)
    return h

# 全局 CTC 损失，用于序列级对抗目标
blank_id = 0  # 根据 tokenizer 确定 blank token id
ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

def collapse_runs(labels: torch.Tensor) -> list:
    """
    对 labels (长度 T) 做运行长度编码 (RLE)，返回 [(token, start, end), ...]
    """
    runs = []
    start = 0
    T = labels.size(0)
    for i in range(1, T+1):
        if i == T or labels[i] != labels[start]:
            runs.append((labels[start].item(), start, i-1))
            start = i
    return runs

def run_margin_loss(logits: torch.Tensor, labels: torch.Tensor, num_runs: int = None) -> torch.Tensor:
    """
    基于 collapse 后的 runs 计算 margin:
    对每个 run 随机或按需选取，累加 run 内每帧 hinge(z[token] - z[alt]).
    """
    runs = collapse_runs(labels)
    if num_runs and len(runs) > num_runs:
        runs = random.sample(runs, num_runs)
    Lr = torch.tensor(0.0, device=logits.device)
    for token, s, e in runs:
        avg_logits = logits[s:e+1].mean(dim=0)
        avg_logits[token] = -1e9
        avg_logits[blank_id] = -1e9
        alt = torch.argmax(avg_logits).item()
        for f in range(s, e+1):
            Lr += F.relu(logits[f, token] - logits[f, alt])
    return Lr


def attack_loss(logits: torch.Tensor,
                labels: torch.Tensor,
                delta: torch.Tensor,
                alpha: float = 1.0,
                k: float = 1.0,
                lambda_ctc: float = 1.0,
                lambda_run: float = 0.5) -> torch.Tensor:
        """
        综合对抗损失:
            L2_loss + lambda_ctc*CTC_NLL + lambda_run*(S*H*run_margin)
        """
        # === Step1: L2 范数正则化，最小化扰动能量 ||δ||₂² ===
        l2 = torch.norm(delta.view(-1), p=2) ** 2
            
        # === Step2: 计算全局置信度 S 和平滑开关 H，用于缩放 run_margin ===
        # S = exp(α * ∑_f log softmax(logits[f])[labels[f]])
        S = recognition_score(logits, labels, alpha)
        # H = sigmoid(k * min_f (logits[f, y_f] - max_{j≠y_f} logits[f, j]))
        H = step_function(logits, labels, k)
        # CTC 序列级 NLL: 降低对原始标签序列的整体置信度
        log_probs = F.log_softmax(logits, dim=-1)
        T, C = logits.shape
        input_lengths = torch.full((1,), T, dtype=torch.long, device=logits.device)
        target = labels.unsqueeze(0)
        target_lengths = torch.full((1,), labels.numel(), dtype=torch.long, device=logits.device)
        ctc_nll = ctc_loss_fn(
            log_probs.unsqueeze(1),  # [T, 1, C]
            target,
            input_lengths,
            target_lengths
        )
        # === Step4: run-level margin，通过 RLE 对每个符号区段计算 hinge margin ===
        # 使用 S 和 H 进行缩放，重点惩罚尚未被成功攻击的区段
        run_term = S * H * run_margin_loss(logits, labels, num_runs=3)
        # 总 loss
        return l2 + lambda_ctc * ctc_nll + lambda_run * run_term