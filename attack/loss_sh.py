"""
loss_sh.py

定义序列级攻击损失函数，包括：
1. margin loss(逐帧边缘损失)
2. recognition score(S)
3. step-like function(H)
4. 组合最终损失 L = ||δ||_2^2 + c * (S * H * L_margin)
"""
import torch
import torch.nn.functional as F


def margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算 margin loss：
    d_f = z_f[y_f] - max_{j≠y_f} z_f[j]
    L_margin = sum(max(d_f, 0))
    logits: [T, C]
    labels: [T]
    返回标量
    """
    # 取每帧对应正确标签的 logit
    true_logits = logits[torch.arange(logits.size(0)), labels]
    # 将正确标签位置置为 -inf，方便取最大值
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), labels] = float('-inf')
    # 取每帧除正确标签外的最大 logit
    other_max, _ = masked.max(dim=1)
    # 计算 d_f
    d = true_logits - other_max
    # L_margin
    loss_margin = F.relu(d).sum()
    return loss_margin


def recognition_score(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    计算识别置信度分数 S = exp(alpha * sum_f log softmax(z_f)[y_f])
    logits: [T, C]
    labels: [T]
    """
    log_probs = F.log_softmax(logits, dim=1)
    sel = log_probs[torch.arange(log_probs.size(0)), labels].sum()
    score = torch.exp(alpha * sel)
    return score


def step_function(logits: torch.Tensor, labels: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    用 sigmoid(k * min_f d_f) 近似 step 函数
    d_f 同 margin 定义
    """
    # 复用 margin_loss 计算 d_f 向量
    true_logits = logits[torch.arange(logits.size(0)), labels]
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), labels] = float('-inf')
    other_max, _ = masked.max(dim=1)
    d = true_logits - other_max
    min_d = d.min()
    h = torch.sigmoid(k * min_d)
    return h


def attack_loss(logits: torch.Tensor,
                labels: torch.Tensor,
                delta: torch.Tensor,
                c: float,
                alpha: float = 1.0,
                k: float = 1.0) -> torch.Tensor:
    """
    组合最终损失：L = ||δ||_2^2 + c * (S * H * L_margin)
    logits: [T, C], labels: [T], delta: [L]
    c: 权重系数
    """
    # L2 term
    l2 = torch.norm(delta.view(-1), p=2) ** 2
    # sequence losses
    lm = margin_loss(logits, labels)
    s = recognition_score(logits, labels, alpha)
    h = step_function(logits, labels, k)
    loss = l2 + c * (s * h * lm)
    return loss
