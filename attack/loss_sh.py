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


def attack_loss(logits: torch.Tensor,
                labels: torch.Tensor,
                delta: torch.Tensor,
                c: float,
                alpha: float = 1.0,
                k: float = 1.0) -> torch.Tensor:
    """
    组合最终损失: L = ||δ||_2^2 + c * (S * H * L_margin)
    logits: [T, C], labels: [T], delta: [L]
    c: 权重系数
    """
    # 1. L2 正则项: 对扰动 delta 的所有元素求二范数平方
    l2 = torch.norm(delta.view(-1), p=2) ** 2
    # 2. 逐帧 margin loss
    lm = margin_loss(logits, labels)
    # 3. 识别置信度分数 S
    s = recognition_score(logits, labels, alpha)
    # 4. Step-like 函数近似 H
    h = step_function(logits, labels, k)
    # 5. 组合最终损失: L = ||delta||^2 + c * (S * H * L_margin)
    loss = l2 + c * (s * h * lm)
    return loss
