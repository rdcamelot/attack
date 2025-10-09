# ASR Untargeted Sequence-Level Attack

实现了一种“节省扰动的序列级攻击”方法，将图像领域的思路迁移到 CTC 基于 wav2vec2 的语音识别模型上。

## 文件结构

```
attack/
├── attack_sh.py      # 攻击主脚本
├── loss_sh.py        # 损失函数模块：margin, recognition score, step function
├── utils_sh.py       # 辅助函数：音频读写, 对齐, SNR, 成功判定
└── README_attack.md  # 使用说明
```

## 环境依赖

```bash
pip install torch transformers soundfile pyctcdecode
```  
建议在 GPU 环境下运行以加速迭代。

## 使用方法

```bash
# 假设你在项目根目录，脚本位于 attack/ 下，音频位于同级根目录：
# 方式一：直接调用脚本文件（无需切换目录）
python attack/attack_sh.py \
  --input original.wav   # 原始音频在根目录
  --output attack/attack_original.wav  # 对抗样本保存到 attack/ 目录，如果不指定，自动在同目录下生成 attack_original.wav
  --model facebook/wav2vec2-base-960h \
  --iterations 100 \
  --lr 1e-2 \
  --c 1.0 \
  --alpha 1.0 \
  --k 1.0

# 方式二：先进入 attack/ 目录再运行（输入路径需回溯到上级）
cd attack
python attack_sh.py \
  --input ../original.wav \
  --output attack_original.wav \ 
  --model facebook/wav2vec2-base-960h \
  --iterations 100 \
  --lr 1e-2 \
  --c 1.0 \
  --alpha 1.0 \
  --k 1.0
```

```bash
python .\attack\attack_sh.py `
  --input .\1089-134686-0000.flac `
  --iterations 100 `
  --lr 1e-2 `
  --c 1.0 `
  --alpha 1.0 `
  --k 1.0
```

- `--input`: 原始干净音频（.wav/.flac）路径
- `--output`: 保存对抗样本
- `--model`: 预训练模型名称或路径
- `--iterations`: 最大迭代次数
- `--lr`: Adam 学习率
- `--c`, `--alpha`, `--k`: 损失函数中的权重参数

## 评估指标

- **Attack Success**: Untargeted 攻击，只要任何一帧 token 预测与原始对齐结果不同即视为成功。  
- **SNR (dB)**: 信噪比，衡量扰动强度，越大越 imperceptible。

在脚本结束时会打印：
```
Original transcription: ...
Adversarial transcription: ...
SNR(dB): XX.XX
Attack Succeeded/Failed
```

## 实现细节

1. **对齐**: 使用 CTC 贪心对齐（`utils_sh.get_alignment`），获得每帧 logits 与标签。
2. **损失**: `loss_sh.attack_loss`:
   - L2 范数约束 (`||δ||_2^2`)
   - Margin loss: d_f = z_f[y_f] - max_{j≠y_f} z_f[j]，仅对 d_f>0 的部分求和。
   - Recognition score S: exp(α * Σ_f log softmax(z_f)[y_f])
   - Step-like H: sigmoid(k * min_f d_f)
   - 最终 L = L2 + c * (S * H * L_margin)
3. **优化**: 直接对声音样本增量 δ 优化，x' = clamp(x+δ,-1,1)。

## 播放音频

对抗样本生成后，可用任意音频播放器播放 `adversarial.wav`，或使用 Python 脚本：
```python
import soundfile as sf, sounddevice as sd
adv, sr = sf.read('adversarial.wav')
sd.play(adv, sr)
```  

---  
代码已注释说明每个部分功能，欢迎复现、对比和扩展研究。