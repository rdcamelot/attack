# 导入错误修复说明

## 问题描述

在 Windows 终端直接运行脚本时出现：
```
ImportError: attempted relative import with no known parent package
```  
或
```
ModuleNotFoundError: No module named 'attack'
```

这是由于 Python 对模块搜索和相对导入机制导致的：当你用 `python attack/attack_sh.py` 或在子目录里直接运行 `attack_sh.py`，脚本目录会被当作最顶层模块搜索路径，**Python 并不会自动把脚本的上一级目录视作包**，因此相对导入 `from .loss_sh import ...` 或绝对导入 `import attack.loss_sh` 都会失败。

## 生成原因

1. **缺少 `__init__.py`**  
   - 最初 `attack/` 目录没有 `__init__.py`，Python 无法将其识别为可导入的 package。  

2. **直接执行脚本**  
   - `python attack/attack_sh.py` 会将 `attack/attack` 目录加入 `sys.path`，而不是项目根目录，导致父包不存在。  

3. **相对导入 vs. 绝对导入**  
   - 相对导入（`from .loss_sh import ...`）要求模块必须是以 package 形式加载；绝对导入（`from attack.loss_sh import ...`）要求 package 在 `sys.path` 中。

## 修复方案

1. **声明 Package**  
   在 `attack/` 目录下添加空文件 `__init__.py`：
   ```bash
   touch attack/__init__.py
   ```  

2. **改用绝对导入并设置搜索路径**  
   在 `attack_sh.py` 顶部插入：
   ```python
   import os, sys
   # 将项目根目录加入模块搜索路径
   SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
   PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
   sys.path.insert(0, PROJECT_ROOT)
   
   from attack.loss_sh import attack_loss
   from attack.utils_sh import load_audio, ...
   ```  
   这样无论你如何运行脚本，Python 都能找到顶层 `attack` 包。  

3. **使用 `-m` 模式加载**（推荐）  
   在项目根目录运行：
   ```bash
   python -m attack.attack_sh --input ... --iterations ...
   ```  
   `-m` 会将项目根目录自动加入搜索路径，并按包方式加载模块，支持相对导入。

## 验证运行

在项目根目录执行以下命令，确保能够正常启动并加载依赖：
```bash
python attack/attack_sh.py \
  --input path/to/audio.flac \
  --iterations 100 \
  --lr 1e-2 \
  --c 1.0 \
  --alpha 1.0 \
  --k 1.0
```
或：
```bash
python -m attack.attack_sh --input path/to/audio.flac --iterations 100 --lr 1e-2 --c 1.0 --alpha 1.0 --k 1.0
```

修复完成后，脚本即能直接运行，无需额外设置 `PYTHONPATH`。


问题原因：  
使用 `-m .\attack\attack_sh.py` 时，Python 认为这是一个“相对模块名”（以点开头），但 `-m` 只接受“包.模块”的形式，不支持文件路径或以 `.` 开头的相对名称。

正确做法有两种：

1. 直接调用脚本（已在开头加了 `sys.path` 修复）  
   在项目根目录下运行：  
   ```powershell
   python .\attack\attack_sh.py --input .\7729-102255-0000.flac --iterations 1000 --lr 1e-2 --c 1.0 --alpha 1.0 --k 1.0
   ```

2. 以模块方式执行（推荐）  
   先确保当前工作目录是项目根（与 attack 同级）：  
   ```powershell
   cd D:\code\python\attack
   python -m attack.attack_sh --input 7729-102255-0000.flac --iterations 1000 --lr 1e-2 --c 1.0 --alpha 1.0 --k 1.0
   ```  
   注意：
   - `-m attack.attack_sh` 中不要带 attack、不要用文件路径  
   - 参数 `--input` 直接给文件名（或相对路径）即可  

这样就不会再报 “Relative module names not supported” 错误了。


---


**从调试输出看**，梯度一直为 0，说明你的攻击损失对 δ 根本没有梯度信号。原因在于：

1. Margin/Score/Step 这三项都是基于 logits 计算的，而 logits 是通过  
   ```python
   inputs = processor(x_adv, …)
   logits = model(inputs).logits
   ```  
   得到的。可惜 Hugging Face 的 `processor.feature_extractor` 在内部用的是 numpy（或非 PyTorch）操作来做归一化、分帧等预处理，这部分被包裹在 `torch.no_grad()` 或干脆不支持反向传播，一旦你把 waveform 交给 `processor`，后面的梯度链就被切断了。

2. 你在迭代里又改用 `x_adv.unsqueeze(0)` 直接送给 model，这确实绕过了 processor，但 Wav2Vec2 模型本身也要先对 waveform 做 conv‐feature 抽取，如果你在脚本里把它放到 `model.eval()` 并且没有显式开启 `model.feature_extractor` 的反向传播（它默认不 require_grad），那么特征提取层也不起梯度。

简单说：**攻击失效的核心是“预处理 + 模型特征抽取”两个环节都没有给 δ 留下反向通道，所以梯度始终为零。**

修复思路有两条路径（任选其一或合并）：

1. 让预处理／特征抽取全程在 PyTorch 中支持梯度  
   - 不调用 `processor.feature_extractor` 的 numpy 逻辑，而是直接把 `x_adv` 送给模型内部的 `feature_extractor`（它是 `nn.Conv1d` 系列），并且把这些层都设置 `requires_grad=True`  
   - 示例：  
     ```python
     # 关闭外部 processor
     input_values = x_adv.unsqueeze(0)  # [1,L]
     # 确保 conv feature extractor 支持梯度
     for p in model.feature_extractor.parameters():
         p.requires_grad = False  # 冻结权重，但 conv 层 forward 会保留梯度到输入
     logits = model(input_values).logits
     ```
   - 这样就能把 loss 的梯度真正传到 `input_values`（也就是 x_adv），再流到 δ。

2. 自己实现一套可微分的预处理  
   - 用 PyTorch 实现 waveform normalization、padding、特征抽取（可参考源码里的卷积结构）  
   - 把这段代码写到脚本里，确保全程在 PyTorch 计算图里，不再用 numpy

实际最省事的是方案 1：把处理直接交给模型的 `feature_extractor`（它是 Conv1d），并保证 `input_values` 的反向通道没被任何 `no_grad` 或外部 numpy 层截断。  

请在你的主循环里删除所有对 `processor` 的调用，只保留直接 `input_values = x_adv.unsqueaudio_adversarial_examplesze(0)`，并检查 `model.feature_extractor` 下的参数：

```python
# 在加载模型后：
for module in model.feature_extractor.modules():
    if isinstance(module, torch.nn.Conv1d):
        module.weight.requires_grad = False  # 冻结权重
        module.bias.requires_grad = False
# 这时 Conv1d 层的 forward 会把 grad 传回 input_values
```

然后再跑一遍 debug，你应该会看到 `delta.grad.norm` 从 0 变成非零，再接着就能更新 δ 了。  

这里先采用 方案1 ，绕过 HF 的 numpy 预处理，直接让音频走到模型内部的 PyTorch conv 特征提取层，从而保证这条路径对输入保持可导

1. 在加载模型后冻结 conv 层参数  
   ```python
   # model = Wav2Vec2ForCTC.from_pretrained(…)
   for p in model.feature_extractor.parameters():
       p.requires_grad = False
   ```
   这样做不会改变模型输出，但能保证它的前几层（Conv1d）在 forward 时，会把梯度传递到它们的输入张量上。
   `requires_grad = False` 只表示不需要给它自己算梯度，并不会阻断梯度从输出传到它的输入
   因此这里相当于冻结卷积核，不让它们被更新，但是可以计算音频输入的梯度

2. 攻击循环中不再调用 processor，直接做  
   ```python
   x_adv = x_orig + delta           # delta.requires_grad=True
   input_values = x_adv.unsqueeze(0)  # [1, L]
   logits = model(input_values).logits
   ```
   因为 Conv1d 的实现是纯 PyTorch，当你对 `logits` 反向时，梯度会一路从 logits→卷积→`input_values`→`delta`。

3. 这样 `loss.backward()` 就能在 `delta.grad` 上看到非零值，后续 `optimizer.step()` 才能更新扰动。

为什么成立？  
- HF 的 `Wav2Vec2Processor` 内部归一化、分帧等都是 numpy 操作或包在 `no_grad` 里，断开了计算图；  
- 而模型自带的 `feature_extractor` 是一组 `nn.Conv1d`，对其输入默认保留梯度；只要不在它外面再做 `torch.no_grad()`，就能把梯度“传回”到输入张量。

在之前 audio_adversarial_examples 的实现里，他们也是**自己用 PyTorch（或 torchaudio）实现完整的特征抽取流程**，不落回外部 numpy，这样才能完整保持从对抗损失到原始 waveform 的计算图，才能在攻击时拿到梯度并更新输入。

原先使用 `processor`，内部会把 waveform 变成 float、做归一化、分帧、拼 batch，再转给模型。
这些步骤往往是 numpy 或包在 torch.no_grad() 里，断开了对原始 waveform（δ）的计算图，梯度无法回传到 δ。


