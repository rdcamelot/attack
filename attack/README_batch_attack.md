# 批量对抗攻击脚本说明

本文档介绍如何使用 `batch_attack_sh.py` 在 LibriSpeech 等数据集目录下批量生成对抗语音样本，并统计实验指标。

## 文件说明

```
attack/
├── batch_attack_sh.py       # 批量攻击脚本
├── attack_sh.py             # 单样本攻击脚本
├── loss_sh.py               # 损失函数模块
├── utils_sh.py              # 工具函数
└── README_batch_attack.md   # 本文档
```

## 环境依赖

```bash
pip install torch transformers soundfile pyctcdecode tqdm
```

## 功能简介

- 递归遍历 `--input_dir` 下的所有 `.wav` 和 `.flac` 文件，保持目录结构地对每个文件执行 untargeted 序列级攻击。  
- 将对抗样本保存到 `--output_dir` 下对应位置。  
- 统计并输出以下指标：  
  - 总样本数  
  - 成功数 / 失败数  
  - 攻击成功率  
  - 平均成功迭代轮次（仅对成功样本）  
  - 平均 SNR(dB)（仅对成功样本）  

## 使用方法

```bash
# 方式一：直接从项目根目录运行 (PowerShell 风格)
python .\attack\batch_attack_sh.py `
  --input_dir .\data\LibriSpeech\test-clean\7729 `
  --output_dir .\attack_results\attack_7729 `
  --model facebook/wav2vec2-base-960h `
  --iterations 100 `
  --lr 1e-2 `
  --c 1.0 `
  --alpha 1.0 `
  --k 1.0 `
  --sr 16000
```

```bash
# 方式二：先进入 attack 目录再运行 (PowerShell 风格)
cd attack
python .\batch_attack_sh.py `
  --input_dir ..\data\LibriSpeech\test-clean\7729 `
  --output_dir ..\attack_results\attack_7729 `
  --model facebook/wav2vec2-base-960h `
  --iterations 100 `
  --lr 1e-2 `
  --c 1.0 `
  --alpha 1.0 `
  --k 1.0 `
  --sr 16000
```
- `--input_dir`: 包含待攻击音频的根目录（支持递归子目录）。  
- `--output_dir`: 对抗样本保存根目录，脚本会自动创建子目录。  
- 其他参数与 `attack_sh.py` 一致，用于调整优化超参。

未指定输出目录
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

脚本执行完毕后，会在终端打印：  
```
= 批量攻击汇总 =  
总样本数: 10  
成功数: 8  
失败数: 2  
成功率: 80.00%  
平均成功迭代轮次: 23.50  
平均 SNR(dB): 25.32  
```  

对所有成功样本，对抗音频会保存在 `output_dir` 下，与 `input_dir` 的文件结构保持一致。
