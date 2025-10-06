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