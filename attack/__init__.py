"""
__init__.py 之所以是空的，是因为它的唯一目的是告诉 Python：「这个目录是一个 package，可以被 import」。它本身不需要写任何代码，只要存在，Python 在加载时就会把它当作可导入模块的标志。

为什么要加 __init__.py

Python 在寻找 import attack.xxx 时，会把含有 __init__.py 的文件夹当成一个顶级 package。
如果没有这个文件，import attack.loss_sh 或 from attack import utils_sh 都会找不到模块。
之前不能运行的原因

你的脚本放在 attack_sh.py 里，当直接 python [attack_sh.py](http://_vscodecontentref_/2) … 或从子目录 attack 里运行时，Python 并不知道上一级目录的 attack 文件夹是个 package，也就 import 不到 attack.loss_sh，报 ModuleNotFoundError: No module named 'attack'。
在没有 __init__.py 的情况下，目录就只是普通文件夹，Python 搜索路径里找不到任何可导入的包。
解决方案

在 attack 目录下，新建一个空的 __init__.py，让该目录成为 package。
同时在 attack_sh.py 中改用相对导入（from .loss_sh import …），并在项目根目录用 python -m attack.attack_sh … 启动脚本，这样 Python 就能正确识别并加载子模块了。

"""