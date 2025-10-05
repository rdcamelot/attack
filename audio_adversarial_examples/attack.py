## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

"""
负责把 DeepSpeech logits/梯度接入后端并迭代优化音频，生成指定目标转录的对抗样本：

解析必要参数(原始音频、目标文本、学习率、迭代次数、checkpoint 等)。
Attack 类在 TF1 会话中创建扰动变量、掩码、目标标签，占位符经 get_logits 构建 DeepSpeech 前向图并恢复权重。
选择 CTC 损失, 按梯度符号更新扰动(Adam)，每轮可选 MP3 模式并周期性用 beam search 检查是否已识别成目标短语。
保存最佳扰动音频、噪声统计和命中轮次，同时导出 tmp/attack-时间戳.csv。
整体流程：加载原音 → 在 Bound & Mask 限制下优化扰动 → 得到满足目标转录的对抗音频。
"""

import numpy as np  # 数值运算与数组处理
import tensorflow as tf  # 构建并执行 DeepSpeech 前向图，获取梯度+
import scipy.io.wavfile as wav  # 读取与写入 WAV 音频
import struct  # 用于按字节解包音频数据（MP3 转换时使用）
import pandas as pd  # 整理指标并输出 CSV 报告
import time  # 记录运行时间与生成时间戳
import sys  # 调整模块搜索路径，导入 DeepSpeech 源码
from collections import namedtuple  # 保留原仓库需求（当前版本未使用）

sys.path.append("DeepSpeech")  # 让 Python 能找到本地 DeepSpeech 训练代码
import DeepSpeech  # pylint: disable=unused-import  # 触发依赖初始化，虽未直接引用
try:
    import pydub  # MP3 模式需要用到的音频编解码库
except:  # noqa: E722 - 保留原项目的宽松捕获方式
    print("pydub was not loaded, MP3 compression will not work")

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse  # 将稠密标签转换成 CTC 期望的稀疏格式
from tf_logits import get_logits  # 本仓库封装的“波形 → DeepSpeech logits”函数

# 原本就是 DeepSpeech 单独拆出的 CTC 解码器组件，便于复用在不同语言模型或外部项目中，所以以独立 Python 包形式发布；
# 虽然由 DeepSpeech 官方维护，但并未直接打包在 DeepSpeech 模块内部
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer  # DeepSpeech 的 CTC beam search 解码器与语言模型打分器

create_flags()

# 会把 DeepSpeech 训练脚本中定义的命令行参数（如 --scorer_path、--beam_width、--lm_alpha 等）注册到 absl.flags; 
# FLAGS 是该框架的全局配置对象，读写这些参数都通过它。攻击代码复用这套定义，省得重写一遍并保持与 DeepSpeech 原始配置同步。
# 如果换成别的模型，需要换成那个模型自己的参数注册方式：有的提供类似的 flag/配置接口，就引用；没有就自己用 absl.flags 或 argparse 去定义必需的超参。
from deepspeech_training.util.flags import create_flags, FLAGS  # DeepSpeech 官方的 flag 定义，复用其超参配置

# 持有 DeepSpeech 的全局超参（采样率、alphabet、特征维度、RNN 层数等）
# 而 initialize_globals() 会读入这些设置并注册到 FLAGS/Config 里
# 确保后续调用 Config.alphabet、Config.audio_win_len、FLAGS.lm_alpha 等属性时与训练时保持一致；
# 攻击代码需要这些配置来构建前向图、做 CTC 解码和语言模型加载，所以运行主流程前必须先调用一次。
from deepspeech_training.util.config import Config, initialize_globals  # 读入全局配置（采样率、alphabet 等）并完成初始化
import absl.flags  # 命令行参数解析库

# 定义命令行参数（沿用 DeepSpeech 的 flag 系统，保持项目习惯）
# 用 absl.flags 注册命令行参数，结合 DeepSpeech 的配置来驱动攻击脚本：
# 既包含 DeepSpeech 推理所需的 restore_path、语言模型相关参数
# 也包含对抗优化本身的输入音频、目标文本、学习率、迭代次数、MP3 模式等设置。
f = absl.flags
f.DEFINE_multi_string('input', None, 'Input audio .wav file(s), at 16KHz (separated by spaces)')  # 待攻击的原始 WAV 音频，可一次传多段
f.DEFINE_multi_string('output', None, 'Path for the adversarial example(s)')  # 如提供，则逐个写入指定路径
f.DEFINE_multi_string('finetune', None, 'Initial .wav file(s) to use as a starting point')  # 可选“微调音频”，作为扰动初始值

f.DEFINE_string('outprefix', None, 'Prefix of path for adversarial examples')  # 未指定 output 时，使用前缀+索引命名
f.DEFINE_string('target', None, 'Target transcription')  # 期望模型输出的目标文本
f.DEFINE_integer('lr', 100, 'Learning rate for optimization')  # 对抗优化的学习率（Adam 见下文）
f.DEFINE_integer('iterations', 1000, 'Maximum number of iterations of gradient descent')  # 最大迭代轮次
f.DEFINE_float('l2penalty', float('inf'), 'Weight for l2 penalty on loss function')  # L2 正则项权重；默认无穷大表示忽略该项
f.DEFINE_boolean('mp3', False, 'Generate MP3 compression resistant adversarial examples')  # 是否启用 MP3 鲁棒模式
f.DEFINE_string('restore_path', None, 'Path to the DeepSpeech checkpoint (ending in best_dev-1466475)')  # DeepSpeech checkpoint 路径
f.DEFINE_string('lang', "en", 'Language of the input audio (English: en, German: de)')  # 语言标签，仅用于输出指标
# 在 absl.flags 中注册一个只允许出现一次的字符串参数，命令行通过 --name=value 传入；
# DEFINE_multi_string 注册可重复出现的字符串参数，可多次写 --name=value，最终在代码里通过 FLAGS.name 获取（多值对应列表）。


# 声明必填参数，缺失时 absl 会直接报错退出
f.mark_flag_as_required('input')
f.mark_flag_as_required('target')
f.mark_flag_as_required('restore_path')
    

# 不是为了满足 DeepSpeech 输入，是为了在“MP3 模式”下评估压缩鲁棒性：先把扰动波形写成 WAV，再经 MP3 编码/解码得到被压缩后的 PCM
# 随后用这份 PCM 送入 DeepSpeech 检查目标短语是否仍能被识别，从而确保生成的对抗样本在被 MP3 压缩后依旧有效。
def convert_mp3(new, lengths):
    """将当前扰动后的音频写成 WAV → 压缩成 MP3 → 再解码回 PCM, 用于评估 MP3 鲁棒性。"""

    # 只在需要 MP3 模式时才导入 pydub，避免模块被视为强依赖(即即使不用 MP3 功能也必须安装)
    import pydub  # 局部导入，避免未启用 MP3 模式时强依赖

    # 把对抗音频写入临时 WAV：保持 16kHz、16bit PCM，与 DeepSpeech 输入一致
    # clip 进行裁剪
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(new[0][:lengths[0]]),
                               -2**15, 2**15-1), dtype=np.int16))

    # 利用 pydub 先从 WAV 读入并导出 MP3，模拟压缩过程
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")

    # 再把 MP3 解码回 PCM，得到压缩后的波形（单位：int16）
    raw = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3ed = np.array([
        struct.unpack("<h", raw.raw_data[i:i + 2])[0]
        for i in range(0, len(raw.raw_data), 2)
    ])[np.newaxis, :lengths[0]]
    return mp3ed
    

# 采样点是指把连续音频在固定采样率下离散化得到的单个波形数值；例如 16 kHz 表示每秒有 16 000 个采样点。
class Attack:
    # sess: 现有 TensorFlow Session，对应 DeepSpeech 计算图的执行上下文。
    # loss_fn: 选择损失函数类型（当前支持 "CTC"）。
    # phrase_length: 目标短语最大长度（字符数），用于配置标签张量。
    # max_audio_len: 输入音频的采样点上限，用来申请扰动、掩码等变量。
    # learning_rate: Adam 优化器的学习率。
    # num_iterations: 最大迭代次数。
    # batch_size: 同时攻击的样本数。
    # mp3: 是否启用 MP3 鲁棒模式。
    # l2penalty: L2 正则权重（默认无穷大即忽略）。
    # restore_path: DeepSpeech checkpoint 的路径，用于恢复模型权重。
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                    learning_rate=10, num_iterations=5000, batch_size=1,
                    mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        建立攻击流程。这里会构建用于实际生成对抗样本的 TensorFlow 计算图。

        完成：预设超参、创建带 qq_ 前缀的扰动/掩码等变量、构造 new_input 并注入高斯噪声、调用 get_logits 搭建 DeepSpeech 前向图、用 Saver 载入 checkpoint 权重
        根据 loss_fn 配置 CTC 损失和可选 L2 项、初始化 Adam 及其新建状态、预备 softmax 概率张量，并在需要时构造语言模型评分器。
        这样既复用了训练模型的权重，又保证攻击所需的变量和梯度路径完整可用。
        """
        # Attack 对象持有 TensorFlow Session，并在其中构造对抗优化所需的全部变量与图结构
        print("\nInitializing attack..\n")
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        """
        创建攻击所需的全部变量，它们统一带上 qq_ 前缀，方便区分这些是我们新增的变量；
        这样在恢复 DeepSpeech checkpoint 时不会与原模型变量冲突或被覆盖。
        """

        # 所有新增变量都以 qq_ 前缀命名，避免与 DeepSpeech checkpoint 中的变量冲突

        # tf.Variable 是 TensorFlow 1.x 用于在计算图中创建可训练张量的接口；这里用零初始化创建 delta（扰动）、mask（长度掩码）等变量。
        # delta 会被优化器更新生成对抗噪声，mask 用于屏蔽 padding，使扰动只作用在真实采样点。
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        # mask 指示有效采样点（长度外的 padding 区域保持原值）
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')

        # CW 损失指 Carlini & Wagner 提出的对抗攻击目标函数
        # 核心是通过构造 margin-based 目标推动模型输出目标类（或远离原类），常写成 max(max_{j≠t}(Z(x)_j) - Z(x)_t, -κ) 等形式，以 logits 差距约束分类边界；
        # 在音频攻击中若启用 CW 损失，会直接最小化该 margin 以获得更稳定的定向攻击效果。
        # cwmask 指示目标短语长度，虽然当前未使用 CW 损失，但保留以兼容原始代码
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        # original 保存原始音频，供叠加扰动
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        # lengths 记录每条音频的帧长（后续供 DeepSpeech 使用）
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        # importance 是 Carlini&Wagner 攻击的默认配置，这里设置为全 1
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        # target_phrase、target_phrase_lengths 存放目标文本的索引序列及长度
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        # rescale 控制扰动幅度的缩放因子，迭代过程中会动态收缩
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')


        # Carlini&Wagner 风格的 l_inf 约束：扰动默认限制在 ±2000，并乘以 rescale 动态调整
        # 可以依据数据集来改变这个常数
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale

        # 扰动加上掩码，使得填充区的采样点保持恒为 0
        # new_input = 原音频 + 扰动；
        # pad 区域乘以 0，保持原有静音
        # 填充区并非真实信号；若扰动作用在这些 padding 上会浪费优化预算且影响梯度计算，所以用 mask 保证填充段始终为 0，只在有效采样点上注入噪声。
        self.new_input = new_input = self.apply_delta*mask + original

        # 量化截断指在把连续值映射到有限精度（如 16 位整数）时，超范围的数被裁到边界或多个值映射为同一整数，导致梯度在反向传播时变为 0；
        # 加入微小高斯噪声可缓和这一现象，让优化时仍能得到非零梯度。
        # 加入微小高斯噪声，避免量化截断导致梯度停滞
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # 调用封装好的 get_logits 构建 DeepSpeech 前向图
        # TensorFlow 会自动把这段运算加入计算图，虽然当下只拿 logits，但图中每个算子都保留了梯度信息；
        # 后续调用 optimizer.compute_gradients 时，TF 就沿着这条图反向传播求出对 delta 的梯度，无需手写。
        self.logits = logits = get_logits(pass_in, lengths)

        """
        先构图再恢复权重是 TF1 的常规流程: get_logits 会搭好节点但变量仍是默认值，之后 tf.train.Saver(...).restore(sess, restore_path) 才把 checkpoint 中保存的 DeepSpeech 原始参数（训练阶段学得的权重）写入图里。
        保留“原始权重”即排除带 qq_ 前缀的自建变量，只加载模型自身的参数，不覆盖攻击用变量。
        """

        # 加载 DeepSpeech checkpoint，将权重填充到图中（忽略 qq 前缀变量）
        # 这里用 tf.train.Saver([...]) 过滤掉前缀 qq 的变量，只保留 DeepSpeech 原始权重，然后执行 saver.restore(sess, restore_path) 把 checkpoint 中的参数加载进当前计算图。
        # 即便不再训练，也要把预训练权重写入图里，模型才能输出正确 logits。
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)
        # 已有模型权重不是攻击脚本自己生成的。它来自 DeepSpeech 官方训练好的模型
        # 需先从 Mozilla 提供的发布包下载并在命令行参数 --restore_path 里指定该权重路径，攻击流程再把它加载进计算图。

        # 选择损失
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            # CTC 损失需要稀疏标签，因此先把目标序列转成 sparse tensor
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # 可选的 L2 正则：默认设为 inf，即跳过该项以提升速度
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1) + l2penalty*ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
            
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        else:
            raise NotImplemented

        self.loss = loss
        self.ctcloss = ctcloss
        
        # 记录现有变量，方便后续仅初始化新增变量
        # 为了区分已有变量和后面的新建攻击变量，避免破坏已恢复的模型权重
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # 仅对 delta 求梯度，取符号进行更新（对应 FGSM 风格的 l_inf 约束）
        grad, var = optimizer.compute_gradients(self.loss, [delta])[0]
        # optimizer.apply_gradients([(tf.sign(grad), var)]) 则将梯度的符号（FGSM 风格的 L∞ 约束更新方向）应用到变量 delta 上，从而完成一次基于 Adam 的梯度下降步
        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])
        
        # 调用 optimizer.compute_gradients/apply_gradients 时，Adam 会为目标变量 delta 自动创建额外的优化器状态（如一阶/二阶动量槽变量等）
        # 这些就是新出现的 TensorFlow 全局变量，所以用 tf.global_variables() 再取一次即可捕捉到它们并单独初始化。
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        # 这是 TensorFlow 的变量初始化操作，调用 tf.variables_initializer 为前面新建的优化器状态变量和 delta 赋初值，确保它们在会话中可用后才开始迭代更新。
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # 预先把 logits 经 softmax 变成概率，以便周期性跑 beam search
        self.probs = tf.squeeze(tf.nn.softmax(self.logits, name='logits'))
        
        # 如提供语言模型 scorer 文件，则初始化带语言模型的解码器；否则仅按声学概率解码
        # 若命令行提供了语言模型评分器（--scorer_path）就构造 Scorer，让 CTC beam search 同时参考声学概率和语言模型；
        # 否则 self.scorer=None，解码仅依赖声学概率。最后打印提示初始化完成。
        if FLAGS.scorer_path:
            self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            self.scorer = None
        print("Initialization done.\n")


    def attack(self, audio, lengths, target, toks, finetune=None):
        # audio: 批量原始波形，每条已补零到相同长度。
        # lengths: 各条音频的真实采样点数，用于掩码和特征长度计算。
        # target: 目标短语的字符索引列表（按 DeepSpeech alphabet 编码）。
        # toks: 字符索引与字符之间的映射字符串 " abc...'", 便于从解码结果或目标标签的整数序列还原出可读文本。
        # finetune: 可选的初始对抗音频，用作扰动起点（默认从原音开始）。

        """执行多轮迭代优化，生成目标文本的对抗样本。"""
        """
        Attack.attack()先把当前批次的原始音频、长度与目标文本写入各自的 TF 变量，重置扰动 delta, 必要时用已有对抗样本初始化；
        随后进行最多 num_iterations 次循环：每 10 步取回 new_input/δ/logits 做 CTC 解码与调试输出（可选 MP3 复验）
        每步都 sess.run(self.train) 以 Adam 的梯度符号更新 delta, 再根据解码结果判断是否命中目标短语；
        命中后记录首/再次命中迭代，收紧 rescale 限制并保存当前最佳对抗音频。
        循环结束返回各样本的最终对抗波形及命中轮次，用于后续落盘与统计。
        """

        print("Start attack..\n")
        # self.sess 是传入的 TensorFlow Session 句柄，表示已载入 DeepSpeech 图与权重的执行上下文；
        # 保存的不是图本身，而是执图、初始化变量、运行节点时所需的会话环境。
        sess = self.sess

        # 每次攻击前重置所有工作变量,填充成当前批次的真实值,使后续迭代能正确计算损失和梯度。
        # “取出会话”这一步在 PyTorch 中可改成取出并使用模型/优化器对象，无需额外包装 Session
        # sess.run 来执行图
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        # DeepSpeech 的特征长度 = (样本数 - lookahead) // 320，此处转换成帧数量
        sess.run(self.lengths.assign((np.array(lengths)-(2*Config.audio_step_samples/3))//320))
        # mask: 真实采样点为 1，padding 区域为 0，避免扰动污染尾部的补零
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        # cwmask：基于帧数构建的掩码，保留原代码结构
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        # 右侧补 0，使目标序列长度统一，便于 dense → sparse 转换
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # final_deltas 用于记录每个样本当前最优的对抗音频
        final_deltas = [None]*self.batch_size

        # finetune 给出的是完整的候选对抗音频，想把它作为初始解，只需从中减去原始波形即可得到纯扰动 delta
        # 这样再加回 audio 时就恢复那段对抗音频，相当于从已有解出发继续微调。
        if finetune is not None and len(finetune) > 0:
            # 若提供“微调”音频，则把初始扰动设为 finetune - 原音，提高收敛速度
            # 从已有的对抗候选触发,加快收敛
            sess.run(self.delta.assign(finetune-audio))
        
        # 多轮梯度下降迭代
        #now = time.time()
        MAX = self.num_iterations  # 之前设定的最大迭代次数
        first_hits = np.zeros((self.batch_size,))
        best_hits = np.zeros((self.batch_size,))
        for i in range(MAX):
            # 10 轮进行一次输出
            if i % 10 == 0:
                # `sess.run(...)` 会在当前 TensorFlow Session 中执行计算图，按给定的张量列表返回对应的数值结果。
                # 这里一次性取出 `self.new_input`（加扰后的波形）、`self.delta`（当前扰动）
                # `self.probs`（softmax 后的概率矩阵）和 `self.logits`（模型原始输出），便于查看当前迭代的音频及模型响应。
                
                # “执行一次计算图”指调用 sess.run(...) 让 TensorFlow 在当前会话里对指定节点（张量或操作）做一次求值；
                # 它会算出这些节点依赖的所有运算，但不一定对应完整的一次迭代。(值计算和梯度更新是分开的,这里只是计算当前计算图中这些张量的数值)
                # 一次迭代通常是循环里先 sess.run 获取当前值，再 sess.run(self.train) 做一次更新。
                new, delta, probs_out, r_logits = sess.run((self.new_input, self.delta, self.probs, self.logits))

                # 元素为 (概率矩阵, logits) 的二元组；后续会遍历里面的每组输出，逐个执行 CTC 解码与评估
                lst = [(probs_out, r_logits)]
                # 这样一组是一个路径

                # 这段分支在启用 `--mp3` 时会先把当前对抗音频经 `convert_mp3` 重压缩，再用压缩后的波形跑一次 DeepSpeech 前向 (`sess.run` 取概率与 logits)
                # 做 CTC 解码并把结果加入 `lst`, 便于在同一次迭代里同时检查不同处理路径的识别情况。
                # 用于检测扰动是否能在 MP3 压缩后仍触发目标识别。
                # 当前版本直接抛出 `NotImplemented`，意味着 MP3 鲁棒性流程尚未完成。
                if self.mp3:
                    # TODO: Implement mp3 support 
                    raise NotImplemented("The current version does not support mp3 conversion.")
                    mp3ed = convert_mp3(new, lengths)
                    mp3_probs, mp3_logits = sess.run((self.probs, self.logits),
                                                   {self.new_input: mp3ed})
                    mp3_out = ctc_beam_search_decoder(mp3_probs, Config.alphabet, FLAGS.beam_width,
                                                scorer=self.scorer, cutoff_prob=FLAGS.cutoff_prob,
                                                cutoff_top_n=FLAGS.cutoff_top_n)
                    lst.append((mp3_out, mp3_logits))
                
                batch_size = r_logits.shape[1]
                for out, logits in lst:
                    out_list = []
                    for ii in range(batch_size):
                        # 取出对应样本的概率分布
                        if batch_size == 1:
                            probs = probs_out
                        else:
                            # 取出第 ii 个样本在所有时间步、全部字符类别上的 softmax 概率矩阵
                            # 随后把这份概率送入 CTC beam search 解码，就相当于对该样本的概率分布进行一次识别。
                            probs = probs_out[:,ii,:]
                        # 调用 CTC beam search 得到当前识别文本
                        decoded = ctc_beam_search_decoder(probs, Config.alphabet, FLAGS.beam_width,
                                                        scorer=self.scorer, cutoff_prob=FLAGS.cutoff_prob,
                                                        cutoff_top_n=FLAGS.cutoff_top_n)
                        print(decoded[0][1])
                        out_list.append(decoded)
                    # 额外输出 argmax 序列，便于调试
                    # 打印 argmax 序列只是调试用：它展示逐帧贪心解码的字符（不含语言模型、束搜索），可帮助快速对比 CTC beam search 结果、观察对齐情况或排查异常。
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-int(2*Config.audio_step_samples/3))//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))


            if self.mp3:
                new = sess.run(self.new_input)
                mp3ed = convert_mp3(new, lengths)
                feed_dict = {self.new_input: mp3ed}
            else:
                feed_dict = {}
                
            # 单步执行：获取当前 delta、损失、logits、新输入，并做一次梯度更新
            # 在 init 中定义了 self.train = optimizer.apply_gradients([(tf.sign(grad), var)])
            # 这里 sess.run(self.train) 就会执行 Adam 的一步更新，把 delta 沿梯度符号方向移动一个步长。
            # 其它张量（new_input、logits 等）都是计算图中以 delta 为输入的派生节点，它们会在下一次 sess.run 时根据更新后的 delta 重新计算，所以不需要单独更新变量。
            
            # 所以这里会对 delta 做一步梯度更新,其他张量只是顺带取回当前值
            d, el, cl, l, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss, self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    
            # 打印当前 batch 的 CTC 损失
            print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            for ii in range(self.batch_size):
                # 每 100 次迭代检查一次是否成功；如果成功（或已到最后一轮），就记录结果并调小 rescale 常量。
                # rescale 用于动态收缩扰动幅度：apply_delta = clip(delta, ±2000) * rescale。
                # 攻击成功后把它乘以 0.8（或调到当前幅度），等于逐步缩小 L∞ 边界，让噪声更小。

                # out_list[ii][0][1] == ... 判断解码后是否等于目标文本
                if (self.loss_fn == "CTC" and i%10 == 0 and out_list[ii][0][1] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    # 两者比较能判断“实际噪声是否已经远小于现有上限
                    # rescale[ii] * 2000就是当前样本允许的扰动幅度上限
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # 如果当前幅度已经低于阈值，就把阈值调到当前幅度，省去额外迭代时间。
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # 否则就按常数缩小：这个常数越接近 1，结果质量越好；常数越小，收敛越快，但音质更差。
                    rescale[ii] *= .8

                    # final_deltas 里保存当前最优的对抗音频
                    # 只在满足成功判定（或最后一轮仍未成功）时执行，因此保存的是当下判定为“可用”的对抗音频。
                    final_deltas[ii] = new_input[ii]

                    # 打印当前样本索引 ii、对应的 CTC 损失 cl[ii] 以及当前扰动上限 2000 * rescale[ii][0]，用于监控成功时的损失和约束范围。
                    print("Worked i=%d ctcloss=%f bound=%f"%(ii, cl[ii], 2000*rescale[ii][0]))
                    
                    # 记录首次/再次命中迭代用于评估攻击收敛速度与稳定性，便于后续统计对抗效率、比较不同参数或模型的表现
                    # 也能在生成多样本时挑选收敛更快或更稳定的结果。
                    if (first_hits[ii] == 0):
                        # first_hit 记录第一次成功的迭代步，用于分析收敛速度
                        print("First hit for audio {} at iteration {}".format(ii, i))
                        first_hits[ii]=i
                    else:
                        # best_hit 保存后续再次命中的迭代步
                        best_hits[ii]=i

                    # 这行调用 self.rescale.assign(rescale) 并 sess.run，把刚算出的列表（新的扰动上限）写回 rescale 变量
                    # 保证下次迭代按更新后的缩放因子约束扰动。
                    sess.run(self.rescale.assign(rescale))

                    # 便利起见，把当前成功样本写入 tmp/adv.wav，便于人工试听
                    wav.write("tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))
        
        return final_deltas, first_hits, best_hits  
    

def main(_):
    """
    读取并预处理输入音频(16kHz、单声道、记录长度与原始 dB), 对批次做尾部补零并计算 maxlen。
    构造 Attack 实例：在会话中搭建攻击所需变量、复用 get_logits 构建前向图并用 checkpoint 恢复预训练权重，初始化优化器与可选语言模型 scorer。
    调用 attack.attack(...)：把当前批次数据写入 TF 变量，迭代优化 delta(每步 sess.run(self.train))
        周期性解码检查是否已被识别为目标，命中时收紧 rescale、记录 first/best hit 并保存当前最佳对抗音频。
    将生成的对抗波形写盘（WAV/MP3），并计算/记录扰动指标（峰值、最小非零改动、相对 dB 等）。
    汇总结果到 DataFrame/CSV（包含文件名、长度、运行时间、噪声响度、first/best hit 等），便于统计分析与复现实验。

    在复用预训练模型权重与一致配置前提下，通过梯度优化最小化扰动同时强制模型输出指定目标转录，最终产出可回放的对抗样本及评估指标。
    """

    """脚本主入口：加载音频、初始化攻击器、执行攻击并保存结果。"""
    # initialize_globals() 会加载 DeepSpeech 的全局配置（alphabet、特征窗口、语言模型参数等）并写入 Config/FLAGS
    # 确保后续构建前向图与 CTC 解码时使用与训练一致的超参。
    initialize_globals()
    # 这些字符就是允许使用的字母表；其中 - 是特殊符号，对应 CTC 解码里的空白（epsilon），不能出现在目标短语里
    # DeepSpeech 的 alphabet：与 Config.alphabet 保持一致，'-' 表示 CTC 空白符
    toks = " abcdefghijklmnopqrstuvwxyz'-"
    
    # 创建一个 TensorFlow 会话并在代码块结束后自动关闭它
    with tf.Session() as sess:
        # 初始化容器，用于存储批量音频及指标
        finetune = []
        audios = []
        lengths = []
        names = []
        source_dBs = []
        distortions = []
        high_pertub_bounds = []
        low_pertub_bounds = []

        if FLAGS.output is None:
            # 未指定 output，则必须提供 outprefix
            assert FLAGS.outprefix is not None
        else:
            # 指定 output 时不再使用 outprefix，且输入/输出数量需一致
            assert FLAGS.outprefix is None

            # FLAGS.input、FLAGS.output、FLAGS.finetune 都是多值参数列表。
            # 若显式给出 --output，就要求每个输入音频对应一个输出文件，所以检查 len(input) == len(output)。
            assert len(FLAGS.input) == len(FLAGS.output)
        if FLAGS.finetune is not None and len(FLAGS.finetune):
            assert len(FLAGS.input) == len(FLAGS.finetune)
            
        # Load the inputs that we're given
        # TODO: [FINDBUG] loading multiple inputs is possible, 
        #       but there are some weird things going on at the end of every transcription 
        # 加载我们提供的输入
        for i in range(len(FLAGS.input)):
            # 采样率 fs，int16 波形 audio
            # FLAGS.input 是多值参数列表，包含所有待攻击音频的路径，在前面用 f.DEFINE_multi_string() 定义
            fs, audio = wav.read(FLAGS.input[i])
            names.append(FLAGS.input[i])
            assert fs == 16000
            assert audio.dtype == np.int16
            if (audio.shape[-1] == 2):
                # 若是立体声，取第二个声道，与 DeepSpeech 训练数据兼容
                # 因为后续会送入 get_logits 构建的 DeepSpeech 前向图
                audio = np.squeeze(audio[:,1])
                print(audio.shape)
            # 计算音频的最大幅值（峰值）对应的分贝（dB，分贝值），公式是 source_dB = 20 * np.log10(最大绝对幅值)。
            # dB（分贝） 是衡量音频信号强度（响度、能量）的对数单位，这里反映原始音频的最大响度。
            source_dB = 20 * np.log10(np.max(np.abs(audio)))
            print('source dB', source_dB)
            source_dBs.append(source_dB)
            audios.append(list(audio))
            lengths.append(len(audio))

            if FLAGS.finetune is not None:
                finetune.append(list(wav.read(FLAGS.finetune[i])[1]))   
            
        maxlen = max(map(len,audios))
        # 对齐长度，尾部补零，方便批量喂入
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        
        # 目标文本
        phrase = FLAGS.target 
        print("\nAttack phrase: ", phrase) 
        
        attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=len(audios),
                        mp3=FLAGS.mp3,
                        learning_rate=FLAGS.lr,
                        num_iterations=FLAGS.iterations,
                        l2penalty=FLAGS.l2penalty,
                        restore_path=FLAGS.restore_path)

        start_time = time.time() 
        deltas, first_hits, best_hits = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               toks,
                               finetune)
        runtime = time.time() - start_time

        print("Finished in {}s.".format(runtime))
        # 保存到指定的输出位置
        if FLAGS.mp3:
            # MP3 模式：利用上文 convert_mp3 写入压缩音频
            convert_mp3(deltas, lengths)
            copyfile("/tmp/saved.mp3", FLAGS.output[0])
            print("Final distortion", np.max(np.abs(deltas[0][:lengths[0]]-audios[0][:lengths[0]])))
        else:
            for i in range(len(FLAGS.input)):
                # 如果命令行给了 --output（逐项对应），就用 FLAGS.output[i]；否则用 outprefix+索引生成文件名。
                if FLAGS.output is not None:
                    path = FLAGS.output[i]
                else:
                    path = FLAGS.outprefix+str(i)+".wav"
                # 把第 i 个对抗波形写成 16 kHz、16‑bit PCM 的 WAV。
                # 写入前对样本做 round 并用 np.clip 限定在 int16 范围（-215 … 215-1）。
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
                
                # 指标计算(用于后续统计/分析)
                # 比较有效帧,得到扰动信号
                diff = deltas[i][:lengths[i]]-audios[i][:lengths[i]] 

                # 扰动的最大/小绝对值(峰值)
                high_pertub_bound = np.max(np.abs(diff))
                low_pertub_bound = np.min(np.abs(diff[diff!=0]))
                
                # 换算成 db,, 并减去原始音频的 dB 得到相对响度(越小说明扰动越小)
                distortion = 20 * np.log10(np.max(np.abs(diff))) - source_dBs[i]

                high_pertub_bounds.append(high_pertub_bound)
                low_pertub_bounds.append(low_pertub_bound)
                distortions.append(distortion)
                print("Final noise loudness: ", distortion)

    # 存放将写入 CSV 文件的各项数值
    data_dict = {
        'filename': names,
        'length' : lengths,
        'attack_runtime': [runtime]*len(names),
        'source_dB': source_dBs,
        'noise_loudness': distortions,
        'high_pertubation_bound' : high_pertub_bounds,
        'low_pertubation_bound' : low_pertub_bounds,
        'first_hit' : first_hits,
        'best_hit' : best_hits
    }     
    # 汇总为 DataFrame，方便下游分析与复现
    df = pd.DataFrame(data_dict, columns=['filename', 'length', 'attack_runtime', 'source_dB', 'noise_loudness', 'high_pertubation_bound', 'low_pertubation_bound', 'first_hit', 'best_hit'])
    csv_filename = "tmp/attack-{}.csv".format(FLAGS.lang, time.strftime("%Y%m%d-%H%M%S"))    
    df.to_csv(csv_filename, index=False, header=True)   
    """
    下游分析: 把每条样本的元数据和指标（文件名、长度、运行时间、原始 dB、噪声响度、扰动峰值、首次/最佳命中迭代等）聚合到表格
             方便后续统计汇总、绘图、比对不同参数/模型/策略的效果（成功率、平均扰动、收敛速度、鲁棒性等）。

    复现(reproducibility): 保存可复现实验所需的信息与结果(输入文件名、目标短语、checkpoint、运行时长、生成的对抗样本路径及对应指标),
                           便于别人 / 自己在不同时间重复实验或对比改动后的行为。
    """
 
            
"""
run_script() 只是把 DeepSpeech 原有的命令行参数注册好(create_flags())，然后交给 absl.app.run(main) 去解析参数并执行 main。
换言之, 它负责构造命令行接口并触发整套攻击流程；没有这步就无法用 --input … --target … --restore_path … 方式运行脚本。
"""
def run_script():
    """脚本入口包装：注册 DeepSpeech flags 并交给 absl 框架执行。"""
    create_flags()
    absl.app.run(main)
    
    
if __name__ == "__main__":
    # 作为脚本运行时，执行整个攻击流程；若被其他模块导入则不会自动运行
    run_script()