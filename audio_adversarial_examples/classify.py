"""
之所以不直接“调用模型库”是因为攻击需要访问网络的 logits、梯度等中间量, 仅靠预训练权重不足, 所以必须按 DeepSpeech 的源码方式手动搭图。
若改用别的模型，也需要写一个功能等价的包装脚本，把该模型的前向推理、解码和参数管理梳理清楚，方便攻击代码调度与复用。
"""


# 导入标准库 sys，后续需要操作解释器路径等信息
import sys
# 导入 NumPy，负责以 ndarray 形式管理音频样本
import numpy as np
# 导入 TensorFlow 1.x，用于构建和执行 DeepSpeech 计算图
import tensorflow as tf
# 从 SciPy 导入 wavfile 读取器，用于解析 WAV 文件
import scipy.io.wavfile as wav
# 导入 time 模块，为输出文件生成时间戳
import time
# 导入 os 模块，处理路径与环境变量
import os
# 通过设置 CUDA_VISIBLE_DEVICES 为空字符串，强制脚本仅使用 CPU（方便在无 GPU 环境下复现论文设置）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 再次导入 sys，保持与原始脚本一致（虽冗余但不改动原行为）
import sys  # pylint: disable=reimported
# 导入 pandas，将识别结果保存为表格形式
import pandas as pd

try:
    # 尝试导入 pydub，用于解析 MP3 音频
    import pydub
    # struct 模块用于把原始字节序列解码为 16 位有符号整数
    import struct
except Exception:  # 捕获导入失败异常
    # 若 pydub 不可用则提示用户，MP3 支持将失效
    print("pydub was not loaded, MP3 compression will not work")

# 将本地 DeepSpeech 源码目录加入 Python 搜索路径，便于导入内部模块
# 方便后面导入 DeepSpeech 以及使用其工具代码
# 因为攻击脚本需要复建 DeepSpeech 的计算图以便在迭代的过程中多次前向推理和求解梯度，不能只依赖模型的预训练权重
sys.path.append("DeepSpeech")
# 导入 DeepSpeech 包（此处并未直接调用，但保持依赖完整）
import DeepSpeech  # pylint: disable=unused-import
# 从项目的 tf_logits 模块引入 get_logits，用于构建 DeepSpeech 的前向 logits 图
from tf_logits import get_logits

# 从 DeepSpeech 训练工具集中导入 create_flags 与全局 FLAGS
# 这些工具是用于定义并保存如语言模型这些命令行选项的
from deepspeech_training.util.flags import create_flags, FLAGS

# 导入 DeepSpeech 的配置对象和初始化函数
# 脚本借助它读取这些默认设置，从而确保与训练时的参数一致
from deepspeech_training.util.config import Config, initialize_globals
# 导入 CTC Beam Search 解码器以及语言模型打分器
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
# absl.flags 提供命令行参数解析框架
import absl.flags

# 为 absl.flags 创建简写引用，方便后续定义参数
f = absl.flags
# 定义 input 参数，指定要转录的音频文件路径（支持多个文件，用空格分隔）
# 待转写的音频文件路径
f.DEFINE_string('input', None, 'Input audio .wav file(s), at 16KHz (separated by spaces)')

# 定义 restore_path 参数，指向 DeepSpeech checkpoint 文件
# checkpoint 保存了神经网络的权重和优化器状态，供脚本加载模型时使用
f.DEFINE_string('restore_path', None, 'Path to the DeepSpeech checkpoint (ending in best_dev-1466475)')
# 注册验证器，确保输入音频路径存在且可读
f.register_validator('input', os.path.isfile, message='The input audio pointed to by --input must exist and be readable.')


def classify():
    """读取单个音频文件，执行 DeepSpeech 转录，并将结果打印及保存。"""

    # 创建 TensorFlow Session，所有图执行都在该上下文中完成
    # Session 用于管理和执行计算图
    with tf.Session() as sess:
        # 判断输入文件扩展名是否为 MP3
        if FLAGS.input.split('.')[-1] == 'mp3':
            # 使用 pydub 将 MP3 转为 AudioSegment 对象
            # AudioSegment 可解析 MP3 等压缩格式
            raw = pydub.AudioSegment.from_mp3(FLAGS.input)
            # 将原始字节流按 2 字节切片并解码为 16 位整数，生成 NumPy 数组
            # DeepSpeech 期望输入为 16 位有符号 PCM 波形（int16）
            # 也就是要遵循目标模型的接口规范
            audio = np.array([
                struct.unpack('<h', raw.raw_data[i:i + 2])[0]
                for i in range(0, len(raw.raw_data), 2)
            ])
        # 如果是 WAV 文件，则使用 scipy.io.wavfile 读取
        elif FLAGS.input.split('.')[-1] == 'wav':
            _, audio = wav.read(FLAGS.input)  # 返回采样率（忽略）和样本数据
            # 当音频为双声道时，取第二个声道以匹配 DeepSpeech 训练格式
            if audio.shape[-1] == 2:
                audio = np.squeeze(audio[:, 1])  # 压缩成一维数组
                print(audio.shape)  # 输出形状供调试参考
        else:
            # 非 WAV/MP3 格式会直接抛出异常
            raise Exception('Unknown file format')

        # 获取音频样本长度，后续定义占位符需要该信息
        N = audio.shape[0]
        # 定义浮点类型的输入占位符，批大小为 1，长度为音频长度
        new_input = tf.placeholder(tf.float32, [1, N])
        # 定义长度占位符，记录真实帧数
        lengths = tf.placeholder(tf.int32, [1])

        # 使用变量作用域包装 logits 构建过程，允许重用变量以兼容 TF1.x 约定
        # 用 with 来管理作用域
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            # 调用项目自带的 get_logits 构建 DeepSpeech 前向图
            # 输出每个时间步对应的声学 logits
            logits = get_logits(new_input, lengths)

        # 创建 Saver 对象，负责从 checkpoint 中恢复模型权重
        saver = tf.train.Saver()
        # 根据命令行参数指定的路径加载 DeepSpeech 预训练模型
        saver.restore(sess, FLAGS.restore_path)

        # logits 是未归一化分数,尚未满足概率约束
        # 对 logits 施加 softmax，得到每个时间步的类别概率分布
        probs = tf.nn.softmax(logits, name='logits')
        # 去掉多余的首维度，使张量形状与解码器输入匹配
        # 这是因为输出形状为 [1, T, C],而 CTC 解码期望 [T, C]
        probs = tf.squeeze(probs)

        # 根据特征提取规则估计有效帧长，匹配 DeepSpeech 0.9.3 的预处理
        length = (N - (2 * Config.audio_step_samples / 3)) // 320

        # 执行 Session，获得声学概率矩阵
        r = sess.run(probs, {new_input: [audio], lengths: [length]})

        # 若提供了语言模型 scorer 文件，则构建带语言模型的评分器
        if FLAGS.scorer_path:
            # 语言模型分值配合声学概率共同决定候选序列的得分
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet)
        else:
            scorer = None  # 否则不启用语言模型，仅凭声学概率解码

        # 使用 CTC Beam Search 进行解码，获取概率最高的转录结果
        decoded = ctc_beam_search_decoder(
            r,
            Config.alphabet,
            FLAGS.beam_width,
            scorer=scorer,
            cutoff_prob=FLAGS.cutoff_prob,
            cutoff_top_n=FLAGS.cutoff_top_n,
        )

        # 打印分隔线，便于阅读终端输出
        print('-' * 80)
        print('-' * 80)
        # 打印提示信息
        print('Classification:')
        # 输出 beam search 得到的第一名转录文本
        print(decoded[0][1])
        # 再打印分隔线结束输出块
        print('-' * 80)
        print('-' * 80)

        # 保存结果到 CSV 文件，便于后续分析

        # 将结果打包成字典，包含音频文件名与识别文本
        data_dict = {'name': [FLAGS.input], 'transcript': [decoded[0][1]]}
        # 构建 DataFrame，便于导出 CSV
        df = pd.DataFrame(data_dict, columns=['name', 'transcript'])
        # 构造输出文件名，带时间戳避免覆盖旧数据
        csv_filename = 'tmp/classify-{}.csv'.format(time.strftime('%Y%m%d-%H%M%S'))
        # 将识别结果写入 CSV 文件
        df.to_csv(csv_filename, index=False, header=True)


"""
main 是传给 absl.app.run 的入口函数：先调用 initialize_globals() 加载 DeepSpeech 配置，再执行 classify() 完成一次转录。
run_script 负责先 create_flags() 注册命令行参数，再触发 absl.app.run(main)。
if __name__ == '__main__': 是 Python 启动约定，表示只有脚本被直接执行时才运行 run_script()；
当文件被别处 import 时, __name__ 不等于 '__main__'，这些语句会跳过。
双下划线 __ 只是 Python 为特殊变量保留的命名方式。
"""

def main(_):
    """absl.app 入口函数，负责初始化并触发转录流程。"""

    # 初始化 DeepSpeech 的全局配置，例如加载 alphabet 及相关参数
    initialize_globals()
    # 调用上方定义的分类函数执行完整流程
    classify()


def run_script():
    """脚本入口封装：注册命令行参数并交给 absl.app 运行。"""

    # 创建 DeepSpeech 所需的 flag 定义（包含语言模型、beam 宽度等）
    create_flags()
    # 解析命令行并运行 main
    absl.app.run(main)


# 仅当脚本作为主程序执行时才进入此分支
if __name__ == '__main__':
    # 调用入口函数，开始执行分类流程
    run_script()
