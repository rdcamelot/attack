"""
负责把 DeepSpeech 的特征提取与前向网络封装成可直接调用的函数：
    接收原始波形 → 做 STFT/MFCC 特征 → 构建 DeepSpeech 模型 → 返回 logits。
攻击脚本借此拿到 logits 和梯度，因此必须引入 DeepSpeech 源码、自定义特征管线与模型搭图逻辑；
换用其他模型时，也需要写一份等价的“特征 + 前向 + logits”封装供攻击算法调用。

本文件主要复现 DeepSpeech 训练端的预处理与前向图: 波形→STFT→MFCC→拼窗→送入 create_model 得到 logits
供攻击脚本直接访问和求梯度。
"""


import mfcc  # 复用项目内的 MFCC 提取函数，保持与论文实现一致

from deepspeech_training.train import create_model, create_overlapping_windows  
# 从官方训练代码直接导入模型构建与帧处理工具
from deepspeech_training.util.config import Config  
# 加载 DeepSpeech 统一配置（如特征长度、RNN 规模等）
from deepspeech_training.util.flags import FLAGS  
# 访问 DeepSpeech 定义的命令行参数（采样率等）

import DeepSpeech  # pylint: disable=unused-import  
# 导入包以确保其依赖初始化（即便未直接使用）
import tensorflow as tf  
# 使用 TF1.x 构建静态图
import numpy as np  # 供窗函数等计算调用
import sys  # 用于调整模块搜索路径

# 将 DeepSpeech 源码目录加入 sys.path，后续 import 才能解析本地子模块
sys.path.append("DeepSpeech")


# 直接照搬 TensorFlow 源码，确保特征提取与原始 DeepSpeech 完全一致。
# DeepSpeech 在训练时依赖“波形→STFT→MFCC→网络”的流水线，推理或攻击必须复现同样的特征处理才能喂给模型；
# 换用其他模型时也要遵循其原生预处理流程（有的框架自带，有的需手写），否则输入分布不对会导致识别或攻击失效。
def periodic_hann_window(window_length, dtype):
    """
    Return periodic Hann window. Implementation based on:
    https://github.com/tensorflow/tensorflow/blob/bd962d8cdfcda01a23c7051fa05e3db86dd9c30f/tensorflow/core/kernels/spectrogram.cc#L28-L36
    """
    # DeepSpeech 的 STFT 使用“周期性 Hann 窗”，这里复制官方实现以确保特征与训练时完全对齐
    # 在做短时傅里叶变换（STFT）时对每一帧样本乘以此窗，减小帧截断带来的频谱泄漏
    return 0.5 - 0.5 * tf.math.cos(2.0 * np.pi * tf.range(tf.to_float(window_length), dtype=dtype) / (tf.to_float(window_length)))


def get_logits(audio, length):
    """
    Compute the logits for a given waveform
    using functions from DeepSpeech v0.9.3.
    """
    # DeepSpeech 期望输入为归一化到 [-1, 1] 的浮点波形，这里把原始 int16 样本完成缩放与类型转换
    audio = tf.cast(audio / 2 ** 15, tf.float32)

    # 复现训练端的 STFT：512 点窗，步长 320（对应 20ms/12.5ms），并使用周期性 Hann 窗
    stfts = tf.signal.stft(
        audio,
        frame_length=512,
        frame_step=320,
        fft_length=512,
        window_fn=periodic_hann_window
    )
    # 取得功率谱（幅度平方），后续供 MFCC 计算
    spectrogram = tf.square(tf.abs(stfts))

    # 通过项目中的 MFCC 工具把功率谱映射到梅尔倒谱系数；采样率、特征维度均取自 DeepSpeech 配置
    features = mfcc.compute_mfcc(spectrogram=spectrogram, sample_rate=FLAGS.audio_sample_rate,
                                 upper_edge_hertz=FLAGS.audio_sample_rate / 2, dct_coefficient_count=Config.n_input)

    # DeepSpeech 在训练/推理时会拼接重叠帧（lookahead/lookback），以提供上下文
    features = create_overlapping_windows(features)

    # 构建完整的 DeepSpeech 计算图；推理阶段关闭 dropout，因此传入全 None
    no_dropout = [None] * 6
    logits, _ = create_model(features, seq_length=length,
                             dropout=no_dropout, overlap=False)

    # 返回未归一化的分类分数（logits），供外部执行 softmax、CTC 或求梯度
    return logits


# 攻击代码复用已训练好的 DeepSpeech 检查点，不再重新训练，但必须手动按原流程做特征预处理并构建同一前向图
# 这样既能喂入模型获得 logits，又能借助 TensorFlow 计算这些 logits 对输入的梯度，从而迭代生成对抗扰动。

# 我们要得到 logits 对输入的梯度，而这个 logits 是通过模型得到的，于是我们要借助模型

# 攻击要在输入上迭代“减损”，核心就是用 DeepSpeech 前向图得到 logits
# 再借由 TensorFlow 反向求出 logits（或损失）对音频的梯度，从而调整波形生成对抗扰动。