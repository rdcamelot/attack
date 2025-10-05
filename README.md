### 相关文件
`audio_adversarial_examples` 来自于仓库 https://github.com/carlini/audio_adversarial_examples ，用于学习如何对一个语音模型进行攻击，论文中的一些思想也来自于这里

但因为论文的时间比较久远，里面使用的代码是 Tensorflow 下的，以及威胁模型 DeepSpeech 也已经停止了维护，因此主要是本人作为初学者学习下相关的流程

### 攻击模型
攻击的模型选择为 Wav2vec ，因为它能拿到梯度，从而能更新扰动，同时开源且效果较强

选择使用 Hugging Face Transformers 的方式，来安装预训练模型

解码时使用 beam search ,不额外使用 LM