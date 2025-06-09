# So-VITS-SVC 4.1 模型结构说明

## 1. 项目简介

So-VITS-SVC（Soft Voice Conversion Singing Voice Conversion）是一个基于 VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）的歌声转换AI模型。该模型通过提取源音频的语音特征和F0（基频）信息，替换VITS中原本的文本输入，从而实现歌声转换效果。

So-VITS-SVC 4.1版本在4.0版本的基础上进行了多项改进，包括更换特征输入为ContentVec的第12层Transformer输出、增加浅层扩散模型、支持Whisper语音编码器、增加静态/动态声线融合、增加响度嵌入以及特征检索等功能。

## 2. 模型整体架构

So-VITS-SVC 4.1的整体架构由以下几个主要模块组成：

### 2.1 语音编码器 (Speech Encoder)

语音编码器负责从源音频中提取内容特征，So-VITS-SVC 4.1支持多种编码器：

- **ContentVec**：默认推荐的编码器，使用第12层Transformer输出作为特征
- **HubertSoft**：另一种可选的编码器
- **Whisper-PPG**：4.1版本新增的编码器选项
- **ONNX版本编码器**：包括ContentVec和HubertSoft的ONNX实现，用于加速推理

### 2.2 F0预测器 (F0 Predictor)

F0预测器负责提取和处理音高信息，支持多种算法：

- **PM (Parselmouth)**：默认的F0提取方法
- **CREPE**：基于深度学习的F0提取方法
- **Harvest**：另一种F0提取算法
- **DIO**：另一种F0提取算法

### 2.3 主干网络 (SynthesizerTrn)

主干网络基于VITS架构，但进行了修改以适应歌声转换任务：

- **编码器 (Encoder)**：处理输入特征
- **流模型 (Flow)**：使用ResidualCouplingBlock进行特征变换
- **解码器 (Decoder)**：生成最终的音频波形
- **说话人编码器 (Speaker Encoder)**：处理目标说话人的身份信息

### 2.4 判别器 (Discriminator)

采用多周期判别器(MultiPeriodDiscriminator)进行对抗训练，包括：

- **周期判别器 (DiscriminatorP)**：在不同周期上评估生成音频的质量
- **尺度判别器 (DiscriminatorS)**：在不同尺度上评估生成音频的质量

### 2.5 浅层扩散模型 (Shallow Diffusion)

4.1版本新增的浅层扩散模型，用于提升音质和解决电音问题：

- 基于DDSP-SVC的扩散模型
- 可以通过k_step参数控制扩散步数

### 2.6 NSF-HiFiGAN增强器

可选的音频增强器，用于提升音质：

- 对于训练数据较少的模型有一定增强效果
- 但对于训练充分的模型可能有反面效果

## 3. 数据处理流程

### 3.1 数据预处理

1. **音频切片**：将音频切成5-15秒的片段
2. **重采样**：将音频重采样至44100Hz单声道
3. **响度匹配**：可选的响度处理
4. **数据集划分**：自动划分训练集和验证集
5. **特征提取**：提取内容特征和F0信息

### 3.2 训练流程

训练过程主要包括以下步骤：

1. **特征提取**：从输入音频中提取内容特征和F0
2. **生成器前向传播**：通过SynthesizerTrn生成转换后的音频
3. **判别器评估**：使用MultiPeriodDiscriminator评估生成音频的质量
4. **损失计算**：计算多种损失函数，包括对抗损失、特征匹配损失、KL散度损失等
5. **参数更新**：使用AdamW优化器更新模型参数

## 4. 推理流程

推理过程主要包括以下步骤：

1. **音频预处理**：加载源音频并进行必要的预处理
2. **特征提取**：提取内容特征和F0信息
3. **音高调整**：可以对F0进行调整以改变音高
4. **模型推理**：通过训练好的模型生成目标说话人的音频
5. **后处理**：可选的增强处理，如NSF-HiFiGAN增强或浅层扩散

### 4.1 推理参数说明

- **trans**：音高调整（半音）
- **auto_predict_f0**：是否自动预测音高（歌声转换不建议开启）
- **cluster_infer_ratio**：聚类方案或特征检索占比
- **noise_scale**：噪音级别，影响咬字和音质
- **f0_predictor**：选择F0预测器（pm、crepe、dio、harvest）
- **enhance**：是否使用NSF_HIFIGAN增强器
- **shallow_diffusion**：是否使用浅层扩散
- **use_spk_mix**：是否使用角色融合

## 5. 模型创新点

So-VITS-SVC 4.1相比之前版本的主要创新点：

1. **特征输入更换**：使用ContentVec的第12层Transformer输出，提供更好的内容表示
2. **浅层扩散**：引入浅层扩散模型提升音质和解决电音问题
3. **Whisper编码器支持**：增加对Whisper语音编码器的支持
4. **声线融合**：支持静态和动态声线融合，实现多角色混合
5. **响度嵌入**：增加响度嵌入功能，更好地保留源音频的响度特征
6. **特征检索**：引入来自RVC的特征检索技术，提升转换效果

## 6. 文件结构说明

项目的主要文件及其功能：

- **models.py**：定义了模型的核心结构，包括SynthesizerTrn、Encoder、Discriminator等
- **train.py**：训练脚本，实现模型的训练逻辑
- **inference_main.py**：推理脚本，用于使用训练好的模型进行歌声转换
- **preprocess_flist_config.py**：数据预处理脚本，生成训练所需的配置文件
- **utils.py**：工具函数，包括F0处理、模型加载保存等功能
- **modules/**：包含模型的各个模块实现
- **vencoder/**：包含各种语音编码器的实现
- **configs/**：配置文件目录
- **logs/**：训练日志和模型保存目录

## 7. 使用流程

### 7.1 环境准备

1. 安装所需依赖：`pip install -r requirements.txt`
2. 下载预训练模型：ContentVec、NSF-HiFiGAN等

### 7.2 数据准备与预处理

1. 准备数据集：将音频文件放入dataset_raw目录
2. 音频切片：使用audio-slicer等工具切分音频
3. 重采样：`python resample.py`
4. 生成配置文件：`python preprocess_flist_config.py --speech_encoder vec768l12`

### 7.3 模型训练

执行训练脚本：`python train.py`

### 7.4 推理转换

执行推理脚本：`python inference_main.py -m [模型路径] -c [配置文件路径] -n [输入音频] -t [音高调整] -s [目标说话人]`

## 8. 总结

So-VITS-SVC 4.1是一个功能强大的歌声转换模型，通过结合VITS架构、ContentVec特征提取和浅层扩散等技术，实现了高质量的歌声转换效果。该模型支持多种语音编码器和F0提取方法，并提供了丰富的推理参数以满足不同场景的需求。其模块化设计使得模型具有良好的扩展性和可定制性，为歌声转换领域提供了一个强大的开源解决方案。 
