# VET-voice_evaluate_tools
VET:voice_evaluate_tools

## 使用conda创建环境：
conda env create -f environment.yml

data中为示例音频，clean.wav和clean_2.wav为两段完全相同的音频

## 使用方法
python [method].py（替换method为实际需要使用的方法）

具体参数在代码中修改，后续可能会改为命令行参数指定参数，看哪一种方便

对于WER:word error rate：
支持中文，英语，韩语，日语的WER计算，在WER文件夹下分别运行

对于SIM:speaker similarity：使用预训练的wavlm模型来完成，在sim文件夹下使用

## 原理介绍
## 1.WER(Word error rate)

### 步骤：

        1.通过ASR将语音转换为文本

        2.对文本进行预处理

            英语：转换大小写

            中文，韩语，日语：需要进行分词，才能进行比较

        3.通过公式进行计算
        
### 实现方式：

#### 1.ASR转换：

            使用whisper large v3能够完成对四种语言的ASR任务

            但seedTTS中对中文使用的是Paraformerzh

#### 2.文本预处理

            jiwer库提供了标点符号和大小写的预处理能力

            对中文，韩语，日语分别进行分词

#### 3.公式计算

            使用jiwer库进行计算



## 2.SIM(Speaker Similarity)

### 步骤：

        1.对于每个测试语音片段，使用经过微调的WavLM-large模型提取说话人嵌入向量。

        2.使用同一个模型对参考语音片段提取说话人嵌入向量。

        3.计算测试语音片段与参考语音片段的嵌入向量之间的余弦相似度。

### 实现方式：

        找到了现有的hugging face库，针对说话人识别任务作了微调：

        [huggingface.co](https://huggingface.co/microsoft/wavlm-base-sv)


