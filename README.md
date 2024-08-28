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

