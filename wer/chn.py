import jiwer
import jieba
import opencc
from asr import process_audio_file

# 定义自定义的标准化转换
class CustomChineseStandardization:
    def __init__(self, conversion='t2s'):
        self.transformation = jiwer.Compose([
            self.convert_to_same_form(conversion),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            self.custom_segmentation
        ])
        self.converter = opencc.OpenCC(conversion)
    
    def custom_segmentation(self, s):
        return " ".join(jieba.lcut(s))
    
    def convert_to_same_form(self, conversion):
        def convert(text):
            return self.converter.convert(text)
        return convert

    def __call__(self, s):
        return self.transformation(s)

def compute_wer_from_file(txt_file_path, language="chinese", conversion='t2s'):
    wer_scores = []
    
    # 初始化自定义标准化处理
    standardization = CustomChineseStandardization(conversion=conversion)
    
    with open(txt_file_path, 'r') as f:
        for line in f:
            # 分割音频路径和目标文本
            audio_path, reference_text = line.strip().split('|')
            
            # 使用ASR模型处理音频文件
            hypothesis_text = process_audio_file(audio_path, language)
            
            # 应用自定义标准化处理
            reference_text_standardized = standardization(reference_text)
            hypothesis_text_standardized = standardization(hypothesis_text)
            
            # 计算WER
            wer = jiwer.wer(reference_text_standardized, hypothesis_text_standardized)
            wer_scores.append(wer)
            
            # 打印结果
            print(f"Audio Path: {audio_path}")
            print(f"Reference Text: {reference_text}")
            print(f"Hypothesis Text: {hypothesis_text}")
            print(f"WER: {wer}\n")
    
    # 计算所有WER的平均值
    avg_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average WER: {avg_wer}")

# 示例调用
txt_file_path = "/home/gyz/voice_eva/data/chinese.txt"
language = "chinese"
compute_wer_from_file(txt_file_path, language, conversion='t2s')  # t2s表示简体转繁体
