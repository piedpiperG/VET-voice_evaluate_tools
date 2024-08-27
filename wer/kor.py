import jiwer
from pecab import PeCab
from asr import process_audio_file

# 定义自定义的韩语标准化转换
class CustomKoreanStandardization:
    def __init__(self):
        self.transformation = jiwer.Compose([
            self.custom_segmentation,
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces()
        ])
        self.pecab = PeCab()  # 使用 pecab 进行韩语分词
    
    def custom_segmentation(self, s):
        # 使用 pecab 对韩语文本进行分词
        return " ".join(self.pecab.morphs(s))

    def __call__(self, s):
        return self.transformation(s)

def compute_wer_from_file(txt_file_path, language="korean"):
    wer_scores = []
    
    # 初始化自定义标准化处理
    standardization = CustomKoreanStandardization()
    
    with open(txt_file_path, 'r') as f:
        for line in f:
            # 分割音频路径和目标文本
            audio_path, reference_text = line.strip().split('|')
            
            # 使用ASR模型处理音频文件
            hypothesis_text = process_audio_file(audio_path, language)
            
            # 应用自定义标准化处理（确保 Reference Text 也被分词）
            reference_text_standardized = standardization(reference_text)
            hypothesis_text_standardized = standardization(hypothesis_text)
            
            # 计算WER
            wer = jiwer.wer(reference_text_standardized, hypothesis_text_standardized)
            wer_scores.append(wer)
            
            # 打印结果
            print(f"Audio Path: {audio_path}")
            print(f"Reference Text: {reference_text_standardized}")
            print(f"Hypothesis Text: {hypothesis_text_standardized}")
            print(f"WER: {wer}\n")
    
    # 计算所有WER的平均值
    avg_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average WER: {avg_wer}")

# 示例调用
txt_file_path = "/home/gyz/voice_eva/data/korean.txt"
language = "korean"
compute_wer_from_file(txt_file_path, language)
