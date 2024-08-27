import jiwer
import fugashi
from asr import process_audio_file

# 定义自定义的日语标准化转换
class CustomJapaneseStandardization:
    def __init__(self):
        self.transformation = jiwer.Compose([
            self.custom_segmentation,
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces()
        ])
        self.tagger = fugashi.Tagger()  # 使用 fugashi 进行日语分词
    
    def custom_segmentation(self, s):
        # 使用 fugashi 对日语文本进行分词
        words = [word.surface for word in self.tagger(s)]
        return " ".join(words)

    def __call__(self, s):
        return self.transformation(s)

def compute_wer_from_file(txt_file_path, language="japanese"):
    wer_scores = []
    
    # 初始化自定义标准化处理
    standardization = CustomJapaneseStandardization()
    
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
txt_file_path = "/home/gyz/voice_eva/data/japanese.txt"
language = "japanese"
compute_wer_from_file(txt_file_path, language)
