import jiwer
from asr import process_audio_file

def compute_wer_from_file(txt_file_path, language="english"):
    wer_scores = []
    
    with open(txt_file_path, 'r') as f:
        for line in f:
            # 分割音频路径和目标文本
            audio_path, reference_text = line.strip().split('|')
            
            # 使用ASR模型处理音频文件
            hypothesis_text = process_audio_file(audio_path, language)
            
            # 计算WER
            wer = jiwer.wer(reference_text, hypothesis_text)
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
txt_file_path = "/home/gyz/voice_eva/data/english.txt"
language = "english"
compute_wer_from_file(txt_file_path, language)
