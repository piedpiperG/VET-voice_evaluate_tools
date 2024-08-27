import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def process_audio_file(audio_path, language="english"):
    # 初始化模型和处理器
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    # 处理单个音频文件
    print(f"Processing: {audio_path}")
    result = pipe(audio_path, generate_kwargs={"language": language})
    
    transcription = result['text']
    print(f"Result for {audio_path}: {transcription}\n")
    
    return transcription

# 示例调用
# audio_path = "/home/gyz/voice_eva/data/clean_2.wav"
# language = "english"
# asr_result = process_audio_file(audio_path, language)

# # 打印结果
# print(f"Transcription: {asr_result}")
