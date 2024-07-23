# STIO指标需要两端音频具有相同长度
# 这里对长度不同的音频作了裁剪处理，也可以作填充处理

import soundfile as sf
from pystoi import stoi

clean, fs = sf.read('data/clean.wav')
denoised, fs = sf.read('data/denoised.wav')

# 确保两个音频长度相同
min_length = min(len(clean), len(denoised))
clean = clean[:min_length]
denoised = denoised[:min_length]

# 计算STOI指数
d = stoi(clean, denoised, fs, extended=False)
print("STOI index:", d)
