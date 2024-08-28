from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import torchaudio

# Step 1: Load the WavLM model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')

# Step 2: Load your audio files and resample to 16kHz if necessary
def load_and_resample(audio_path, target_sr=16000):
    waveform, original_sr = torchaudio.load(audio_path)
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

audio1 = load_and_resample("/home/gyz/voice_eva/data/clean.wav")
audio2 = load_and_resample("/home/gyz/voice_eva/data/denoised.wav")

# Step 3: Extract features from the audio
inputs = feature_extractor([audio1.squeeze().numpy(), audio2.squeeze().numpy()], 
                           return_tensors="pt", 
                           sampling_rate=16000, 
                           padding=True)

# Step 4: Get embeddings from the model
with torch.no_grad():
    embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

# Step 5: Compute cosine similarity between the embeddings
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings[0], embeddings[1])

# Step 6: Determine if the speakers are the same
threshold = 0.86  # Adjust this threshold based on your dataset
if similarity < threshold:
    print("Speakers are not the same!")
else:
    print("Speakers are the same!")

# Print the similarity score
print(f"Cosine similarity: {similarity.item()}")
