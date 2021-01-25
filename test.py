import torchaudio
import torch

torchaudio.set_audio_backend("soundfile")  # switch backend

wav, sr = torchaudio.load("speech_orig.wav")
print(wav.size())
transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=16000)
wav = transform(wav)
batch = wav.squeeze(0)
batch = list([batch])

max_seqlength = max(max([len(_) for _ in batch]), 12800)
inputs = torch.zeros(len(batch), max_seqlength)
for i, wav in enumerate(batch):
    inputs[i, :len(wav)].copy_(wav)

print(inputs)
