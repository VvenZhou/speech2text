import serial
import torchaudio
import torch
import silero
from utils import Decoder

batchSize = 10000
sr = 5000
device = torch.device('cpu')

model = torch.jit.load('en_v2_jit.model', map_location=device)
model.eval()
decoder = Decoder(model.labels)

raw = []
batch = []

ser = serial.Serial('/dev/ttyACM0', 2000000)
print(ser.name)
print("start recording")
for i in range(batchSize):
    raw.append(ser.readline())
ser.close()
print("finish recording")

for line in raw:
    if line[0:-2] != b'':
        value = int(line[0:-2])
        value = float(value)*(-1)/100000
        batch.append(value)
transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=16000)
raw_wav = torch.tensor([batch])
resam_wav = transform(raw_wav)
resam_wav = resam_wav.squeeze(0)

batch = list([resam_wav])
max_seqlength = max(max([len(_) for _ in batch]), 12800)
inputs = torch.zeros(len(batch), max_seqlength)
for i, wav in enumerate(batch):
    inputs[i, :len(wav)].copy_(wav)

inputs = inputs.to(device)
print(inputs.size())
print(inputs)

output = model(inputs)
for example in output:
    print(decoder(example.cpu()))
