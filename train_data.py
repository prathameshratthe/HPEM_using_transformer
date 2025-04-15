import sys, os
sys.path.append(os.path.abspath("TimeSformer"))

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from TimeSformer.timesformer.models.vit import TimeSformer

# ============================= 
# 1️⃣ Fine-Grained Segment Dataset
# =============================
class SegmentedDataset(Dataset):
    def __init__(self, segments, labels, transform=None, num_frames=8):
        self.segments = segments
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        video_path, start, end = self.segments[idx]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

        frames = []
        for _ in range(int((end - start) * fps)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames from {video_path} [{start}s - {end}s]")

        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        selected = torch.stack([frames[i] for i in indices]).permute(1, 0, 2, 3)
        return selected, torch.tensor(self.labels[idx], dtype=torch.long)

# =============================
# 2️⃣ Define Fine-Grained Labels
# =============================
# Label mapping: 0 = Walking, 1 = Running, 2 = Jumping
segments = [
    # train1.mp4 (6s): 0–5s running, 5–6s jumping
    ("train1.mp4", 0, 1), 1,
    ("train1.mp4", 1, 2), 1,
    ("train1.mp4", 2, 3), 1,
    ("train1.mp4", 3, 4), 1,
    ("train1.mp4", 4, 5), 1,
    ("train1.mp4", 5, 6), 2,

    # train2.mp4 (7s): 0–6s running, 6–7s jumping
    ("train2.mp4", 0, 1), 1,
    ("train2.mp4", 1, 2), 1,
    ("train2.mp4", 2, 3), 1,
    ("train2.mp4", 3, 4), 1,
    ("train2.mp4", 4, 5), 1,
    ("train2.mp4", 5, 6), 1,
    ("train2.mp4", 6, 7), 2,

    # test_video1.mp4 (9s): 0–6s running, 6–9s jumping
    ("test_video1.mp4", 0, 1), 1,
    ("test_video1.mp4", 1, 2), 1,
    ("test_video1.mp4", 2, 3), 1,
    ("test_video1.mp4", 3, 4), 1,
    ("test_video1.mp4", 4, 5), 1,
    ("test_video1.mp4", 5, 6), 1,
    ("test_video1.mp4", 6, 7), 2,
    ("test_video1.mp4", 7, 8), 2,
    ("test_video1.mp4", 8, 9), 2,

    # test_video2.mp4 (7s): 0–4s running, 5–7s jumping
    ("test_video2.mp4", 0, 1), 1,
    ("test_video2.mp4", 1, 2), 1,
    ("test_video2.mp4", 2, 3), 1,
    ("test_video2.mp4", 3, 4), 1,
    ("test_video2.mp4", 4, 5), 0,  # assume missing segment, padding with walking
    ("test_video2.mp4", 5, 6), 2,
    ("test_video2.mp4", 6, 7), 2,

    # test_video3.mp4 (4s): 0–2s running, 2–4s jumping
    ("test_video3.mp4", 0, 1), 1,
    ("test_video3.mp4", 1, 2), 1,
    ("test_video3.mp4", 2, 3), 2,
    ("test_video3.mp4", 3, 4), 2,

    # test_video4.mp4 (8s): 0–2s running, 2–4s jumping, 4–7s running, 7–8s jumping
    ("test_video4.mp4", 0, 1), 1,
    ("test_video4.mp4", 1, 2), 1,
    ("test_video4.mp4", 2, 3), 2,
    ("test_video4.mp4", 3, 4), 2,
    ("test_video4.mp4", 4, 5), 1,
    ("test_video4.mp4", 5, 6), 1,
    ("test_video4.mp4", 6, 7), 1,
    ("test_video4.mp4", 7, 8), 2,
]

video_segments = segments[::2]
labels = segments[1::2]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

dataset = SegmentedDataset(video_segments, labels, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# =============================
# 3️⃣ Model Definition
# =============================
model = TimeSformer(img_size=224, num_classes=3, num_frames=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =============================
# 4️⃣ Training Loop
# =============================
def train():
    model.train()
    for epoch in range(5):
        running_loss = 0
        for x, y in dataloader:
            x = x.float()
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"✅ Epoch {epoch+1}, Avg Loss: {running_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), "timesformer_trained.pth")
    print("📦 Model saved as timesformer_trained.pth")

# =============================
# 5️⃣ Execute Training
# =============================
if __name__ == "__main__":
    train()
