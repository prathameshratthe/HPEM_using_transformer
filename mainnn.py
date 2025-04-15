import sys
import os
sys.path.append(os.path.abspath("TimeSformer"))
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from TimeSformer.timesformer.models.vit import TimeSformer

# ==========================
# 1️⃣ Dataset from frame segments
# ==========================
class VideoSegmentDataset(Dataset):
    def __init__(self, video_path, segment_duration=1.0, transform=None, num_frames=8):
        self.video_path = video_path
        self.segment_duration = segment_duration
        self.transform = transform
        self.num_frames = num_frames

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.segment_size = int(self.fps * segment_duration)

        # Align segments perfectly to video duration in seconds
        self.segment_indices = [
            int(i * self.fps * self.segment_duration)
            for i in range(int(self.duration // self.segment_duration))
        ]

        self.cap.release()

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.segment_indices[idx])

        frames = []
        for _ in range(self.segment_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from segment {idx} in video {self.video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # Pad with last frame

        frame_indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in frame_indices]

        video_tensor = torch.stack(sampled_frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return video_tensor

# ==========================
# 2️⃣ Transform
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

# ==========================
# 3️⃣ Model loading
# ==========================
def load_model(model_path="timesformer_trained.pth"):
    model = TimeSformer(img_size=224, num_classes=3, num_frames=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# ==========================
# 4️⃣ Predict and show segmented labels
# ==========================
def predict_video_segments(model, video_path):
    dataset = VideoSegmentDataset(video_path, segment_duration=1.0, transform=transform, num_frames=8)
    print(f"🔍 Video: {video_path}, Total segments: {len(dataset)}")  # Moved inside function
    dataloader = DataLoader(dataset, batch_size=1)

    labels_map = {0: "Walking", 1: "Running", 2: "Jumping"}
    predictions = []

    with torch.no_grad():
        for video_segment in dataloader:
            video_segment = video_segment.to(torch.float32)
            output = model(video_segment)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)

    return predictions, labels_map


# ==========================
# 5️⃣ Show with predicted labels
# ==========================
def show_video_with_predictions(video_path, predictions, labels_map, segment_duration=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    current_segment = 0
    frame_count = 0
    label = labels_map[predictions[0]] if predictions else "Unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update label based on frame index
        if frame_count >= (current_segment + 1) * int(fps * segment_duration):
            current_segment += 1
            if current_segment < len(predictions):
                label = labels_map[predictions[current_segment]]

        cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Segmented Prediction", frame)
        if cv2.waitKey(int(frame_duration * 1000)) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# 6️⃣ Run Predictions
# ==========================
if __name__ == "__main__":
    model = load_model()

    for video_file in ["test_video2.mp4","test_video3.mp4","test_video4.mp4"]:
        print(f"🔎 Processing {video_file}")
        predictions, labels_map = predict_video_segments(model, video_file)
        print("Predicted segment labels:", [labels_map[p] for p in predictions])
        show_video_with_predictions(video_file, predictions, labels_map)
