import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 데이터셋 정의
class PostureDataset(Dataset):
    def __init__(self, angles_files, labels_files):
        """
        Args:
        - angles_files (list[str]): 관절 각도 .npy 파일 경로 리스트
        - labels_files (list[str]): 라벨 .npy 파일 경로 리스트
        """
        self.angles_files = angles_files
        self.labels_files = labels_files

    def __len__(self):
        return len(self.angles_files)

    def __getitem__(self, idx):
        # 각 파일 로드
        angles = np.load(self.angles_files[idx])  # (frames, 15)
        labels = np.load(self.labels_files[idx])  # (frames,)

        # Tensor 변환
        angles_tensor = torch.tensor(angles, dtype=torch.float32)  # Shape: (frames, 15)
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # Shape: (frames,)

        return angles_tensor, labels_tensor

# 모델 정의
class PostureClassifierFCNN(nn.Module):
    def __init__(self, input_size=15, num_classes=4):
        super(PostureClassifierFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 훈련 루프
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 데이터를 GPU로 이동
            inputs = inputs.to(device)  # Shape: (batch, frames, 15)
            labels = labels.to(device)  # Shape: (batch, frames)

            # (batch * frames, 15) 형태로 변환
            inputs = inputs.view(-1, inputs.shape[-1])  # Flatten frames
            labels = labels.view(-1)  # Flatten frames

            # 순전파
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

# 추론 함수
def predict(model, angle_input, device):
    """
    Args:
    - model (nn.Module): 훈련된 모델
    - angle_input (numpy.ndarray): (frames, 15) 형태의 관절 각도 데이터
    - device (torch.device): 사용 장치 (CPU/GPU)

    Returns:
    - predictions (list[int]): 각 프레임의 예측 라벨 리스트
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        angle_input = torch.tensor(angle_input, dtype=torch.float32).to(device)  # Shape: (frames, 15)
        output = model(angle_input)  # Shape: (frames, num_classes)
        _, predicted = torch.max(output, 1)  # Shape: (frames,)
    return predicted.cpu().numpy().tolist()