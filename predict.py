import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# 1. 定义模型（必须和训练时完全一致）
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitRecognizer().to(device)
model.load_state_dict(torch.load("best_model.pth"))  # 确保模型文件存在
model.eval()

# 3. 单张图片预测函数
def predict(image_path):
    image = Image.open(image_path).convert("L")  # 转为灰度
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
    return pred

# 4. 摄像头实时预测函数（修复窗口关闭问题）
def predict_from_camera():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            # 预处理（白底黑字 → 黑底白字）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

            # 预测
            image_tensor = transforms.ToTensor()(resized).unsqueeze(0).to(device)
            image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)
            with torch.no_grad():
                output = model(image_tensor)
                pred = output.argmax(dim=1).item()

            # 显示结果
            cv2.putText(frame, f"Pred: {pred}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Digit Recognizer", frame)

            # 按Q或ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("正在关闭摄像头...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 确保窗口关闭

# 5. 主程序
if __name__ == "__main__":
    # 方式1：测试单张图片（取消注释下面两行）
    # result = predict("your_digit.png")  # 替换为你的图片路径
    # print(f"预测结果: {result}")

    # 方式2：摄像头实时预测
    predict_from_camera()