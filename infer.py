from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision import datasets, transforms
import torch

# 加载预训练ResNet‑50
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# MNIST本身就是PIL单通道图片；调整大小 + 转为3通道
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3)
])

# 下载MNIST测试集
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=None)

correct = 0
total = 100

for i in range(total):
    img_pil, label = test_set[i]
    img_rgb = transform(img_pil)   # 输出已经是3通道PIL图片
    inputs = processor(img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    if pred == label:
        correct += 1

acc = correct / total
print(f"Accuracy: {acc:.4f}")