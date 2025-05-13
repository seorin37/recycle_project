'''
import torch
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# 클래스 정의
classes = ['paper', 'plastic', 'vinyl']
num_classes = len(classes)

# 모델 로딩
model = efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('efficientnet_b2_model.pth', map_location='cpu'))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
'''
'''
# 이미지 테스트
img_path = '/Users/eumseorin/Desktop/pythonProject1/dataset/testdata/'  # 테스트할 이미지 경로
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# 예측
with torch.no_grad():
    outputs = model(img_tensor)
    _, pred = torch.max(outputs, 1)

print(f"예측 결과: {classes[pred.item()]}")
'''
'''
# 테스트 폴더 경로
test_root = '/Users/eumseorin/Desktop/pythonProject1/dataset/testdata'

# 예측 loop
correct = 0
total = 0

for folder in os.listdir(test_root):
    folder_path = os.path.join(test_root, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_class = classes[pred.item()]

            true_class = folder.replace("test", "")  # testpaper → paper
            is_correct = (predicted_class == true_class)

            print(f"[{file}] 예측: {predicted_class}, 실제: {true_class}, {'✅' if is_correct else '❌'}")

            total += 1
            correct += int(is_correct)

# 전체 정확도 출력
accuracy = correct / total * 100
print(f"\n총 정확도: {accuracy:.2f}% ({correct}/{total})")
'''
import torch
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

# 클래스 정의
classes = ['paper', 'plastic', 'vinyl']
num_classes = len(classes)

# 모델 로딩
model = efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('/Users/eumseorin/Desktop/pythonProject1/recycle_project/checkpoint/efficientnet_b2_model.pth', map_location='cpu'))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 테스트 폴더 경로
test_root = '/Users/eumseorin/Desktop/pythonProject1/dataset/test_new'

# 예측 결과 저장용
y_true = []
y_pred = []
csv_results = []
correct = 0
total = 0

# 예측 loop
for folder in os.listdir(test_root):
    folder_path = os.path.join(test_root, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_class = classes[pred.item()]

            true_class = folder.replace("test", "")  # testpaper → paper
            is_correct = (predicted_class == true_class)

            print(f"[{file}] 예측: {predicted_class}, 실제: {true_class}, {'✅' if is_correct else '❌'}")

            # 결과 저장
            y_true.append(classes.index(true_class))
            y_pred.append(pred.item())
            csv_results.append([file, true_class, predicted_class])

            total += 1
            correct += int(is_correct)

# 정확도 출력
accuracy = correct / total * 100 if total > 0 else 0.0
print(f"\n총 정확도: {accuracy:.2f}% ({correct}/{total})")

# ✅ Confusion Matrix 출력 & 시각화
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
print("\n📊 Confusion Matrix:")
disp.plot(cmap='Blues', xticks_rotation=45)
plt.tight_layout()
plt.show()

# ✅ CSV 저장
csv_path = "/Users/eumseorin/Desktop/pythonProject1/recycle_project/result/test_new/efficientnet_predictions.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "predicted_label"])
    writer.writerows(csv_results)
    writer.writerow([])  # 빈 행 추가하여 구분
    writer.writerow(["accuracy", f"{accuracy:.2f}%"])

print(f"\n📁 예측 결과가 CSV로 저장되었습니다: {csv_path}")
