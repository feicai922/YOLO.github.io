## 农林机器人YOLO模型
话不多说，先上模型链接（如果不想自己练的可以直接下载）
[模型链接](https://pan.baidu.com/s/1orCco9f862lsgxseRpf6AQ?pwd=wzhx)

## 处理效果

![YOLO处理效果展示](images/test_1.jpg)

## 使用方法：
先跟着老师的步骤把环境配合，默认你是配好环境的
创建demo.py:
```python
import os
from ultralytics import YOLO
from tqdm import tqdm
import logging
import torch

# 设置Ultralytics日志等级
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def process_images(input_folder, output_folder, model_path):
    # 判断设备：CUDA（GPU）可用则使用GPU，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用的设备：{device}")

    # 加载YOLO模型
    model = YOLO(model_path)
    model.to(device)  # 显式将模型移至指定设备

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入图片列表
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    # 处理每张图片
    for image_name in tqdm(image_files, desc="处理图片", unit="image"):
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        # 模型预测，设置置信度阈值为0.3，并使用指定设备
        results = model(image_path, conf=0.3, device=device)

        # 保存检测结果
        if isinstance(results, list):
            for result in results:
                result.save(output_path)
        else:
            results.save(output_path)


# 示例调用
input_folder = r"D:\1111training_code\tests"  # 你的输入图片文件夹路径
output_folder = r"D:\1111training_code\outputs"  # 你的输出图片文件夹路径
model_path = r"D:\1111training_code\best.pt"  # 你的YOLO模型路径

process_images(input_folder, output_folder, model_path)

```

然后就可以在你的YOLO环境中运行了

