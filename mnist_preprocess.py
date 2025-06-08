# 0. 导入需要的库（不用改这部分）
from PIL import Image, ImageOps  # 处理图片
import numpy as np  # 处理数组
import matplotlib.pyplot as plt  # 显示图片

# 1. 预处理函数（不用改）
def preprocess_image(image_path):
    # 打开图片并转为黑白
    img = Image.open(image_path).convert('L')
    # 反相：如果背景是白色，数字是黑色，需要反相（MNIST是黑底白字）
    img = ImageOps.invert(img)
    # 调整大小为28x28像素
    img = img.resize((28, 28), Image.LANCZOS)
    # 转为NumPy数组并归一化到[0,1]
    img_array = np.array(img) / 255.0
    # 打印形状检查（应该是(28, 28)）
    print("图片数组形状：", img_array.shape)
    return img_array

# 2. 使用示例（修改这里！）
image_path = "my_digit.png"  # 替换为你的图片路径
processed_image = preprocess_image(image_path)

# 3. 显示处理后的图片（可选）
plt.imshow(processed_image, cmap='gray')
plt.title("处理后的28x28图片")
plt.show()

# 4. 保存处理后的图片（可选）
Image.fromarray((processed_image * 255).astype('uint8')).save("processed_28x28.png")