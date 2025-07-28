import json
import os
import time

import cv2
import numpy as np

# 控制是否生成二值化 mask
binary_mask = True  # 设置为 False 生成灰度边缘图，True 生成二值图
start = time.time()
for i in range(1, 2):
    # 设置路径
    image_dir = 'rice/demo{}/deblur'.format(i)
    mask_dir = 'rice/demo{}/masks'.format(i)
    json_path = 'rice/demo{}/transforms.json'.format(i)

    # 创建 masks 文件夹（如果不存在）
    os.makedirs(mask_dir, exist_ok=True)

    # 遍历图像，生成拉普拉斯边缘图
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            
            # 读取图像为灰度图
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 拉普拉斯边缘检测
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.uint8(np.absolute(laplacian))

            # 归一化处理
            laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
            laplacian_norm = np.uint8(laplacian_norm)

            if binary_mask:
                # 阈值二值化
                _, laplacian_binary = cv2.threshold(laplacian_norm, 15, 255, cv2.THRESH_BINARY)

                # 使用高斯模糊去噪，平滑边缘
                # blurred = cv2.GaussianBlur(laplacian_binary, (3, 3), 0)

                # 再次阈值化去除模糊带来的灰度
                _, output_mask = cv2.threshold(laplacian_binary, 15, 255, cv2.THRESH_BINARY)
            else:
                output_mask = laplacian_norm

            # 保存 mask 图像
            mask_output_path = os.path.join(mask_dir, filename)
            # cv2.imwrite(mask_output_path, output_mask)
            print(f'Mask saved: {mask_output_path}')

    # 读取 JSON 文件
    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    # # 更新每个 frame，添加 mask_path
    # for frame in data.get('frames', []):
    #     file_path = frame.get('file_path', '')
    #     if file_path.startswith('images/'):
    #         mask_path = file_path.replace('images/', 'masks/')
    #         frame['mask_path'] = mask_path

    # 保存更新后的 JSON 文件
    # with open(json_path, 'w') as f:
    #     json.dump(data, f, indent=2)

    print(f'Updated transforms.json with mask paths using Laplacian edge detection ({"binary with Gaussian blur" if binary_mask else "non-binary"}).')

print(f'Total time: {time.time() - start} seconds')