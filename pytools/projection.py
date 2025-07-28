import json
import os
import time

import cv2
import numpy as np
import open3d as o3d

for i in range(1, 11):
    # 设置路径
    start = time.time()
    base_dir = "render_quick/rice_demo{}_reg_run0".format(i)
    eval_txt_path = os.path.join(base_dir, "eval.txt")
    ply_path = os.path.join(base_dir, "demo{}_point_cloud_world.ply".format(0))
    json_path = "rice/demo{}/transforms.json".format(i)
    img_dir = base_dir
    output_dir = os.path.join(base_dir, "projection")
    os.makedirs(output_dir, exist_ok=True)

    # 读取 eval.txt
    with open(eval_txt_path, "r") as f:
        image_list = eval(f.read())

    # 加载点云
    cloud_o3d = o3d.io.read_point_cloud(ply_path)
    cloud = np.asarray(cloud_o3d.points)
    colors = np.asarray(cloud_o3d.colors)
    colors = (colors * 255).astype(np.uint8)

    # 读取相机参数
    with open(json_path, "r") as f:
        data = json.load(f)

    fl_x = data['fl_x']
    fl_y = data['fl_y']
    cx = data['cx']
    cy = data['cy']
    w = int(data['w'])
    h = int(data['h'])
    dist = [data['k1'], data['k2'], data['p1'], data['p2'], data.get('k3', 0.0)]

    # 相机内参矩阵
    camera_matrix = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])

    # 相机畸变参数
    dist_coeffs = np.array(dist)

    # 用于标记所有黑色点
    valid_mask = np.ones(len(cloud), dtype=bool)

    for idx, image_path in enumerate(image_list):
        # print(image_path)
        image_name = f"eval_img_{idx:04d}.png"
        full_image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"图像未找到: {full_image_path}")
            continue

        file_name = os.path.basename(image_path)
        # print(file_name)
        frame_data = None
        for frame in data['frames']:
            if file_name in frame['file_path']:
                # print(frame)
                frame_data = frame
                break

        if frame_data is None:
            print(f"未在 transforms.json 中找到对应的 transform_matrix: {file_name}")
            continue

        ex_matrix = np.array(frame_data['transform_matrix'])
        ex_matrix[:3, 1:3] *= -1  # Y, Z 轴反转

        ex_matrix = np.linalg.inv(ex_matrix)

        rotation_matrix = ex_matrix[:3, :3]
        translation_vector = ex_matrix[:3, 3]
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        tvec = translation_vector

        projected_points, _ = cv2.projectPoints(cloud, rvec, tvec, camera_matrix, dist_coeffs)
        x = projected_points[:, 0, 0]
        y = projected_points[:, 0, 1]

        in_image_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x_int = x[in_image_mask].astype(np.int32)
        y_int = y[in_image_mask].astype(np.int32)
        indices = np.where(in_image_mask)[0]

        # 在图像上绘制投影点
        vis_image = image.copy()
        for xi, yi in zip(x_int, y_int):
            cv2.circle(vis_image, (xi, yi), radius=1, color=(0, 255, 0), thickness=-1)

        # 保存投影结果图像
        projection_output_path = os.path.join(output_dir, f"projected_{idx:04d}.png")
        cv2.imwrite(projection_output_path, vis_image)
        print(f"已保存投影图像: {projection_output_path}")

        # 取图像上的颜色值
        pixel_colors = image[y_int, x_int]  # shape: (N, 3)
        is_black = np.all(pixel_colors == [0, 0, 0], axis=1)

        # 将黑色像素对应的点设置为无效
        black_indices = indices[is_black]
        valid_mask[black_indices] = False

        print(f"图像 {image_name} 中发现黑色点: {len(black_indices)} 个")

    # 最终过滤点云和颜色
    filtered_cloud = cloud[valid_mask]
    filtered_colors = colors[valid_mask]

    # 保存新的点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float32) / 255.0)

    filtered_ply_path = os.path.join(base_dir, "filtered_point_cloud.ply")
    o3d.io.write_point_cloud(filtered_ply_path, filtered_pcd)
    print(f"已保存去除黑色像素点后的点云: {filtered_ply_path}")
    print(f"处理完成，耗时: {time.time() - start:.2f} 秒")
    break