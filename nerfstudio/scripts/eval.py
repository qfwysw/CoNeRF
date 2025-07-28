# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


def load_depth_image(path):
    """加载灰度深度图（或彩色转灰度）"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图像：{path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def compute_depth_contrast(depth_img):
    """计算整体和局部深度对比度"""
    global_std = np.std(depth_img)
    global_var = np.var(depth_img)

    window_size = 15
    kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
    local_mean = cv2.filter2D(depth_img.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((depth_img - local_mean) ** 2, -1, kernel)
    mean_local_var = np.mean(local_var)

    return {
        "global_std": global_std,
        "global_var": global_var,
        "mean_local_var": mean_local_var
    }

def compute_edge_sharpness(depth_img):
    """使用拉普拉斯算子计算边缘清晰度"""
    laplacian = cv2.Laplacian(depth_img, cv2.CV_64F)
    edge_magnitude = np.abs(laplacian)
    edge_score = np.mean(edge_magnitude)
    return {
        "edge_mean": edge_score,
        "edge_map": edge_magnitude
    }

# def compute_edge_sharpness(depth_img):
#     """使用 Sobel 梯度计算边缘清晰度"""
#     sobelx = cv2.Sobel(depth_img, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(depth_img, cv2.CV_64F, 0, 1, ksize=3)
#     edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
#     edge_score = np.mean(edge_magnitude)
#     return {
#         "edge_mean": edge_score,
#         "edge_map": edge_magnitude
#     }

def analyze_depth_images(folder):
    """
    分析指定文件夹中的所有深度图像，返回对比度与边缘清晰度的平均指标。
    
    参数:
        folder (str): 包含深度图的文件夹路径。
    
    返回:
        dict: 包含平均全局标准差、方差、局部方差及边缘清晰度的指标。
    """
    image_paths = sorted(glob(os.path.join(folder, "eval_depth*.png")))

    if not image_paths:
        raise FileNotFoundError("没有找到任何图像，请检查路径是否正确。")

    total_metrics = {
        "global_std": [],
        "global_var": [],
        "mean_local_var": [],
        "edge_mean": []
    }

    for path in image_paths:
        try:
            depth_img = load_depth_image(path)
            contrast = compute_depth_contrast(depth_img)
            edge = compute_edge_sharpness(depth_img)

            total_metrics["global_std"].append(contrast["global_std"])
            total_metrics["global_var"].append(contrast["global_var"])
            total_metrics["mean_local_var"].append(contrast["mean_local_var"])
            total_metrics["edge_mean"].append(edge["edge_mean"])

            # print(f"✅ {os.path.basename(path)} 分析完成。")
            # # print(f"▶ 平均标准差（深度对比度）：{contrast['global_std']:.2f}")
            # # print(f"▶ 平均方差：{contrast['global_var']:.2f}")
            # print(f"▶ 局部方差：{contrast['mean_local_var']:.2f}")
            # print(f"▶ 边缘清晰度（Laplacian）：{edge['edge_mean']:.2f}")
            # print("---------------------------------------------------------")
        except Exception as e:
            print(f"⚠️ 处理图像 {path} 出错：{e}")

    # 计算平均指标
    avg_metrics = {
        "avg_global_std": float(np.mean(total_metrics["global_std"])),
        "avg_global_var": float(np.mean(total_metrics["global_var"])),
        "avg_mean_local_var": float(np.mean(total_metrics["mean_local_var"])),
        "avg_edge_mean": float(np.mean(total_metrics["edge_mean"]))
    }

    print("\n📊 所有图像的平均指标：")
    # print(f"▶ 平均全局标准差（深度对比度）：{avg_metrics['avg_global_std']:.2f}")
    # print(f"▶ 平均全局方差：{avg_metrics['avg_global_var']:.2f}")
    print(f"▶ 平均局部方差：{avg_metrics['avg_mean_local_var']:.2f}")
    print(f"▶ 平均边缘清晰度（Laplacian）：{avg_metrics['avg_edge_mean']:.2f}")

    return avg_metrics




@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        print('psnr:', metrics_dict['psnr'])
        print('ssim:', metrics_dict['ssim'])
        print('lpips:', metrics_dict['lpips'])
        avg_metrics = analyze_depth_images(self.render_output_path)
        benchmark_info.update(avg_metrics)
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
