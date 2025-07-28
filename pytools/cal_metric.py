import json
import os

# 是否包含 avg_edge_mean 的统计（默认开启）
include_edge_mean = True

# 设置根目录
# root_dir = "render1"
# root_dir = "render_quick"
root_dir = "render2"

# 存储指标的列表
psnrs = []
ssims = []
lpips = []
edge_means = []

# 遍历 render1 下的所有子文件夹
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    
    # 确保是文件夹
    if os.path.isdir(folder_path):
        json_path = os.path.join(folder_path, "output_nerf_v21.json")
        
        # 检查文件是否存在
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                results = data.get("results", {})
                psnr = results.get("psnr")
                ssim = results.get("ssim")
                lpi = results.get("lpips")

                edge_mean = data.get("avg_edge_mean") if include_edge_mean else None
                print(f"Processing {json_path}:")
                print(psnr, edge_mean)
                # 只有三项主指标都存在时才加入统计
                if None not in (psnr, ssim, lpi):
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    lpips.append(lpi)

                    if edge_mean is not None:
                        edge_means.append(edge_mean)
            except Exception as e:
                print(f"读取 {json_path} 出错：{e}")
        else:
            print(f"{json_path} 不存在")

# 计算平均值
count = len(psnrs)
if count > 0:
    avg_psnr = sum(psnrs) / count
    avg_ssim = sum(ssims) / count
    avg_lpips = sum(lpips) / count

    print(f"总共处理了 {count} 个文件夹")
    print(f"平均 PSNR: {avg_psnr:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 LPIPS: {avg_lpips:.4f}")

    if include_edge_mean and len(edge_means) > 0:
        avg_edge_mean = sum(edge_means) / len(edge_means)
        print(f"平均 avg_edge_mean: {avg_edge_mean:.4f}")
else:
    print("没有找到有效的 output_nerf_v21.json 文件。")