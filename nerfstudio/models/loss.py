import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_edge_map(image):  # image: (1, 3, H, W)
    # 转为灰度
    gray = 0.2989 * image[:,0:1] + 0.5870 * image[:,1:2] + 0.1140 * image[:,2:3]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).reshape(1,1,3,3)
    sobel_y = sobel_x.transpose(-1, -2)

    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    edge_map = torch.sqrt(edge_x**2 + edge_y**2)  # Shape: (1,1,H,W)
    edge_map = edge_map / (edge_map.max() + 1e-8)
    return edge_map  # 值域 [0, 1]，越接近1越边缘


def sobel_edge_map(img):
    """
    输入: img [B, C, H, W]
    输出: edge_map [B, C, H, W]
    """
    sobel_kernel_x = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=torch.float32)
    sobel_kernel_y = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=torch.float32)

    # 升维为 [1, 1, 3, 3]，再拓展为 [C, 1, 3, 3]
    sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)

    device = img.device
    C = img.shape[1]  # 通道数

    sobel_kernel_x = sobel_kernel_x.to(device).repeat(C, 1, 1, 1)
    sobel_kernel_y = sobel_kernel_y.to(device).repeat(C, 1, 1, 1)

    # 使用 group convolution 分别对每个通道进行卷积
    grad_x = F.conv2d(img, sobel_kernel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_kernel_y, padding=1, groups=C)

    # 计算梯度幅值
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # 避免 sqrt(0)
    return edge_magnitude

def sobel_loss(pred, target, loss_type='l2'):
    """
    Sobel 边缘损失函数
    pred, target: [B, C, H, W]
    loss_type: 'l1' or 'l2'
    """
    pred = pred.reshape(-1, 1, 3, 3)
    target = target.reshape(-1, 1, 3, 3)
    pred_edge = sobel_edge_map(pred)
    target_edge = sobel_edge_map(target)

    if loss_type == 'l1':
        return torch.mean(torch.abs(pred_edge - target_edge))
    elif loss_type == 'l2':
        return torch.mean((pred_edge - target_edge) ** 2)
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'")

class S3IM(torch.nn.Module):
    def __init__(self, repeat_time=1, patch_height=64, size_average=True):
        super().__init__()
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.size_average = size_average

    def forward(self, src_vec, tar_vec, mask=None):
        batch_size = len(tar_vec)
        indices = torch.cat([
            torch.randperm(batch_size) if i > 0 else torch.arange(batch_size)
            for i in range(self.repeat_time)
        ]).to(src_vec.device)

        tar_all = tar_vec[indices]
        src_all = src_vec[indices]

        tar_patch = tar_all.transpose(0, 1).reshape(1, 1, self.patch_height, -1)
        src_patch = src_all.transpose(0, 1).reshape(1, 1, self.patch_height, -1)

        # 转换为频域
        tar_fft = torch.fft.fft2(tar_patch)
        src_fft = torch.fft.fft2(src_patch)

        # 使用复数模长（magnitude）作为特征比较
        tar_mag = torch.abs(tar_fft)
        src_mag = torch.abs(src_fft)

        # 计算频域 L2 损失
        loss = F.mse_loss(src_mag, tar_mag, reduction='mean' if self.size_average else 'none')
        return loss
    
class S3IM_HighFreqWeighted(torch.nn.Module):
    def __init__(self, repeat_time=10, patch_height=64, size_average=True, highfreq_boost=5.0):
        super().__init__()
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.size_average = size_average
        self.highfreq_boost = highfreq_boost  # 高频增强因子

    def forward(self, src_vec, tar_vec, mask=None):
        batch_size = len(tar_vec)
        indices = torch.cat([
            torch.randperm(batch_size) if i > 0 else torch.arange(batch_size)
            for i in range(self.repeat_time)
        ]).to(src_vec.device)

        tar_all = tar_vec[indices]
        src_all = src_vec[indices]

        tar_patch = tar_all.transpose(0, 1).reshape(1, 3, self.patch_height, -1)
        src_patch = src_all.transpose(0, 1).reshape(1, 3, self.patch_height, -1)

        # 频域转换
        tar_fft = torch.fft.fft2(tar_patch)
        src_fft = torch.fft.fft2(src_patch)

        tar_mag = torch.abs(tar_fft)
        src_mag = torch.abs(src_fft)

        # 构建频率权重
        B, C, H, W = tar_mag.shape
        fy = torch.fft.fftshift(torch.fft.fftfreq(H)).to(tar_mag.device)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W)).to(tar_mag.device)
        freq_y, freq_x = torch.meshgrid(fy, fx, indexing='ij')
        freq_radius = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        freq_weight = (freq_radius / freq_radius.max()) ** 2
        freq_weight = 1.0 + self.highfreq_boost * freq_weight
        freq_weight = freq_weight[None, None, :, :]

        # 空间边缘掩码（额外增强边缘区域）
        with torch.no_grad():
            edge_map = compute_edge_map(tar_patch)
        edge_weight = 1.0 + edge_map * self.highfreq_boost  # (1,1,H,W)

        # 加权差异损失
        loss_map = torch.abs(src_mag - tar_mag) * freq_weight * edge_weight
        loss = loss_map.mean() if self.size_average else loss_map

        return loss


import torch
import torch.nn as nn


class EdgeSharpnessLoss1(nn.Module):
    def __init__(self, mode='sobel', size_average=True):
        super(EdgeSharpnessLoss1, self).__init__()
        self.mode = mode
        self.size_average = size_average
        self.mse_loss = nn.MSELoss(reduction='none')  # 使用 reduction='none' 以获取每个元素的损失

    def forward(self, pred, gt):
        # 计算每个元素的 MSE 损失
        loss = self.mse_loss(pred, gt)

        # 获取损失的形状
        loss_shape = loss.shape

        # 将损失展平为一维
        loss_flat = loss.view(-1)

        # 计算总元素数量
        num_elements = loss_flat.numel()

        # 随机选择一半的索引
        indices = torch.randperm(num_elements)[:num_elements // 2]

        # 将选中的损失乘以 2
        loss_flat[indices] *= 2

        # 将损失重新调整为原始形状
        loss = loss_flat.view(loss_shape)

        # 根据 size_average 参数计算最终损失
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


import torch
import torch.nn as nn
import torch.nn.functional as F


# 如果你已有 compute_edge_map 实现，直接导入；否则自行实现
def compute_edge_map(img):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return edge.mean(dim=1, keepdim=True)

class S3IM_HighFreqWeighted(nn.Module):
    def __init__(self, repeat_time=10, patch_height=64, size_average=True):
        super().__init__()
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.size_average = size_average

    def forward(self, src_vec, tar_vec, highfreq_boost=5.0, mask=None):
        batch_size = tar_vec.size(0)
        indices = torch.cat([
            torch.randperm(batch_size) if i>0 else torch.arange(batch_size)
            for i in range(self.repeat_time)
        ], dim=0).to(src_vec.device)
        tar_all = tar_vec[indices]
        src_all = src_vec[indices]

        C = tar_vec.size(1)
        N = tar_all.size(0)
        L = N // self.patch_height
        tar_patch = tar_all.transpose(0,1).reshape(1, 1, self.patch_height, -1)
        src_patch = src_all.transpose(0,1).reshape(1, 1, self.patch_height, -1)

        tar_fft = torch.fft.fft2(tar_patch)
        src_fft = torch.fft.fft2(src_patch)
        tar_mag = torch.abs(tar_fft)
        src_mag = torch.abs(src_fft)

        H, W = tar_mag.shape[-2:]
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=1.0)).to(tar_mag.device)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=1.0)).to(tar_mag.device)
        freq_y, freq_x = torch.meshgrid(fy, fx, indexing='ij')
        freq_radius = torch.sqrt(freq_x**2 + freq_y**2)
        weight = (freq_radius / freq_radius.max())**2
        freq_weight = (1.0 + highfreq_boost * weight).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            edge_map = compute_edge_map(tar_patch)
        edge_weight = 1.0 + highfreq_boost * edge_map

        loss_map = torch.abs(src_mag - tar_mag)**2 * freq_weight * edge_weight
        return loss_map.mean() if self.size_average else loss_map

import torch
import torch.nn as nn
import torch.nn.functional as F

total_steps = 30000
import math


# 三种策略的 alpha 曲线
def cosine_annealing(step, total_steps):
    return 0.5 * (1 + math.cos(math.pi * step / total_steps))

def sigmoid_schedule(step, total_steps, midpoint=15000, k=0.0005):
    return 1 / (1 + math.exp(k * (step - midpoint)))

def linear_decay(step, total_steps):
    return max(0.0, 1 - step / total_steps)


import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_annealing(step, total_steps, lr_min=0.0, lr_max=1.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + torch.cos(torch.tensor(step / total_steps * 3.1415926535)))

class EdgeSharpnessLoss(nn.Module):
    def __init__(self, mode='laplacian', size_average=True, center_weight=-2.0):
        super().__init__()
        self.mode = mode
        self.size_average = size_average
        self.center_weight = center_weight
        self.count = 1  # 用于动态调节

    def dynamic_center_weight(self, total_epochs=30000, min_weight=-4.0, max_weight=-9.0):
        progress = min(self.count / total_epochs, 1.0)
        return (1 - progress) * max_weight + progress * min_weight

    def forward(self, pred, gt):
        # 输入维度为 [N, 3]
        pred = pred.transpose(0, 1).unsqueeze(0)  # [1, 3, N]
        gt = gt.transpose(0, 1).unsqueeze(0)      # [1, 3, N]

        # Laplacian on RGB channels
        # rgb_loss = 0.0
        # for c in range(3):
        #     pred_c = pred[:, c:c+1, :]
        #     gt_c = gt[:, c:c+1, :]
        #     edge_pred = self._laplacian_1d(pred_c)
        #     edge_gt = self._laplacian_1d(gt_c)
        #     diff = torch.norm(edge_pred - edge_gt, p=2, dim=1)  # [1, N]
        #     rgb_loss += diff

        # rgb_loss = rgb_loss / 3

        # Laplacian on grayscale
        pred_gray = pred.mean(dim=1, keepdim=True)  # [1, 1, N]
        gt_gray = gt.mean(dim=1, keepdim=True)
        edge_pred_gray = self._laplacian_1d(pred_gray)
        edge_gt_gray = self._laplacian_1d(gt_gray)
        gray_loss = torch.norm(edge_pred_gray - edge_gt_gray, p=2, dim=1)  # [1, N]
        # import pdb;pdb.set_trace()
        # 融合损失
        # alpha = cosine_annealing(self.count, total_steps=30000)
        fused_loss = gray_loss
        # print(fused_loss)
        self.count += 1
        return fused_loss.mean() if self.size_average else fused_loss
    
    def _laplacian_1d(self, x):
        # 对原始信号做两次卷积（stride=1, no padding）
        x_even = x[:, :, 0::2]  # 取偶数位
        x_odd = x[:, :, 1::2]   # 取奇数位

        # 拼接成相邻对：x_pair 的形状是 [B, C, L//2, 2]
        x_pair = torch.stack([x_even, x_odd], dim=-1)  # [B, C, L//2, 2]

        # 计算偶数位置的差（x[i+1] - x[i]）
        diff_even = (x_pair[..., 1] + x_pair[..., 0]) / 2.  # [B, C, L//2]
        # 计算奇数位置的差（x[i] - x[i-1]）
        # diff_odd = x_pair[..., 1] - x_pair[..., 0]   # [B, C, L//2]

        # # 创建输出张量
        # out = torch.zeros_like(x)

        # # 将差值转换为 0 和 1：大于 0 为 1，小于等于 0 为 0
        # # 注意此处不使用 abs，而是根据正负号判断
        # # out[:, :, 0::2] = diff_even >= 0.04  # 偶数位
        # # out[:, :, 1::2] = diff_odd <= 0.04   # 奇数位（方向相反）
        # out[:, :, 0::2] = diff_even.abs()  # 偶数位
        # out[:, :, 1::2] = diff_odd.abs()   # 奇数位（方向相反）
        # import pdb;pdb.set_trace()
        return diff_even.abs() 


import torch
import torch.nn as nn
import torch.nn.functional as F

# class EdgeSharpMask(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, pred, gt, batch):
#         batch_ids = batch['indices'][:, 0].unique()
#         size_len = 4096 // len(batch_ids) // 3 * 3  # 保证 patch 维度是 3 的整数倍

#         pred_edge = []
#         gt_edge = []

#         for batch_id in batch_ids:
#             mask = batch['indices'][:, 0] == batch_id
#             pred_edge.append(pred[mask][:size_len])
#             gt_edge.append(gt[mask][:size_len])

#         # 拼接并重塑形状： [N, patch, rgb] -> [N, 3, 3]
#         pred_edge = torch.cat(pred_edge, dim=0).reshape(-1, 3, 3)
#         gt_edge = torch.cat(gt_edge, dim=0).reshape(-1, 3, 3)

#         # 转置为 [N, rgb, patch]，以便在每个颜色通道上执行 1D 卷积
#         pred_edge = pred_edge.permute(0, 2, 1)  # shape: [N, 3, 3]
#         gt_edge = gt_edge.permute(0, 2, 1)

#         # 定义每个通道的拉普拉斯核 [out_channels=3, in_channels=1, kernel_size=3]
#         kernel = torch.tensor([[-1., 2., -1.]], device=pred.device).repeat(3, 1, 1)  # shape: [3, 1, 3]

#         # 使用 group convolution (groups=3 表示每个通道独立卷积)
#         pred_laplace = F.conv1d(pred_edge, kernel, padding=0, groups=3)  # shape: [N, 3, 1]
#         gt_laplace = F.conv1d(gt_edge, kernel, padding=0, groups=3)
#         # import pdb;pdb.set_trace()
#         # 去除最后一维
#         pred_laplace = pred_laplace.squeeze(-1)  # shape: [N, 3]
#         gt_laplace = gt_laplace.squeeze(-1)
#         # import pdb;pdb.set_trace()
#         # 计算 L1 损失
#         loss = F.l1_loss(pred_laplace, gt_laplace)

#         return loss

import torch
import torch.nn as nn

class EdgeSharpMask(nn.Module):
    def __init__(self, mode='laplacian', size_average=True, center_weight=-2.0):
        super().__init__()
        self.mode = mode
        self.size_average = size_average
        self.center_weight = center_weight
        self.count = 1
        self.mse_loss = nn.MSELoss()
        self.use_mask = False

    def forward(self, pred, gt):
        # 输入维度为 [N, 3]
        pred = pred.transpose(0, 1).unsqueeze(0)  # [1, 3, N]
        gt = gt.transpose(0, 1).unsqueeze(0)      # [1, 3, N]

        # Laplacian on grayscale
        pred_gray = pred.mean(dim=1, keepdim=True)  # [1, 1, N]
        gt_gray = gt.mean(dim=1, keepdim=True)

        if self.use_mask:
            background_mask = gt_gray != 1.00
            edge_pred_gray = self._laplacian_1d(pred_gray)
            edge_gt_gray = self._laplacian_1d(gt_gray)
            fused_loss = self.mse_loss(edge_pred_gray * background_mask, edge_gt_gray * background_mask)
        else:
            edge_pred_gray = self._laplacian_1d(pred_gray)
            edge_gt_gray = self._laplacian_1d(gt_gray)
            fused_loss = self.mse_loss(edge_pred_gray, edge_gt_gray)

        self.count += 1
        return fused_loss

    def _laplacian_1d(self, x):
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        x_pair = torch.stack([x_even, x_odd], dim=-1)
        diff_even = x_pair[..., 1] - x_pair[..., 0]
        diff_odd = x_pair[..., 1] - x_pair[..., 0]
        out = torch.zeros_like(x)

        if self.use_mask:
            out[:, :, 0::2] = (diff_even > 0).float()
            out[:, :, 1::2] = (diff_odd < 0).float()
        else:
            out[:, :, 0::2] = diff_even.abs()
            out[:, :, 1::2] = diff_odd.abs()
        return out.abs()

    def pairwise_diff_loss(self, pred, gt, indices):
        """
        计算每个 batch 内所有像素对之间的灰度差值差异，返回所有 batch 的平均损失。

        参数:
            pred: [N, 3] 预测值
            gt: [N, 3] GT 值
            indices: [N, 3, 3]，第一个维度为像素数，indices[i, 0, 0] 表示第 i 个像素所在的 batch id

        返回:
            标量 loss
        """
        # 转为灰度图 [N, 1]
        pred_gray = pred.mean(dim=1, keepdim=True)  # [N, 1]
        gt_gray = gt.mean(dim=1, keepdim=True)      # [N, 1]

        # 提取每个像素的 batch id
        batch_ids = indices[:, 0].long()  # [N]

        unique_batches = torch.unique(batch_ids)
        losses = []

        for batch_id in unique_batches:
            # 找到当前 batch 的索引
            mask = (batch_ids == batch_id)
            pred_batch = pred_gray[mask]  # [M, 1]
            gt_batch = gt_gray[mask]      # [M, 1]

            if pred_batch.size(0) < 2:
                continue  # 忽略只有一个像素的 batch

            # 计算差值矩阵 [M, M]
            pred_diff = torch.abs(pred_batch - pred_batch.T)
            gt_diff = torch.abs(gt_batch - gt_batch.T)

            # 计算当前 batch 的均方误差
            loss = nn.functional.mse_loss(pred_diff.mean(dim=1), gt_diff.mean(dim=1))
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 所有 batch 的平均
        return torch.stack(losses).mean()

# class EdgeSharpMask(nn.Module):
#     def __init__(self, mode='laplacian', size_average=True, center_weight=-2.0):
#         super().__init__()
#         self.mode = mode
#         self.size_average = size_average
#         self.center_weight = center_weight
#         self.count = 1  # 用于动态调节
#         self.mse_loss = nn.MSELoss()
#         self.use_mask = False

#     def forward(self, pred, gt):
#         # 输入维度为 [N, 3]
#         pred = pred.transpose(0, 1).unsqueeze(0)  # [1, 3, N]
#         gt = gt.transpose(0, 1).unsqueeze(0)      # [1, 3, N]

#         # Laplacian on grayscale
#         pred_gray = pred.mean(dim=1, keepdim=True)  # [1, 1, N]
#         gt_gray = gt.mean(dim=1, keepdim=True)
#         if self.use_mask:
#             background_mask = gt_gray != 1.00
#             edge_pred_gray = self._laplacian_1d(pred_gray)
#             edge_gt_gray = self._laplacian_1d(gt_gray)
            
#             fused_loss = self.mse_loss(edge_pred_gray* background_mask, edge_gt_gray* background_mask) 
#             # print("***********")
#         else:
#             # background_mask = gt_gray != 1.00
#             edge_pred_gray = self._laplacian_1d(pred_gray)
#             edge_gt_gray = self._laplacian_1d(gt_gray)
            
#             # fused_loss = self.mse_loss(edge_pred_gray* background_mask, edge_gt_gray* background_mask) 
#             fused_loss = self.mse_loss(edge_pred_gray, edge_gt_gray) 
#             # fused_loss = self.mse_loss(edge_pred_gray, edge_gt_gray) * 0.1
#         self.count += 1
#         return fused_loss
    
#     def _laplacian_1d(self, x):
#         # 对原始信号做两次卷积（stride=1, no padding）
#         x_even = x[:, :, 0::2]  # 取偶数位
#         x_odd = x[:, :, 1::2]   # 取奇数位

#         # 拼接成相邻对：x_pair 的形状是 [B, C, L//2, 2]
#         x_pair = torch.stack([x_even, x_odd], dim=-1)  # [B, C, L//2, 2]

#         # 计算偶数位置的差（x[i+1] - x[i]）
#         diff_even = x_pair[..., 1] - x_pair[..., 0]  # [B, C, L//2]
#         # 计算奇数位置的差（x[i] - x[i-1]）
#         diff_odd = x_pair[..., 1] - x_pair[..., 0]   # [B, C, L//2]

#         # 创建输出张量
#         out = torch.zeros_like(x)

#         # 将差值转换为 0 和 1：大于 0 为 1，小于等于 0 为 0
#         # 注意此处不使用 abs，而是根据正负号判断
#         if self.use_mask:
#             out[:, :, 0::2] = (diff_even > 0).float()  # 偶数位
#             out[:, :, 1::2] = (diff_odd < 0).float()   # 奇数位（方向相反）
#         else:
#             out[:, :, 0::2] = diff_even.abs()  # 偶数位
#             out[:, :, 1::2] = diff_odd.abs()   # 奇数位（方向相反）
#         # out = x[:, :, 1:] - x[:, :, :-1]
#         return out.abs()
    
    

    # def _laplacian_1d(self, x):
    #     # 对原始信号做两次卷积（stride=1, no padding）
    #     x_even = x[:, :, 0::2]  # 取偶数位
    #     x_odd = x[:, :, 1::2]   # 取奇数位

    #     # 拼接成相邻对：x_pair 的形状是 [B, C, L//2, 2]
    #     x_pair = torch.stack([x_even, x_odd], dim=-1)  # [B, C, L//2, 2]

    #     # 计算偶数位置的差（x[i+1] - x[i]）
    #     diff_even = x_pair[..., 1] - x_pair[..., 0]  # [B, C, L//2]
    #     # 计算奇数位置的差（x[i] - x[i-1]）
    #     diff_odd = x_pair[..., 1] - x_pair[..., 0]   # [B, C, L//2]
    #     # import pdb;pdb.set_trace()
    #     # 重新插入到原始位置
    #     out = torch.zeros_like(x)

    #     # 偶数位置放 diff
    #     out[:, :, 0::2] = diff_even.abs()
    #     # 奇数位置放 diff
    #     out[:, :, 1::2] = (-diff_odd).abs()  # 注意方向
    #     # import pdb;pdb.set_trace()
    #     return out
# class EdgeSharpnessLoss(nn.Module):
#     def __init__(self, mode='laplacian', size_average=True, center_weight=-8.0):
#         super().__init__()
#         self.mode = mode
#         self.size_average = size_average
#         self.center_weight = center_weight
#         self.count = 1

#         # 可训练的融合权重参数，初始化为 0（sigmoid(0)=0.5）
#         # self.alpha_param = nn.Parameter(torch.tensor(0.0))

#     def dynamic_center_weight(self, total_epochs=30000, min_weight=-4.0, max_weight=-9.0):
#         """动态变化的中心权重（可选项）"""
#         progress = min(self.count / total_epochs, 1.0)
#         return (1 - progress) * max_weight + progress * min_weight

#     def forward(self, pred, gt):
#         # batch_size = gt.size(0)
#         # indices = torch.cat([
#         #     torch.randperm(batch_size) if i>0 else torch.arange(batch_size)
#         #     for i in range(4)
#         # ], dim=0).to(pred.device)
#         # gt = gt[indices]
#         # pred = pred[indices]
#         # 处理 2D 输入为图像格式
#         if pred.dim() == 2 and pred.size(1) == 3:
#             N = pred.size(0)
#             L = int(N ** 0.5)
#             pred = pred.view(1, L, L, 3).permute(0, 3, 1, 2)
#             gt = gt.view(1, L, L, 3).permute(0, 3, 1, 2)

#         # Laplacian on RGB channels
#         rgb_loss = 0.0
#         for c in range(3):  # RGB 通道
#             pred_c = pred[:, c:c+1, :, :]
#             gt_c = gt[:, c:c+1, :, :]
#             edge_pred = self._laplacian(pred_c)
#             edge_gt = self._laplacian(gt_c)
#             diff = torch.norm(edge_pred - edge_gt, p=2, dim=1)  # [B, H, W]
#             rgb_loss += diff

#         rgb_loss = rgb_loss / 3.0

#         # Laplacian on grayscale image
#         pred_gray = pred.mean(dim=1, keepdim=True)
#         gt_gray = gt.mean(dim=1, keepdim=True)
#         edge_pred_gray = self._laplacian(pred_gray)
#         edge_gt_gray = self._laplacian(gt_gray)
#         gray_loss = torch.norm(edge_pred_gray - edge_gt_gray, p=2, dim=1)  # [B, H, W]

#         # 使用可训练的 alpha 参数进行融合
#         # alpha = torch.sigmoid(self.alpha_param)  # 保证在 [0, 1]
#         # fused_loss = gray_loss * 1.0 + rgb_loss * 1.0= 
#         alpha = cosine_annealing(self.count, 30000)
#         fused_loss = gray_loss * alpha * 2 + rgb_loss * (1 - alpha) * 0.5
#         # fused_loss = gray_loss * 2 + rgb_loss * 0.5
#         # if self.count <= 2000:
#         #     fused_loss = gray_loss * 2.0 + rgb_loss * 0.0
#         # else:
#         #     fused_loss = gray_loss * 0.0 + rgb_loss * 0.1


#         self.count += 1
#         return fused_loss.mean() if self.size_average else fused_loss

#     def _laplacian(self, img):
#         # 可选：动态中心权重
#         center_weight = int(self.dynamic_center_weight())
#         # center_weight = 3
#         # print(self.count, center_weight)
#         lap = torch.tensor([
#             [1, 1, 1],
#             [1, center_weight, 1],
#             [1, 1, 1]
#         ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
#         # lap1 = torch.tensor([
#         #     [0, 1, 0],
#         #     [1, -1, 1],
#         #     [0, 1, 0]
#         # ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
#         return F.conv2d(img, lap, padding=1).abs()

    # def _laplacian(self, img):
    #     # 定义多尺度空洞率（指数增长策略）
    #     dilation_rates = [1, 2] # 动态中心权重
    #     center_weight = int(self.dynamic_center_weight())
        
    #     # 创建基础卷积核（保持中心权重动态调整）
    #     lap = torch.tensor([
    #         [0, 1, 0],
    #         [1, center_weight, 1],
    #         [0, 1, 0]
    #     ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        
    #     # 多尺度特征融合
    #     features = []
    #     for rate in dilation_rates:
    #         # 动态计算padding保持输出尺寸一致
    #         padding = rate if rate > 1 else 1
            
    #         # 创建不同扩张率的空洞卷积
    #         conv = F.conv2d(
    #             img,
    #             lap,
    #             padding=padding,
    #             dilation=rate
    #         ).abs()
            
    #         features.append(conv)
        
    #     # 合并多尺度特征（可替换为加权求和或拼接）
    #     # return torch.cat(features, dim=1)
    #     # import pdb;pdb.set_trace()
    #     return sum(features)
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EdgeSharpnessLoss(nn.Module):
#     def __init__(self, mode='sobel', size_average=True, center_weight=-8.0):
#         super().__init__()
#         self.mode = mode
#         self.size_average = size_average
#         self.repeat_time = 16
#         self.center_weight = center_weight  # 中心权重可调
#         self.count = 1

#     def dynamic_center_weight(self, total_epochs=30000, min_weight=-2.0, max_weight=-9.0):
#         """
#         动态调整拉普拉斯中心权重值。
        
#         Args:
#             epoch (int): 当前训练轮数。
#             total_epochs (int): 总训练轮数。
#             min_weight (float): 初始时的中心权重（通常较小，如 -1）。
#             max_weight (float): 最终时的中心权重（通常较大，如 -8）。
            
#         Returns:
#             float: 当前 epoch 对应的中心权重值。
#         """
#         progress = min(self.count / total_epochs, 1.0)  # 防止超过1
#         return (1 - progress) * max_weight + progress * min_weight

#     def forward(self, pred, gt):
#         if pred.dim() == 2 and pred.size(1) == 3:
#             N = pred.size(0)
#             L = int(N ** 0.5)
#             pred = pred.view(1, L, L, 3).permute(0, 3, 1, 2)
#             gt = gt.view(1, L, L, 3).permute(0, 3, 1, 2)

#         pred_gray = pred.mean(dim=1, keepdim=True)
#         gt_gray = gt.mean(dim=1, keepdim=True)

#         edge_pred = self._laplacian(pred_gray)
#         edge_gt = self._laplacian(gt_gray)

#         diff = torch.norm(edge_pred - edge_gt, p=2, dim=1)  # 沿 channel 维度计算
#         self.count += 1
#         return diff.mean() if self.size_average else diff

#     def _laplacian(self, img):
#         print(self.count, int(self.dynamic_center_weight()))
#         lap = torch.tensor([
#             [0, 1, 0],
#             [1, int(self.dynamic_center_weight()), 1],
#             [0, 1, 0]
#         ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
#         return F.conv2d(img, lap, padding=1).abs()

# class EdgeSharpnessLoss(nn.Module):
#     def __init__(self, mode='laplacian', size_average=True):
#         super().__init__()
#         self.mode = mode
#         self.size_average = size_average
#         self.repeat_time = 16
#         self.count = 0

#     def forward(self, pred, gt):
#         # batch_size = gt.size(0)
#         # indices = torch.cat([
#         #     torch.randperm(batch_size) if i>0 else torch.arange(batch_size)
#         #     for i in range(self.repeat_time)
#         # ], dim=0).to(gt.device)
#         # pred = pred[indices]
#         # gt = gt[indices]
#         if pred.dim()==2 and pred.size(1)==3:
#             N = pred.size(0)
#             L = int(N**0.5)
#             pred = pred.view(1, L, L, 3).permute(0,3,1,2)
#             gt   = gt.view(1, L, L, 3).permute(0,3,1,2)

#         pred_gray = pred.mean(dim=1, keepdim=True)
#         gt_gray = gt.mean(dim=1, keepdim=True)

#         edge_pred = self._laplacian(pred_gray)
#         edge_gt   = self._laplacian(gt_gray)
#         # diff = torch.abs(edge_pred - edge_gt)
#         diff = torch.norm(edge_pred - edge_gt, p=2, dim=1)  # 沿特征维度计算
#         self.count += 1

#         return diff.mean() if self.size_average else diff

#     def _laplacian(self, img):
#         lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=img.dtype, device=img.device).view(1,1,3,3)
#         return F.conv2d(img, lap, padding=1).abs()

class CombinedLoss(nn.Module):
    """集成高频加权 S3IM 和边缘锐化损失，自动调整权重"""
    def __init__(self,
                 repeat_time=10, patch_height=64,
                 init_hf_boost=5.0,
                 edge_mode='sobel',
                 size_average=True):
        super().__init__()
        self.s3im = S3IM_HighFreqWeighted(repeat_time, patch_height, size_average)
        self.edge = EdgeSharpnessLoss(edge_mode, size_average)
        # 可学习参数（使用 sigmoid 映射控制范围在 0~1）
        self.weight_param = nn.Parameter(torch.tensor([0.5]))
        self.init_hf_boost = init_hf_boost

    def forward(self, src_vec, tar_vec, mask=None):
        weight = torch.sigmoid(self.weight_param)  # 0~1
        hf_boost = self.init_hf_boost * weight.item()
        loss_hf = self.s3im(src_vec, tar_vec, highfreq_boost=hf_boost, mask=mask)
        loss_edge = self.edge(src_vec, tar_vec)
        return (1 - weight) * loss_edge + weight * loss_hf
    
class BrightnessLoss(nn.Module):
    """适用于 [N,3] 或 [B,3,H,W] 的亮度损失"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        if pred.dim() == 2 and pred.size(1) == 3:
            # [N,3] 形式，直接计算灰度
            pred_gray = (0.299 * pred[:,0] + 0.587 * pred[:,1] + 0.114 * pred[:,2])
            gt_gray   = (0.299 * gt[:,0] + 0.587 * gt[:,1] + 0.114 * gt[:,2])
            pred_mean = pred_gray.mean()
            gt_mean   = gt_gray.mean()
            return torch.abs(pred_mean - gt_mean)

        elif pred.dim() == 4 and pred.size(1) == 3:
            # [B,3,H,W] 形式
            pred_gray = (0.299 * pred[:,0:1] + 0.587 * pred[:,1:2] + 0.114 * pred[:,2:3])
            gt_gray   = (0.299 * gt[:,0:1] + 0.587 * gt[:,1:2] + 0.114 * gt[:,2:3])
            pred_mean = pred_gray.view(pred_gray.size(0), -1).mean(dim=1)
            gt_mean   = gt_gray.view(gt_gray.size(0), -1).mean(dim=1)
            return torch.mean(torch.abs(pred_mean - gt_mean))
        
        else:
            raise ValueError("Unsupported input shape: expected [N,3] or [B,3,H,W]")


import torch
import torch.fft


class AdaptiveFrequencyLoss(nn.Module):
    def __init__(self, size=64, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('freq_filter', self._generate_filter(size))
        self.weight = 0.0  # 动态调整
    
    def _generate_filter(self, size):
        freq_band = torch.fft.fftfreq(size).repeat(size, 1)
        freq_map = torch.sqrt(freq_band**2 + freq_band.T**2)
        # 高斯高通滤波器（减少噪声敏感度）
        filter = 1 - torch.exp(-(freq_map**2) / (2*(0.2**2)))  # σ=0.2控制截止频率
        return filter.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    def forward(self, pred, target):
        # 输入形状调整
        pred = pred.reshape(-1, 3, 64, 64)
        target = target.reshape(-1, 3, 64, 64)
        
        # 计算FFT幅度谱
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # 高频误差加权（重点抑制低频误差主导）
        error = (pred_amp - target_amp) ** 2
        weighted_error = error * self.freq_filter
        
        # 通道加权（可选）
        weighted_error = weighted_error.mean(dim=1, keepdim=True)
        
        if self.reduction == 'mean':
            return self.weight * weighted_error.mean()
        return self.weight * weighted_error.sum()
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# class EdgeSharpnessLoss(nn.Module):
#     def __init__(self):
#         super(EdgeSharpnessLoss, self).__init__()
#         # 定义 Laplacian kernel，形状为 [1, 1, 3, 3]
#         self.lap_kernel = torch.tensor([
#             [-1, -1, -1],
#             [-1,  8, -1],
#             [-1, -1, -1]
#         ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     def forward(self, pred, gt):
#         """
#         pred, gt: 形状为 [B, C, H, W]，C = 3，RGB图像
#         """
#         # 转换为灰度图像 [B, 1, H, W]
#         pred_gray = pred.mean(dim=1, keepdim=False).reshape(-1, 1, 3, 3)  # [B, 1, H, W]
#         gt_gray = gt.mean(dim=1, keepdim=False).reshape(-1, 1, 3, 3)      # [B, 1, H, W]
#         self.lap_kernel = self.lap_kernel.to(gt_gray.device)
#         # 对灰度图进行 padding（上下左右各1）
#         # import pdb;pdb.set_trace()
#         pred_pad = F.pad(pred_gray, pad=(1, 1, 1, 1), mode='replicate')  # [B, 1, H+2, W+2]
#         gt_pad = F.pad(gt_gray, pad=(1, 1, 1, 1), mode='replicate')

#         # 使用 Laplacian kernel 计算边缘（卷积）
#         pred_edge = F.conv2d(pred_pad, self.lap_kernel, padding=0)
#         gt_edge = F.conv2d(gt_pad, self.lap_kernel, padding=0)
#         # import pdb;pdb.set_trace()
#         # 计算 L1 loss 或 L2 loss（可以根据需求修改）
#         loss = F.mse_loss(pred_edge, gt_edge)

#         return loss
    


def compute_acf(tensor, max_lag=5):
    """
    计算输入张量的自相关函数，输入 shape 为 (C, H, W)，返回 shape 为 (C, 2*max_lag+1, 2*max_lag+1)
    """
    C, H, W = tensor.shape
    acf = torch.zeros(C, 2 * max_lag + 1, 2 * max_lag + 1, device=tensor.device)
    
    for dx in range(-max_lag, max_lag + 1):
        for dy in range(-max_lag, max_lag + 1):
            shifted = torch.roll(tensor, shifts=(dx, dy), dims=(1, 2))
            product = tensor * shifted
            acf[:, dx + max_lag, dy + max_lag] = product.mean(dim=(1, 2))  # mean over spatial dims
    return acf

def acf_loss(gt, pred, indices, max_lag=5):
    """
    gt, pred: Tensor of shape (4096, 3)
    indices: Tensor of shape (4096, 3) → [batch_id, x, y]
    """
    device = gt.device
    N, C = gt.shape
    B = indices[:, 0].max().item() + 1
    H = indices[:, 1].max().item() + 1
    W = indices[:, 2].max().item() + 1
    # import pdb;pdb.set_trace()
    gt_image = torch.zeros(B, C, H, W, device=device)
    pred_image = torch.zeros(B, C, H, W, device=device)
    
    batch_idx = indices[:, 0].long()
    x_idx = indices[:, 1].long()
    y_idx = indices[:, 2].long()
    
    # 填充图像
    gt_image[batch_idx, :, x_idx, y_idx] = gt
    pred_image[batch_idx, :, x_idx, y_idx] = pred

    loss = 0.0
    for b in range(B):
        gt_b = gt_image[b]
        pred_b = pred_image[b]

        acf_gt = compute_acf(gt_b, max_lag)  # shape (C, 2L+1, 2L+1)
        acf_pred = compute_acf(pred_b, max_lag)

        # 使用 L2 loss 计算 ACF 差异
        loss += F.mse_loss(acf_gt, acf_pred)

    return loss / B  # 对 batch 数取平均

from collections import defaultdict

import torch


def sparse_acf_loss(gt, pred, indices, max_lag=5):
    """
    更快的稀疏 ACF 损失函数，避免 O(N^2) 点对构造
    gt, pred: (N, C)
    indices: (N, 3) → [batch_id, x, y]
    """
    device = gt.device
    N, C = gt.shape
    B = indices[:, 0].max().item() + 1

    loss = 0.0

    for b in range(B):
        mask = indices[:, 0] == b
        coords = indices[mask][:, 1:]  # (Nb, 2)
        gt_feats = gt[mask]            # (Nb, C)
        pred_feats = pred[mask]        # (Nb, C)

        Nb = coords.shape[0]

        # Build a position → index lookup table
        pos_to_idx = {}
        for i in range(Nb):
            x, y = coords[i].tolist()
            pos_to_idx[(x, y)] = i

        # 初始化 ACF 累加器
        acf_gt = torch.zeros((2 * max_lag + 1, 2 * max_lag + 1), device=device)
        acf_pred = torch.zeros_like(acf_gt)
        count = torch.zeros_like(acf_gt)

        for i in range(Nb):
            xi, yi = coords[i]
            fi_gt = gt_feats[i]
            fi_pred = pred_feats[i]

            for dx in range(-max_lag, max_lag + 1):
                for dy in range(-max_lag, max_lag + 1):
                    xj = xi + dx
                    yj = yi + dy
                    j = pos_to_idx.get((xj.item(), yj.item()), None)
                    if j is not None:
                        fj_gt = gt_feats[j]
                        fj_pred = pred_feats[j]

                        dot_gt = (fi_gt * fj_gt).sum()
                        dot_pred = (fi_pred * fj_pred).sum()

                        acf_gt[dx + max_lag, dy + max_lag] += dot_gt
                        acf_pred[dx + max_lag, dy + max_lag] += dot_pred
                        count[dx + max_lag, dy + max_lag] += 1

        # 归一化并计算差异
        valid = count > 0
        acf_gt[valid] /= count[valid]
        acf_pred[valid] /= count[valid]

        diff = (acf_gt - acf_pred)[valid]
        loss += (diff ** 2).mean()

    return loss / B


import torch
import torch.nn as nn


class SpatialAwareMSELoss(nn.Module):
    def __init__(self, H=800, W=800, lambda_weight=1.0):
        """
        Args:
            H (int): 图像高度
            W (int): 图像宽度
            lambda_weight (float): 控制空间权重的强度
        """
        super().__init__()
        self.H = H
        self.W = W
        self.lambda_weight = lambda_weight

    def forward(self, pred, gt, indices):
        """
        Args:
            pred (Tensor): shape [N, 3]
            gt (Tensor): shape [N, 3]
            indices (Tensor): shape [N, 3] -> (batch_idx, h, w)

        Returns:
            Tensor: scalar loss
        """
        # 均方误差
        mse = (pred - gt) ** 2  # [N, 3]
        mse = mse.mean(dim=1)   # [N]

        # 位置坐标
        h = indices[:, 1].float()
        w = indices[:, 2].float()

        # 归一化坐标中心为 (0, 0)
        h_norm = (h - self.H / 2) / self.H
        w_norm = (w - self.W / 2) / self.W

        # 计算空间权重
        spatial_weight = 1.0 + self.lambda_weight * (h_norm ** 2 + w_norm ** 2)  # [N]
        # import pdb;pdb.set_trace()
        # 加权 MSE
        weighted_mse = spatial_weight.to(mse.device) * mse  # [N]

        return weighted_mse.mean()

from torch.autograd import Variable
from math import exp
class S3IM(torch.nn.Module):
    def __init__(self, s3im_kernel_size = 4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=64, size_average = True):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)

    
    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor([exp(-(x - s3im_kernel_size//2)**2/float(2*sigma**2)) for x in range(s3im_kernel_size)])
        return gauss/gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
        return s3im_kernel

    # def laplacian(self, s3im_kernel_size, sigma):
    #     x = torch.arange(s3im_kernel_size).float()
    #     laplacian = torch.exp(-torch.abs(x - s3im_kernel_size // 2) / sigma)
    #     return laplacian / laplacian.sum()

    # def create_kernel(self, s3im_kernel_size, channel):
    #     _1D_window = self.laplacian(s3im_kernel_size, sigma=1.0).unsqueeze(1)
    #     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #     s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
    #     return s3im_kernel

    def _ssim(self, img1, img2, s3im_kernel, s3im_kernel_size, channel, size_average = True, s3im_stride=None):
        mu1 = F.conv2d(img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)
        mu2 = F.conv2d(img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu2_sq
        sigma12 = F.conv2d(img1*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def ssim_loss(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)

            if img1.is_cuda:
                s3im_kernel = s3im_kernel.cuda(img1.get_device())
            s3im_kernel = s3im_kernel.type_as(img1)

            self.s3im_kernel = s3im_kernel
            self.channel = channel


        return self._ssim(img1, img2, s3im_kernel, self.s3im_kernel_size, channel, self.size_average, s3im_stride=self.s3im_stride)

    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss