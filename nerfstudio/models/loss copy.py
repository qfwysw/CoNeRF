import torch
import torch.nn.functional as F
import numpy as np

class S3IM(torch.nn.Module):
    def __init__(
        self,
        kernel_size=4,
        stride=4,
        repeat_time=10,
        patch_height=64,
        sigma=1.5,
        k1=0.01,
        k2=0.03,
        eps=1e-8,
        size_average=True,
        high_freq_weight=3.0,
        channels=3  # 添加通道数参数
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.sigma = sigma
        self.eps = eps
        self.size_average = size_average
        self.high_freq_weight = high_freq_weight
        self.channels = channels

        # SSIM constants
        self.k1 = k1
        self.k2 = k2
        
        # Register Gaussian kernel
        self.register_buffer('kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self):
        """创建多通道高斯核"""
        x = torch.arange(self.kernel_size, dtype=torch.float32)
        gauss = torch.exp(-(x - self.kernel_size//2)**2 / (2 * self.sigma**2))
        kernel_1d = gauss / gauss.sum()
        
        # 创建2D核
        kernel_2d = kernel_1d.unsqueeze(0) @ kernel_1d.unsqueeze(1)
        
        # 扩展到所需的通道数
        # shape: [out_channels, in_channels/groups, kernel_size, kernel_size]
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        
        return kernel
    
    def _get_freq_weights(self, height, width, device):
        """Create frequency weights with better numerical stability"""
        freq_y = torch.fft.fftfreq(height, device=device)
        freq_x = torch.fft.fftfreq(width, device=device)
        freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing="ij")
        freq_magnitude = torch.sqrt(freq_x**2 + freq_y**2 + self.eps)
        return (1 + freq_magnitude**self.high_freq_weight).unsqueeze(0).unsqueeze(0)

    def _enhance_frequency(self, img_fft, freq_weights):
        """Enhance frequency components while preserving phase"""
        magnitude = torch.abs(img_fft)
        phase = torch.angle(img_fft)
        enhanced_magnitude = magnitude * freq_weights
        return enhanced_magnitude * torch.exp(1j * phase)

    @torch.no_grad()
    def compute_ssim(self, img1, img2):
        """
        Compute SSIM with frequency enhancement
        Args:
            img1, img2: [B, C, H, W] tensors in range [0, 1]
        """
        # Input validation
        if img1.shape != img2.shape:
            raise ValueError(f"Input shapes must match. Got {img1.shape} and {img2.shape}")
        
        # Ensure inputs are in [0, 1]
        img1 = img1.clamp(0, 1)
        img2 = img2.clamp(0, 1)
        
        # Get dynamic range for SSIM constants
        L = 1.0  # assuming normalized inputs
        C1 = (self.k1 * L) ** 2
        C2 = (self.k2 * L) ** 2

        # FFT transform
        img1_fft = torch.fft.fft2(img1)
        img2_fft = torch.fft.fft2(img2)

        # Get and apply frequency weights
        freq_weights = self._get_freq_weights(*img1.shape[-2:], img1.device).permute(0, 1, 3, 2)
        img1_enhanced = self._enhance_frequency(img1_fft, freq_weights)
        img2_enhanced = self._enhance_frequency(img2_fft, freq_weights)

        # Back to spatial domain
        img1_spatial = torch.fft.ifft2(img1_enhanced).real
        img2_spatial = torch.fft.ifft2(img2_enhanced).real

        # Compute SSIM statistics using grouped convolution
        mu1 = F.conv2d(img1_spatial, self.kernel, padding=self.kernel_size//2, groups=self.channels)
        mu2 = F.conv2d(img2_spatial, self.kernel, padding=self.kernel_size//2, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1_spatial**2, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2_spatial**2, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1_spatial*img2_spatial, self.kernel, padding=self.kernel_size//2, groups=self.channels) - mu12

        # Compute SSIM
        numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = numerator / (denominator + self.eps)

        return ssim.mean() if self.size_average else ssim.mean((1, 2, 3))

    # ... 其余方法保持不变 ...

    def forward(self, src_vec, tar_vec, mask=None):
        """
        Forward pass
        Args:
            src_vec: source vectors [N, C]
            tar_vec: target vectors [N, C]
            mask: optional mask [N]
        """
        # Input validation
        if src_vec.shape != tar_vec.shape:
            raise ValueError(f"Input shapes must match. Got {src_vec.shape} and {tar_vec.shape}")

        # Generate indices for repeated sampling
        batch_size = len(tar_vec)
        indices = torch.cat([
            torch.randperm(batch_size) if i > 0 else torch.arange(batch_size)
            for i in range(self.repeat_time)
        ]).to(src_vec.device)

        # Apply mask if provided
        if mask is not None:
            indices = indices[mask[indices]]

        # Reshape data
        tar_all = tar_vec[indices]
        src_all = src_vec[indices]

        # Validate patch height
        total_elements = tar_all.shape[0]
        if total_elements % self.patch_height != 0:
            pad_size = self.patch_height - (total_elements % self.patch_height)
            tar_all = torch.cat([tar_all, tar_all[:pad_size]], dim=0)
            src_all = torch.cat([src_all, src_all[:pad_size]], dim=0)

        # Reshape to image format
        tar_patch = tar_all.transpose(0, 1).reshape(1, 3, self.patch_height, -1)
        src_patch = src_all.transpose(0, 1).reshape(1, 3, self.patch_height, -1)

        # Compute SSIM loss
        return 1 - self.compute_ssim(src_patch, tar_patch)