import torch
import torch.nn as nn
from wavelets.wavelet import create_wavelet_filter, wavelet_transform, inverse_wavelet_transform
import cv2
import numpy as np
import torch.nn.functional as F


def remove_low_frequency(image_tensor, wavelet_type='haar', device='cpu', layer=1):
    """
    小波变换后将低频分量 (LL) 置为 0，并逆变换回去。

    参数:
        image_tensor: torch.Tensor, 输入图像，形状为 (C, H, W)，单通道或多通道张量。
        wavelet_type: str, 小波类型 (如 'haar', 'db4')。
        device: str, 运行设备 ('cpu', 'cuda', 'mps')。

    返回:
        reconstructed: torch.Tensor, 去除低频分量后的图像，形状与输入相同。
    """
    # 确保输入在正确的设备上
    image_tensor = image_tensor.to(device)

    # 获取输入的通道数
    channels = image_tensor.shape[0]

    # 创建小波滤波器
    wt_filter, iwt_filter = create_wavelet_filter(wavelet_type, channels, channels, torch.float)
    wavelet_filters = nn.Parameter(iwt_filter, requires_grad=False)
    wavelet_filters.to(device)

    # 小波分解
    coeffs = wavelet_transform(image_tensor.unsqueeze(0), wavelet_filters)

    if layer == 1:
        # 将低频分量 LL 置为 0
        coeffs[:, :, 0, :, :] = 0  # LL 分量对应的索引为 0
#        coeffs[:, :, 3, :, :] = 0  # HH 分量对应的索引为 0
    elif layer == 2:
        # 对第一层 LL 再次小波变换
        LL1 = coeffs[:, :, 0, :, :]  # shape: (1, C, H//2, W//2)
        coeffs_2 = wavelet_transform(LL1, wavelet_filters)  # shape: (1, C, 4, H//4, W//4)

        # 将第二层 LL 分量置 0
        coeffs_2[:, :, 0, :, :] = 0

        # 第二层逆变换：还原第一层 LL
        LL1_reconstructed = inverse_wavelet_transform(coeffs_2, wavelet_filters)  # shape: (1, C, H//2, W//2)

        # 替换第一层中的 LL 为重建后的 LL
        coeffs[:, :, 0, :, :] = LL1_reconstructed
    else:
        raise ValueError("Only layer=1 or layer=2 is supported.")
        
    # 逆小波变换重建图像   
    reconstructed = inverse_wavelet_transform(coeffs, wavelet_filters)
    return reconstructed.squeeze(0)  # 移除批次维度


def wave_transform(img_tensor, wt_type='haar', in_channels=3, out_channels=3):
    wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
    wt_filter = nn.Parameter(wt_filter, requires_grad=False)
    device = img_tensor.device
    wt_filter = wt_filter.to(device)
    rst = wavelet_transform(img_tensor, wt_filter)

    return rst


def compute_frequency_intensity_map(image_tensor, wave_type='haar'):
    # 确定输入图像的通道数和设备
    in_channels = image_tensor.shape[0]
    device = image_tensor.device

    # 创建小波滤波器
    filters = create_wavelet_filter(wave_type, in_channels, in_channels)

    # 执行小波分解
    coeffs = wavelet_transform(image_tensor.unsqueeze(0), filters)

    # 计算高频部分的能量（LH, HL, HH）
    high_freq_energy = torch.sum(torch.abs(coeffs[:, :, 1:, :, :]) ** 2, dim=2)  # 聚合高频分量

    # 将高频能量图上采样回原始分辨率
    high_freq_energy_upsampled = torch.nn.functional.interpolate(
        high_freq_energy, scale_factor=2, mode='bilinear', align_corners=False
    )

    # 归一化处理
    freq_map = high_freq_energy_upsampled.squeeze(0).mean(dim=0)  # 平均多个通道，返回单通道
    freq_map = freq_map / (freq_map.max() + 1e-6)  # 归一化到 [0, 1]

    return freq_map


def get_mask(ori_image, mode='dwt', layer=1):
    image_tensor = ori_image.unsqueeze(0)
    if mode == 'dwt':
        wo_LL = remove_low_frequency(image_tensor, layer=layer)
        wo_LL = torch.abs(wo_LL)
    
    elif mode == 'sobel':
        kernel_x = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
        gx = F.conv2d(image_tensor.unsqueeze(0), kernel_x, padding=1)
        gy = F.conv2d(image_tensor.unsqueeze(0), kernel_y, padding=1)
        wo_LL = torch.sqrt(gx ** 2 + gy ** 2).squeeze()

    elif mode == 'scharr':
        kernel_x = torch.tensor([[3, 0, -3],
                                 [10, 0, -10],
                                 [3, 0, -3]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[3, 10, 3],
                                 [0, 0, 0],
                                 [-3, -10, -3]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
        gx = F.conv2d(image_tensor.unsqueeze(0), kernel_x, padding=1)
        gy = F.conv2d(image_tensor.unsqueeze(0), kernel_y, padding=1)
        wo_LL = torch.sqrt(gx ** 2 + gy ** 2).squeeze()

    elif mode == 'laplacian':
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32, device=image_tensor.device).view(1, 1, 3, 3)
        wo_LL = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=1).abs().squeeze()
    
    normalized = ((wo_LL - wo_LL.min()) / (wo_LL.max() - wo_LL.min())).squeeze(0)
    # for i in range(10):
    #     normalized = torch.log(normalized + 1)
    #     normalized = normalized / (normalized.max() + 1e-10)  # 归一化到 [0, 1]
    # while normalized.mean() < 0.6:
    #     normalized = torch.log(normalized + 1)
    #     normalized = normalized / (normalized.max() + 1e-10)  # 归一化到 [0, 1]
    return normalized


if __name__ == '__main__':
    # 检查设备（MPS 或 CPU）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 读取图像（灰度图）
    img = cv2.imread('bicycle.png', cv2.IMREAD_GRAYSCALE)
    img_tensor = torch.Tensor(img).unsqueeze(0).to(device)
    normalized = get_mask(img_tensor, img_tensor)
    print(np.shape(normalized), normalized, normalized.max(), normalized.min())
    
    import matplotlib.pyplot as plt
    plt.imshow(normalized.cpu().numpy(), cmap='jet')
    plt.title('Frequency Intensity Map')
    plt.colorbar(label='Frequency Intensity')
    plt.axis('off')
    plt.show()
