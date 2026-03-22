import pywt
import torch
import torch.nn.functional as F  # 用于执行卷积和转置卷积操作


# 创建小波滤波器的函数
# wave: 小波类型
# in_size: 输入的通道数
# out_size: 输出的通道数
# type: 数据类型
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # 获取指定小波类型的滤波器系数
    w = pywt.Wavelet(wave)

    # 获取小波的高通和低通滤波器的分解系数（dec_hi和dec_lo），并进行反转
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)

    # 构造二维小波分解滤波器（4个方向：低-低，低-高，高-低，高-高）
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # 低频-低频
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # 低频-高频
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # 高频-低频
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    # 将滤波器扩展到所有输入通道（每个通道一个滤波器）
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 获取小波的高通和低通滤波器的重构系数，并进行翻转
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

    # 构造二维小波重构滤波器（对应的四个方向：低-低，低-高，高-低，高-高）
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  # 低频-低频
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  # 低频-高频
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  # 高频-低频
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    # 将滤波器扩展到所有输出通道（每个通道一个滤波器）
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    # 返回分解滤波器和重构滤波器
    return dec_filters, rec_filters


# 小波分解的实现
# x: 输入张量
# filters: 分解滤波器
def wavelet_transform(x, filters):
    b, c, h, w = x.shape  # 获取输入的形状（批次、通道、高度、宽度）

    # 计算卷积的填充大小（使输出大小匹配）
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    # 使用二维卷积进行小波变换，步长为2进行下采样，分组数等于通道数
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)

    # 将输出调整为4个子带（低频-低频、低频-高频、高频-低频、高频-高频）
    x = x.reshape(b, c, 4, h // 2, w // 2)

    # 返回小波分解后的输出
    return x


# 逆小波变换的实现
# x: 小波分解后的张量
# filters: 重构滤波器
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape  # 获取输入形状

    # 计算填充大小，使输出与原始大小匹配
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    # 将4个子带重新合并回单个通道
    x = x.reshape(b, c * 4, h_half, w_half)

    # 使用二维转置卷积（反卷积）进行逆小波变换，步长为2进行上采样
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)

    # 返回重构后的输出
    return x