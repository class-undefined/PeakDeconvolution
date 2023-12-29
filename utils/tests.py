import numpy as np
import torch
from torch import tensor
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from ..models.peak import PseudoVoigtPeak, CombinedPeaks  # 替换为包含 PseudoVoigtPeak 类的模块


def test():
    # 示例数据
    x = np.array([0, 1, 2, 1, 0, 1, 2, 3, 1, 0])

    # 寻找峰值
    peaks = find_peaks(x, prominence=3)
    print(peaks)


def test_tensor():
    x = tensor([0, 1, 2, 1, 0, 1, 2, 3, 1, 0], dtype=torch.int32)
    for i in x:
        print(i)


def test_figure_method1():
    # 创建 PseudoVoigtPeak 实例
    peak1 = PseudoVoigtPeak.gen(1.0, 0.0, 1.0, 1.0, 0.5)
    peak2 = PseudoVoigtPeak.gen(1.0, 1.0, 1.0, 1.0, 0.5)

    # 生成输入数据 X
    X = torch.linspace(-5, 5, 100)
    fig, ax = plt.subplots()
    # 调用 figure 方法
    peak1.figure(X, ax=ax)
    peak2.figure(X, ax=ax)
    plt.show()


def test_gaussian():
    # 生成示例数据
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2) + np.random.normal(0, 0.1, 100)  # 添加一些噪声

    # 应用高斯滤波
    sigma = 4
    y_smoothed = gaussian_filter(y, sigma=sigma)

    # 绘制原始数据和平滑后的数据
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original Data', alpha=0.5)
    plt.plot(x, y_smoothed, label='Smoothed Data', color='red')
    plt.title('Gaussian Smoothing Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()
