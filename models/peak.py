import torch
import torch.nn as nn
from torch import Tensor
from typing import *


class PseudoVoigtPeak(nn.Module):
    """伪伏依特峰模块"""

    def __init__(self) -> None:
        super(PseudoVoigtPeak, self).__init__()
        # A: 峰的最大强度（高度）参数
        self.A = nn.Parameter(torch.randn(1))
        # x0: 峰的中心位置参数
        self.x0 = nn.Parameter(torch.randn(1))
        # gamma: 洛伦兹成分的半高宽参数
        self.gamma = nn.Parameter(torch.randn(1))
        # sigma: 高斯成分的标准差参数
        self.sigma = nn.Parameter(torch.randn(1))
        # eta: 高斯成分和洛伦兹成分的混合比例参数
        self.eta = nn.Parameter(torch.randn(1))

    def forward(self, x: Tensor) -> Tensor:
        # 计算洛伦兹成分
        lorentzian = (self.A / torch.pi) * (self.gamma /
                                            ((x - self.x0) ** 2 + self.gamma ** 2))
        # 计算高斯成分
        gaussian = (self.A / (self.sigma * torch.sqrt(2 * torch.pi))) * \
            torch.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)
        # 结合洛伦兹成分和高斯成分，生成伪伏依特峰
        return self.eta * lorentzian + (1 - self.eta) * gaussian


class CombinedPeaks(nn.Module):
    """组合峰模块"""

    def __init__(self, num_peaks: int) -> None:
        super(CombinedPeaks, self).__init__()
        # peaks: 存储多个伪伏依特峰的模块列表
        self.peaks = nn.ModuleList([PseudoVoigtPeak()
                                   for _ in range(num_peaks)])

    def forward(self, x: Tensor) -> Tensor:
        # 将所有峰的贡献加总，以生成组合的光谱
        return sum(peak(x) for peak in self.peaks)
