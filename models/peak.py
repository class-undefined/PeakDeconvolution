import torch
import torch.nn as nn
from torch import Tensor
from typing import *
import matplotlib.pyplot as plt


class PseudoVoigtPeak(nn.Module):
    """伪伏依特峰模块"""
    count = 0

    def __init__(self, id: int = None) -> None:
        super(PseudoVoigtPeak, self).__init__()
        self.typ = "PseudoVoigtPeak"
        self.id = id or PseudoVoigtPeak.count
        PseudoVoigtPeak.count += 1
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
        gaussian = (self.A / (self.sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * \
            torch.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)
        # 结合洛伦兹成分和高斯成分，生成伪伏依特峰
        return self.eta * lorentzian + (1 - self.eta) * gaussian

    @classmethod
    def gen(cls, A: float, x0: float, gamma: float, sigma: float, eta: float) -> "PseudoVoigtPeak":
        """生成伪伏依特峰"""
        peak = cls()
        peak.A = nn.Parameter(torch.tensor(A))
        peak.x0 = nn.Parameter(torch.tensor(x0))
        peak.gamma = nn.Parameter(torch.tensor(gamma))
        peak.sigma = nn.Parameter(torch.tensor(sigma))
        peak.eta = nn.Parameter(torch.tensor(eta))
        return peak

    def name(self):
        return f"Peak{self.id}[A={self.A.item()}, x0={self.x0.item()}, γ={self.gamma.item()}, σ={self.sigma.item()}, η={self.eta.item()}]"

    def figure(self, X: Tensor, ax: Optional[plt.Axes] = None) -> plt.Figure:
        was_training = self.training
        self.eval()
        # 检查是否提供了轴对象，如果没有，则创建新图形和轴
        if ax is None:
            fig, ax = plt.subplots()

        # 生成伪伏依特峰的可视化图像
        Y = torch.stack([self.forward(x) for x in X]).detach()

        # 绘图
        ax.plot(X.numpy(), Y.numpy(), label=self.name())
        if was_training:
            self.train()
        return ax  # 返回轴对象


class CombinedPeaks(nn.Module):
    """组合峰模块"""

    def __init__(self, num_peaks: int) -> None:
        super(CombinedPeaks, self).__init__()
        # peaks: 存储多个伪伏依特峰的模块列表
        self.peaks = nn.ModuleList([PseudoVoigtPeak(id=i)
                                   for i in range(num_peaks)])

    def forward(self, x: Tensor) -> Tensor:
        # 将所有峰的贡献加总，以生成组合的光谱
        return sum(peak(x) for peak in self.peaks)

    def name(self):
        return f"CombinedPeaks [{len(self.peaks)}]"

    def figure(self, X: Tensor) -> plt.Figure:
        # 检查是否提供了轴对象，如果没有，则创建新图形和轴
        # 生成组合峰的可视化图像
        was_training = self.training
        self.eval()
        Y = torch.stack([self.forward(x) for x in X]).detach()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(X.numpy(), Y.numpy(), label=self.name())
        for peak in self.peaks:
            peak.figure(X, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1))
        if was_training:
            self.train()
        return fig, ax

    @classmethod
    def gen(cls, peaks: List["PseudoVoigtPeak"]) -> "CombinedPeaks":
        """生成组合峰"""
        this = cls(0)
        this.peaks = nn.ModuleList(peaks)
        return this

    @classmethod
    def from_peaks(cls, Y: Tensor) -> "CombinedPeaks":
        """通过识别峰值点来构建组合峰模型"""
        from scipy.signal import find_peaks
        peaks = find_peaks(Y.detach().numpy(), prominence=0.5)[0]
        return cls(num_peaks=len(peaks))

    def train(self, X: Tensor, Y: Tensor, epochs: int = 1000, lr: float = 0.01) -> None:
        """训练组合峰模型"""
        # 生成优化器
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 生成损失函数
        loss_fn = nn.MSELoss()
        # 训练
        for epoch in range(epochs):
            # 前向传播
            Y_pred = self(X)
            # 计算损失
            loss = loss_fn(Y_pred, Y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印训练信息
            if epoch % 100 == 0:
                print(f"Epoch {epoch} loss: {loss.item()}")
