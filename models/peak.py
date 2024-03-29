import torch
import torch.nn as nn
from torch import Tensor
from typing import *
import matplotlib.pyplot as plt
from ..utils.functions import get_device
from torch.utils.data import DataLoader, TensorDataset


class PseudoVoigtPeak(nn.Module):
    """伪伏依特峰模块"""

    def __init__(self, peaks: List[Tuple[int, int]]) -> None:
        super(PseudoVoigtPeak, self).__init__()
        self.constructor_params = peaks
        self.typ = "PseudoVoigtPeak"
        self.A, self.x0, self.gamma, self.sigma, self.eta = self.init_params(peaks)

    @staticmethod
    def init_params(peaks: List[Tuple[int, int]]):
        size = len(peaks)
        x0_data = []
        A_data = []
        for x, y in peaks:
            x0_data.append(x)
            A_data.append(y)
        A = nn.Parameter(torch.tensor(A_data))
        x0 = nn.Parameter(torch.tensor(x0_data))
        gammaA = nn.Parameter(torch.randn(size))
        sigma = nn.Parameter(torch.randn(size))
        eta = nn.Parameter(torch.randn(size))
        return A, x0, gammaA, sigma, eta

    @staticmethod
    def from_peaks(X: Tensor, Y: Tensor) -> "PseudoVoigtPeak":
        """通过识别峰值点来构建组合峰模型"""
        from scipy.signal import find_peaks

        Y = Y.cpu().detach().numpy()
        peak_ids = find_peaks(Y)[0]
        peaks = []
        for i in peak_ids:
            peaks.append((X[i], Y[i]))
        return PseudoVoigtPeak(peaks=peaks)

    @staticmethod
    def analyze_peaks(peaks: List[Tuple[int, int]]):
        """分析峰的数量"""
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)  # 按照峰的高度排序
        size = len(peaks)
        final_peaks = []

        return

    def forward(self, X: Tensor) -> Tensor:
        # 计算洛伦兹成分
        X = X.to(self.A.device)
        X_expanded = X.unsqueeze(0).expand(len(self.A), -1)

        # 计算洛伦兹成分
        lorentzian = (self.A.unsqueeze(1) / torch.pi) * (
            self.gamma.unsqueeze(1)
            / ((X_expanded - self.x0.unsqueeze(1)) ** 2 + self.gamma.unsqueeze(1) ** 2)
        )

        # 计算高斯成分
        gaussian = (
            self.A.unsqueeze(1)
            / (self.sigma.unsqueeze(1) * torch.sqrt(torch.tensor(2 * torch.pi)))
        ) * torch.exp(
            -0.5 * ((X_expanded - self.x0.unsqueeze(1)) / self.sigma.unsqueeze(1)) ** 2
        )

        # 结合洛伦兹成分和高斯成分
        result = (
            self.eta.unsqueeze(1) * lorentzian + (1 - self.eta.unsqueeze(1)) * gaussian
        )
        # 结合洛伦兹成分和高斯成分，生成伪伏依特峰
        return result

    def export(self, path: str):
        torch.save(
            {
                "weights": self.state_dict(),
                "constructor_params": self.constructor_params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        checkpoint = torch.load(path, map_location=get_device(device))
        model = cls(checkpoint["constructor_params"]).to(get_device(device))
        model.load_state_dict(checkpoint["weights"])
        return model.eval()

    def cost(self) -> Tensor:
        """TODO: 计算代价
        需要考虑到峰的重叠、峰值是否为负数等因素
        """
        X = self.x0.detach()
        peak_size = self.A.size(0)  # 峰的个数
        peak_matrix = self(X)  # 峰值矩阵
        torch.max(peak_matrix, dim=0)
        peak_matrix.max(dim=1)
        return

    def get_state(self, i: int):
        A = self.A[i].item()
        x0 = self.x0[i].item()
        gamma = self.gamma[i].item()
        sigma = self.sigma[i].item()
        eta = self.eta[i].item()
        return {"A": A, "x0": x0, "gamma": gamma, "sigma": sigma, "eta": eta}

    def label(self, i: int):
        A = self.A[i].item()
        x0 = self.x0[i].item()
        gamma = self.gamma[i].item()
        sigma = self.sigma[i].item()
        eta = self.eta[i].item()
        return f"Peak{i}[A={A:.2f}, x0={x0:.2f}, γ={gamma:.2f}, σ={sigma:.2f}, η={eta:.2f}]"

    def figure(self, X: Tensor, ax: Optional[plt.Axes] = None) -> plt.Figure:
        was_training = self.training
        self.eval()
        # 检查是否提供了轴对象，如果没有，则创建新图形和轴
        if ax is None:
            fig, ax = plt.subplots()
        Y: Tensor = self(X).detach()
        for i, y in enumerate(Y):
            ax.plot(X.cpu().numpy(), y.cpu(), label=self.label(i))
        if was_training:
            self.train()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper left", bbox_to_anchor=(0.8, 1), fontsize=7)

        return ax  # 返回轴对象

    def show(self):
        plt.show()

    def status(self):
        return [self.get_state(i) for i in range(len(self.A))]

    def train_model(
        self,
        X: Tensor,
        Y: Tensor,
        epochs=100,
        batch_size=100,
        lr: float = 0.01,
    ) -> None:
        """训练组合峰模型"""
        # 生成优化器
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 计算对数损失
        loss_fn = nn.MSELoss()
        # 训练
        # X = X.unsqueeze(0).expand(Y.size(0), -1)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            s = 0
            for x_batch, y_batch in dataloader:
                # 前向传播
                y_pred = self(x_batch).sum(dim=0)
                # 计算损失
                loss = loss_fn(y_pred, y_batch)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                s += loss.item()
            print(f"Epoch {epoch}: {s / batch_size}")
