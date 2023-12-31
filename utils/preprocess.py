import torch
import numpy as np
from torch import Tensor
from typing import *
import matplotlib.pyplot as plt

from .functions import get_device

T = TypeVar("T", Tensor, np.ndarray, List[Union[int, float]])


class DataPreprocessor:
    def __init__(self, X: T, Y: T, ax: Optional[plt.Axes] = None) -> None:
        self.X = DataPreprocessor.wapper(X)
        self.Y = DataPreprocessor.wapper(Y)
        self.ax = ax or plt.subplots(figsize=(15, 8))[1]
        self.__step("Original")

    @staticmethod
    def data_type():
        return torch.float64

    @staticmethod
    def wapper(ele: T) -> Tensor:
        """转换为 Tensor"""
        typ = DataPreprocessor.data_type()
        if isinstance(ele, Tensor):
            return ele.to(typ)
        if isinstance(ele, np.ndarray):
            return torch.from_numpy(ele).to(typ)
        if isinstance(ele, list):
            return torch.tensor(ele, dtype=typ)
        raise TypeError(f"Unsupported type: {type(ele)}")

    def __step(self, name: str, show=True):
        """绘制当前数据"""
        if show is False:
            return
        self.ax.plot(self.X.detach().numpy(),
                     self.Y.detach().numpy(), label=name)
    
    def rectify(self, show=True) -> "DataPreprocessor":
        """将 Y 中小于 0 的部分设置为 0"""
        self.Y = torch.max(self.Y, torch.zeros_like(self.Y))
        self.__step("rectified", show=show)
        return self

    def smooth(self, sigma: float = 5, show=True) -> "DataPreprocessor":
        """平滑处理"""
        from scipy.ndimage import gaussian_filter
        self.Y = torch.from_numpy(
            gaussian_filter(self.Y.detach().numpy(), sigma))
        self.__step(f"smoothed (σ={sigma})", show=show)
        return self

    def show(self):
        ax = self.ax
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=7)
        plt.show()
        return self

    def export(self, device: Optional[str] = None):
        """导出数据"""
        device = get_device(device)
        return self.X.to(device), self.Y.to(device)

    @staticmethod
    def from_text(path: str):
        """从文本文件中读取数据"""
        X, Y = [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                x, y = line.split()
                X.append(float(x))
                Y.append(float(y))
        return DataPreprocessor(X, Y)

    @staticmethod
    def from_csv(path: str):
        """从 CSV 文件中读取数据"""
        import pandas as pd
        df = pd.read_csv(path)
        return DataPreprocessor(df["x"].to_numpy(), df["y"].to_numpy())


def test():
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2) + np.random.normal(0, 0.1, 100)  # 添加一些噪声
    d = DataPreprocessor(x, y)
    d.smooth(1).smooth(2).show()


def test1():
    DataPreprocessor.from_text("datas/UV_Vis_TwoPeak.txt").smooth(1).show()
