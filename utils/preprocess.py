import torch
import numpy as np
from torch import Tensor
from typing import *
import matplotlib.pyplot as plt

T = TypeVar("T", Tensor, np.ndarray, List[Union[int, float]])


class DataPreprocessor:
    def __init__(self, X: T, Y: T, ax: Optional[plt.Axes] = None) -> None:
        self.X = DataPreprocessor.wapper(X)
        self.Y = DataPreprocessor.wapper(Y)
        self.ax = ax or plt.subplots(figsize=(12, 6))[1]
        self.__step("original")

    @staticmethod
    def wapper(ele: T) -> Tensor:
        """转换为 Tensor"""
        if isinstance(ele, Tensor):
            return ele
        if isinstance(ele, np.ndarray):
            return torch.from_numpy(ele)
        if isinstance(ele, list):
            return torch.tensor(ele)
        raise TypeError(f"Unsupported type: {type(ele)}")

    def __step(self, name: str):
        """绘制当前数据"""
        self.ax.plot(self.X.detach().numpy(),
                     self.Y.detach().numpy(), label=name)

    def smooth(self, sigma: float = 5) -> "DataPreprocessor":
        """平滑处理"""
        from scipy.ndimage import gaussian_filter
        self.Y = torch.from_numpy(
            gaussian_filter(self.Y.detach().numpy(), sigma))
        self.__step(f"smoothed (σ={sigma})")
        return self

    def show(self):
        ax = self.ax
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1))
        plt.show()
        return self

    def export(self):
        """导出数据"""
        return self.X, self.Y

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


def test():
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2) + np.random.normal(0, 0.1, 100)  # 添加一些噪声
    d = DataPreprocessor(x, y)
    d.smooth(1).smooth(2).show()


def test1():
    DataPreprocessor.from_text("datas/UV_Vis_TwoPeak.txt").smooth(1).show()
