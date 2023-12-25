from .models.peak import PseudoVoigtPeak, CombinedPeaks
from .utils.preprocess import DataPreprocessor
from .utils.functions import get_device
import torch
from torch import Tensor
from typing import Optional
import numpy as np
import random


def set_seed(seed):
    """固定随机数种子以确保结果可重复."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(X: Optional[Tensor] = None,
          Y: Optional[Tensor] = None,
          preprocessor: Optional[DataPreprocessor] = None,
          num_peaks: Optional[int] = None,
          epochs=500,
          batch_size=100,
          lr=0.01,
          seed: Optional[int] = None,
          device: Optional[str] = None
          ):
    """训练组合峰模型
    @param `X`: 输入数据
    @param `Y`: 输出数据
    @param `preprocessor`: 数据预处理器
    @param `num_peaks`: 峰的数量
    @param `epochs`: 训练轮数
    @param `batch_size`: 批大小
    @param `lr`: 学习率
    @param `seed`: 随机数种子
    """
    if seed is not None:
        set_seed(seed)
    if preprocessor is not None:
        X, Y = preprocessor.export(device=device)
    else:
        preprocessor = DataPreprocessor(X, Y)
        X, Y = preprocessor.export(device=device)
    model = CombinedPeaks(
        num_peaks) if num_peaks is not None else CombinedPeaks.from_peaks(X, Y)
    model = model.to(get_device(device))
    model.train_model(X, Y, epochs=epochs,
                      batch_size=batch_size, lr=lr, device=device)
    model.figure(X, ax=preprocessor.ax)
    print(model.status())
    preprocessor.show()


def test():
    x = torch.linspace(-5, 5, 100)
    y = torch.exp(-x**2) + torch.randn(100) * 0.1
    train(x, y)


def test1():
    p = DataPreprocessor.from_text(
        "datas/UV_Vis_TwoPeak.txt")
    train(preprocessor=p, epochs=200, lr=50)


def test2():
    p = DataPreprocessor.from_csv(
        "datas/test-deconvolve.csv")
    train(preprocessor=p,
          epochs=200,
          batch_size=100,
          lr=0.001,
          device="cpu",
          seed=79)


def test3():
    p = DataPreprocessor.from_csv(
        "datas/test-deconvolve.csv")
    train(preprocessor=p, num_peaks=2, epochs=500,
          batch_size=100, lr=0.002, seed=5)
