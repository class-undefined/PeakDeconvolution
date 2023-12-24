from .models.peak import PseudoVoigtPeak, CombinedPeaks
from .utils.preprocess import DataPreprocessor
import torch
from torch import Tensor
from typing import Optional


def train(X: Optional[Tensor] = None,
          Y: Optional[Tensor] = None,
          preprocessor: Optional[DataPreprocessor] = None,
          num_peaks: Optional[int] = 1,
          epochs=1000,
          lr=0.01,
          ):
    if preprocessor is not None:
        X, Y = preprocessor.export()
    else:
        preprocessor = DataPreprocessor(X, Y)
        X, Y = preprocessor.export()
    model = CombinedPeaks(
        num_peaks) if num_peaks is not None else CombinedPeaks.from_peaks(Y)
    model.train_model(X, Y, epochs=epochs, lr=lr)
    model.figure(X, ax=preprocessor.ax)
    preprocessor.show()


def test():
    x = torch.linspace(-5, 5, 100)
    y = torch.exp(-x**2) + torch.randn(100) * 0.1
    train(x, y)


def test1():
    p = DataPreprocessor.from_text(
        "datas/UV_Vis_TwoPeak.txt")
    train(preprocessor=p, num_peaks=1, epochs=2000, lr=20)
