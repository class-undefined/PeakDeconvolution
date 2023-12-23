import numpy as np
from scipy.signal import find_peaks


def test():
    # 示例数据
    x = np.array([0, 1, 2, 1, 0, 1, 2, 3, 1, 0])

    # 寻找峰值
    peaks = find_peaks(x, prominence=3)
    print(peaks)
