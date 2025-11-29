import numpy as np
import scipy.io as sio
from pydysp.downsample import downsample


def test_downsample_basic(tmp_path):
    file_in = tmp_path / "in.mat"
    file_out = tmp_path / "out.mat"

    data = {
        # Column vector: axis 0 has length 10 → will be reduced to 5
        "vec": np.arange(10.0).reshape(-1, 1),
        # 2D matrix: axis 0 length 6 → reduced to 3
        "mat": np.arange(24.0).reshape(6, 4),
        # Non-array: should be preserved
        "skip": [1, 2, 3],
    }

    sio.savemat(file_in, data)

    downsample(str(file_in), str(file_out), factor=2)
    out = sio.loadmat(file_out)

    # 1D-like vector stored as (10,1) → downsampled to (5,1)
    assert out["vec"].shape == (5, 1)

    # 2D matrix: first dimension halved
    assert out["mat"].shape == (3, 4)

    # Non-array structure still present
    assert "skip" in out


def test_downsample_bad_factor(tmp_path):
    file_in = tmp_path / "in.mat"
    sio.savemat(file_in, {"x": np.arange(10.0)})

    file_out = tmp_path / "out.mat"

    # factor=0 should raise
    import pytest

    with pytest.raises(ValueError):
        downsample(str(file_in), str(file_out), factor=0)
