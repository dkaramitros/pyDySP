import numpy as np
import matplotlib.pyplot as plt

from pydysp.arias import AriasResult


def test_ariasresult_plot_smoke():
    t = np.linspace(0.0, 10.0, 100)
    Ia = np.linspace(0.0, 1.0, 100)
    res = AriasResult(t=t, Ia=Ia, t_start=2.0, t_end=8.0)

    fig, ax = plt.subplots()
    res.plot(ax=ax, show_window=True)
    plt.close(fig)
