Quick Start
===========

Creating a ``Channel``
----------------------

A ``Channel`` represents a single time-history with metadata and lazy
processing (drift, filter, baseline, trimming):

.. code-block:: python

    from pydysp.channel import Channel
    import numpy as np

    dt = 0.01
    t = np.arange(0, 10, dt)
    y = np.sin(2 * np.pi * 2 * t)

    ch = Channel(
        data=y,
        dt=dt,
        name_user="Acc1",
        quantity="acceleration",
        units="g",
    )

    # Apply processing (non-destructive)
    ch2 = (
        ch.drift_corrected(points=100)
          .filtered(fc=20)
          .baseline_corrected()
    )

    # Plot processed time-history
    ch2.plot()

Creating a ``Test`` (multi-channel experiment)
----------------------------------------------

A ``Test`` stores and manages multiple channels, with tools for batch
processing, plotting, spectra, transfer functions, and more:

.. code-block:: python

    from pydysp.test import Test

    test = Test.from_channels(
        name="MyTest",
        channels=[ch1, ch2],
    )

    # Plot channels in a grid
    test.plot_channels(ncols=2)

    # Compute transfer function H1
    f, H = test.transfer_function("Acc1", "Acc2", kind="H1")
