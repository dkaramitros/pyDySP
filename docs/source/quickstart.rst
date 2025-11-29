Quick Start
===========

Creating a ``Channel``

.. code-block:: python

    from pydysp.channel import Channel
    import numpy as np

    dt = 0.01
    t = np.arange(0, 10, dt)
    y = np.sin(2*np.pi*2*t)

    ch = Channel(data=y, dt=dt, name_user="Acc1", quantity="acceleration", units="g")

    # Apply processing
    ch2 = ch.drift_corrected(points=100).filtered(fc=20).baseline_corrected()

    # Plot
    ch2.plot()

Creating a ``Test`` (multiple channels)

.. code-block:: python

    from pydysp.test import Test

    test = Test.from_channels(name="MyTest", channels=[ch1, ch2])

    # Batch plot
    test.plot_channels()

    # Compute TF
    f, H = test.transfer_function("Acc1", "Acc2", kind="H1")
