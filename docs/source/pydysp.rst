API Reference
=============

This section documents the public API of **pyDySP**.

The core workflow is built around two main classes:

* :class:`pydysp.channel.Channel` – single time-history with metadata and lazy processing.
* :class:`pydysp.test.Test` – multi-channel experiment with batch tools, spectra,
  transfer functions, I/O and diagnostics.

Additional modules provide spectral representations, Arias intensity, response spectra
and utilities such as downsampling.

.. toctree::
   :maxdepth: 2
   :caption: API:

   channel
   test
   api_other
