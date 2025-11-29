pyDySP: Dynamic Signal Processing in Python
===========================================

pyDySP is a lightweight Python package for **dynamic signal processing of 
experimental data**, designed around two core classes:

- ``Channel`` – a single time-history with metadata and lazy processing
- ``Test`` – a collection of Channels with batch tools, spectra, TFs, plots

It is built for laboratory and field measurements (shaking-table tests, 
soil–structure interaction experiments, structural dynamics, etc.) but is 
general enough for any time-series workflows.

Features
--------

- Drift / filtering / baseline / trimming (non-destructive, lazy)
- Fourier / Welch spectra with peak detection
- Arias intensity & significant duration windows
- Elastic response spectra (Newmark beta method)
- Cross-spectra, coherence, transfer functions
- Batch processing on multi-channel tests
- Publication-ready plots

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   quickstart
   pydysp