# pyDySP — Dynamic Signal Processing for Experimental Data

![License](https://img.shields.io/github/license/dkaramitros/pyDySP)
[![Read the Docs](https://img.shields.io/readthedocs/pydysp)](https://pydysp.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/pydysp)](https://pypi.org/project/pyDySP/)
![PyPI - Status](https://img.shields.io/pypi/status/pydysp)
![Maintenance](https://img.shields.io/maintenance/active/2025)

**pyDySP** is a lightweight Python library for **dynamic signal processing** of experimental data, developed at  
*the Shaking Table and SoFSI Laboratories — University of Bristol*.

It provides:

- A **Channel** class for time-history signals  
  (drift correction, filtering, baseline, trimming, spectra, Arias intensity, response spectra, plots, metadata).
- A **Test** container class for multi-channel experiments  
  (selection by index/name/tags, batch processing, cross-spectra, transfer functions, modal analysis via sdypy, plotting utilities, csv/mat I/O).
- Clean, readable code designed for notebooks, research reports, and reproducible workflows.
- Full internal documentation (docstrings) and a growing library of example notebooks.

---

## Table of Contents

- [pyDySP — Dynamic Signal Processing for Experimental Data](#pydysp--dynamic-signal-processing-for-experimental-data)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Option A — Install from PyPI (recommended)](#option-a--install-from-pypi-recommended)
    - [Option B — Developer installation (editable)](#option-b--developer-installation-editable)
    - [Option C — Use directly from a folder](#option-c--use-directly-from-a-folder)
  - [Environment Setup](#environment-setup)
    - [Virtual environment (`venv`)](#virtual-environment-venv)
    - [Conda / Mamba environment](#conda--mamba-environment)
  - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
    - [Single channel](#single-channel)
    - [Multi‑channel test](#multichannel-test)
    - [Spectra \& trimming](#spectra--trimming)
    - [Response spectrum](#response-spectrum)
    - [Transfer functions](#transfer-functions)
    - [Channel health](#channel-health)
    - [Modal identification (optional)](#modal-identification-optional)
  - [Features](#features)
    - [Channel](#channel)
    - [Test](#test)
  - [Example Notebooks](#example-notebooks)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

---

## Installation

### Option A — Install from PyPI (recommended)

```bash
pip install pydysp
```

If using Jupyter notebooks:

```bash
pip install ipykernel
python -m ipykernel install --user --name pydysp-env
```

### Option B — Developer installation (editable)

```bash
git clone https://github.com/dkaramitros/pyDySP.git
cd pyDySP
pip install -e .
```

### Option C — Use directly from a folder

```python
import sys
sys.path.append("/path/to/parent/of/pydysp")
from pydysp import Channel, Test
```

---

## Environment Setup

### Virtual environment (`venv`)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate       # Windows
pip install pydysp
```

### Conda / Mamba environment

```bash
mamba create -n pydysp python=3.11 pip
mamba activate pydysp
pip install pydysp
```

---

## Dependencies

- numpy  
- scipy  
- matplotlib  
- tabulate  

Optional (for modal analysis):

```bash
pip install sdypy-EMA
```

---

## Quick Start

### Single channel

```python
from pydysp import Channel
import numpy as np

dt = 0.01
t = np.arange(0, 10, dt)
y = np.sin(2*np.pi*2*t)

ch = Channel(data=y, dt=dt, name_user="Acc1", quantity="acceleration", units="g")
t_proc, y_proc = ch.processed()
ch.plot()
```

### Multi‑channel test

```python
from pydysp import Test
test = Test.from_channels(name="MyTest", channels=[ch1, ch2])
print(test.info())
test.plot_channels(ncols=2)
```

### Spectra & trimming

```python
spec = ch.fourier()
f_peak, s_peak = spec.peak()

ch_trim = ch.trim_by_arias()
```

### Response spectrum

```python
rs = ch.response_spectrum()
rs.plot(y="Sa", logx=True)
```

### Transfer functions

```python
f, H = test.transfer_function("Shaker", "Acc1", kind="H1")
```

### Channel health

```python
print(test.channel_health())
```

### Modal identification (optional)

```python
model = test.ema_model(input="Shaker", outputs=["Acc1","Acc2"], lower=2, upper=40)
model.get_poles()
model.select_poles()
model.print_modal_data()
```

---

## Features

### Channel
- Drift, filter, baseline  
- Trim (threshold, fraction, Arias)  
- Fourier & Welch spectra  
- Arias intensity  
- Response spectrum  
- Time‑domain metrics  
- Rich metadata  
- Plot utilities  

### Test
- Channel selection (index, name, tags, slices)  
- Batch processing  
- Cross‑spectra, TFs, delays  
- Modal analysis (sdypy)  
- `.mat` (SoFSI/EQUALS) and `.csv` I/O  
- Grid plots & channel lists  
- Health diagnostics  

---

## Example Notebooks

Available in:

```
examples/
```

---

## Documentation

https://pydysp.readthedocs.io/en/latest/

---

## Contributing

```bash
pip install -e .
pip install pytest
pytest -q
```

---

## License

MIT License
