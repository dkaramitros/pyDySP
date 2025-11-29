import scipy as sp


def downsample(file_in: str, file_out: str, factor: int = 20):
    """Downsample array-like variables stored in a MATLAB .mat file.
    Parameters
    ----------
    file_in : str
        Path to the input .mat file.
    file_out : str
        Path to the output .mat file.
    factor : int, optional
        Downsampling factor (default is 20).
    Returns
    -------
    None
    """
    try:
        imported_data = sp.io.loadmat(file_in)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_in}' not found.")
    for key, val in imported_data.items():
        if isinstance(val, (list, tuple)):
            continue
        if hasattr(val, "shape") and val.ndim >= 1:
            imported_data[key] = val[::factor]
    sp.io.savemat(file_out, imported_data)
    return
