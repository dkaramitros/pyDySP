import scipy.io as sio


def downsample(file_in: str, file_out: str, factor: int = 20) -> None:
    """Downsample array-like variables in a MATLAB ``.mat`` file.

    Parameters
    ----------
    file_in : str
        Path to the input MATLAB ``.mat`` file to read.
    file_out : str
        Path to write the downsampled MATLAB ``.mat`` file.
    factor : int, optional
        Downsampling factor. Must be a positive integer. Default is 20.

    Raises
    ------
    FileNotFoundError
        If the input file cannot be found or opened.
    ValueError
        If ``factor`` is not a positive integer.

    Notes
    -----
    Only array-like values that expose a ``shape`` attribute and have
    at least one dimension (``ndim >= 1``) are downsampled. Python lists
    and tuples are left unchanged so non-numeric containers are not
    inadvertently modified.
    """
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")

    try:
        imported_data = sio.loadmat(file_in)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(
            f"Input file not found or unreadable: {file_in}"
        ) from exc

    # Downsample only numeric/array-like objects (ndim >= 1). Skip lists/tuples.
    for key, val in imported_data.items():
        if isinstance(val, (list, tuple)):
            continue
        if hasattr(val, "shape") and getattr(val, "ndim", 0) >= 1:
            imported_data[key] = val[::factor]

    sio.savemat(file_out, imported_data)
