import warnings
import operator
import sys
import functools as ft
from functools import reduce

import numpy as np
import xarray as xr
import pandas as pd
import xrft


__all__ = [
    "dwvlt",
    "cwvlt",
    "cwvlt2",
    "wvlt_power_spectrum",
    "wvlt_cross_spectrum",
]


def _diff_coord(coord):
    """Returns the difference as a `xarray.DataArray`."""

    v0 = coord.values[0]
    calendar = getattr(v0, "calendar", None)
    if calendar:
        import cftime

        ref_units = "seconds since 1800-01-01 00:00:00"
        decoded_time = cftime.date2num(coord, ref_units, calendar)
        coord = xr.DataArray(decoded_time, dims=coord.dims, coords=coord.coords)
        return np.diff(coord)
    elif pd.api.types.is_datetime64_dtype(v0):
        return np.diff(coord).astype("timedelta64[s]").astype("f8")
    else:
        return np.diff(coord)


def _delta(da, dim):
    """Returns the grid spacing"""

    delta_x = []
    for d in dim:
        diff = _diff_coord(da[d])
        delta = np.abs(diff[0])

        if delta == 0.0:
            raise ValueError(
                "Can't take wavelet transform because spacing in coordinate %s is zero"
                % d
            )
        delta_x.append(delta)

    return delta_x


def _morlet(t0, a, s, t):
    r"""
    Define

    .. math::
        \psi = a e^{-2\pi i f_0 t} e^{-\frac{t^2}{2 t_0^2}}

    as the morlet wavelet. Its transform is

    .. math::
        \psi_h = a 2\pi t_0^2 e^{-2 \pi^2 (f-f_0)^2 t_0^2}

    Units of :math:`a` are :math:`T^{-1}`.
    :math:`f_0` is defaulted to :math:`1/t_0`.
    """

    f0 = 1.0 / t0

    # rotated positions
    tp = s**-1 * t

    arg1 = 2j * np.pi * f0 * tp
    arg2 = -(t**2) / 2 / s**2 / t0**2
    m = a * np.exp(arg1) * np.exp(arg2)

    return m


def _morlet2(x0, ntheta, a, s, y, x, **kwargs):
    r"""
    Define

    .. math::
        \psi = a e^{-2\pi i \boldmath{k}_0 \cdot\boldmath{x}} e^{-\frac{\boldmath{x} \cdot \boldmath{x}}{2 x_0^2}}

    as the morlet wavelet. Its transform is

    .. math::
        \psi_h = a 2\pi x_0^2 e^{-2 \pi^2 (\boldmath{k}-\boldmath{k}_0)^2 x_0^2}

    Units of :math:`a` are :math:`L^{-2}`.
    :math:`k_0` is defaulted to :math:`1/x_0` in the zonal direction.
    """

    k0 = 1.0 / x0

    # compute morlet wavelet
    th = np.arange(int(ntheta / 2)) * 2.0 * np.pi / ntheta
    th = xr.DataArray(th, dims=["angle"], coords={"angle": th})

    if "angle" in kwargs:
        for k, v in kwargs.items():
            chunk = {k: v}
            th = th.chunk(chunk)

    # rotated positions
    yp = np.sin(th) * s**-1 * y
    xp = np.cos(th) * s**-1 * x

    arg1 = 2j * np.pi * k0 * (yp - xp)
    arg2 = -(x**2 + y**2) / 2 / s**2 / x0**2
    m = a * np.exp(arg1) * np.exp(arg2)

    return m, th


_xo_warning = "Input argument `xo` will be deprecated in the future versions of xwavelet and be replaced by `x0`"


def dwvlt(da, s, dim=None, xo=50e3, a=1.0, ntheta=16, wtype="morlet", **kwargs):
    """
    Deprecated function. See cwvlt2 doc.
    """
    msg = (
        "This function has been renamed and will disappear in the future."
        + " Please use `cwvlt2` instead."
    )
    warnings.warn(msg, FutureWarning)

    return cwvlt2(da, s, dim=dim, x0=xo, a=a, ntheta=ntheta, wtype=wtype, **kwargs)


def cwvlt(
    da,
    s,
    dim=None,
    t0=5 * 365 * 86400,
    a=1.0,
    wtype="morlet",
    tau=None,
):
    r"""
    Compute continuous one-dimensional wavelet transform of da. Default is the Morlet wavelet.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed.
    s : `xarray.DataArray`
        One-dimensional array with scaling parameter.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    t0 : float
        Characteristic time scale.
    a : float
        Amplitude of wavelet.
    wtype : str
        Type of wavelet.
    tau : float
        Coordinate where the wavelet is centered around.

    Returns
    -------
    dawt : `xarray.DataArray`
        The output of the wavelet transformation, with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if len(dim) != 1:
        raise ValueError("The transformed dimension should be one-dimensional.")

    sdim = s.dims[0]

    # the axes along which to take wavelets
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    delta_t = _delta(da, dim)

    # grid parameters
    if tau == None:
        tau = da[da.dims[axis_num[0]]].mean()
    else:
        if (
            type(tau) == np.datetime64
            or type(tau) == pd._libs.tslibs.timestamps.Timestamp
        ):
            raise ValueError("The units of `tau` should be in the metrical system.")

    t = da[da.dims[axis_num[0]]] - tau

    if wtype == "morlet":
        wavelet = _morlet(t0, a, s, t)
    else:
        raise NotImplementedError("Only the Morlet wavelet is implemented for now.")

    dawt = (da * np.conj(wavelet)).sum(dim, skipna=True) * delta_t / np.sqrt(np.abs(s))
    dawt = dawt.drop_vars(sdim)
    dawt[sdim] = t0 * s

    return dawt


def cwvlt2(da, s, dim=None, x0=50e3, a=1.0, ntheta=16, wtype="morlet", **kwargs):
    r"""
    Compute continuous two-dimensional wavelet transform of da. Default is the Morlet wavelet.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed.
    s : `xarray.DataArray`
        One-dimensional array with scaling parameter.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    x0 : float
        Characteristic length scale.
    a : float
        Amplitude of wavelet.
    ntheta : int
        Number of azimuthal angles the wavelet transform is taken over.
    wtype : str
        Type of wavelet.

    Returns
    -------
    dawt : `xarray.DataArray`
        The output of the wavelet transformation, with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if len(dim) != 2:
        raise ValueError("The transformed dimensions should be two-dimensional.")

    sdim = s.dims[0]

    # the axes along which to take wavelets
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    delta_x = _delta(da, dim)

    # grid parameters
    y = da[da.dims[axis_num[-2]]] - da[da.dims[axis_num[-2]]].mean()
    x = da[da.dims[axis_num[-1]]] - da[da.dims[axis_num[-1]]].mean()

    if wtype == "morlet":
        wavelet, phi = _morlet2(x0, ntheta, a, s, y, x, **kwargs)
    else:
        raise NotImplementedError("Only the Morlet wavelet is implemented for now.")

    dawt = (da * np.conj(wavelet)).sum(dim, skipna=True) * np.prod(delta_x) / s
    dawt = dawt.drop_vars(sdim)
    dawt[sdim] = x0 * s

    return dawt


def wvlt_power_spectrum(
    da, s, dim=None, x0=50e3, a=1.0, ntheta=16, wtype="morlet", normalize=True, **kwargs
):
    r"""
    Compute discrete wavelet power spectrum of :math:`da`.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to have the spectral estimate.
    s : `xarray.DataArray`
        Non-dimensional scaling parameter. The dimensionalized length scales are
        :math:`x0\times s`.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    x0 : float
        Length scale of the mother wavelet.
    a : float
        Amplitude of wavelet.
    ntheta : int
        Number of azimuthal angles the wavelet transform is taken over.
    wtype : str
        Type of wavelet.

    Returns
    -------
    ps : `xarray.DataArray`
        The output of the wavelet spectrum, with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if "xo" in kwargs:
        x0 = kwargs.get("xo")
        warnings.warn(_xo_warning, FutureWarning)

    if len(dim) == 1:
        dawt = cwvlt(
            da,
            s,
            dim=dim,
            t0=x0,
            a=a,
            wtype=wtype,
        )
    elif len(dim) == 2:
        dawt = cwvlt2(da, s, dim=dim, x0=x0, a=a, ntheta=ntheta, wtype=wtype, **kwargs)
    else:
        raise NotImplementedError(
            "Transformation for three dimensions and higher is not implemented."
        )

    if normalize:
        axis_num = [da.get_axis_num(d) for d in dim]
        N = [da.shape[n] for n in axis_num]
        delta_x = _delta(da, dim)

        Fdims = []
        chunks = dict()
        for d in dim:
            chunks[d] = -1
            Fdims.append("freq_" + d)

        if len(dim) == 1:
            t = da[da.dims[axis_num[0]]] - N[0] / 2.0 * delta_x[0]
            if wtype == "morlet":
                # mother wavelet
                wavelet = _morlet(x0, a, 1.0, t)
            Fw = xrft.fft(
                wavelet.chunk(chunks),
                dim=dim,
                true_phase=True,
                true_amplitude=True,
            )
        elif len(dim) == 2:
            y = da[da.dims[axis_num[-2]]] - N[-2] / 2.0 * delta_x[-2]
            x = da[da.dims[axis_num[-1]]] - N[-1] / 2.0 * delta_x[-1]
            if wtype == "morlet":
                # mother wavelet
                wavelet, phi = _morlet2(x0, ntheta, a, 1.0, y, x, **kwargs)
            Fw = xrft.fft(
                wavelet.isel(angle=0).chunk(chunks),
                dim=dim,
                true_phase=True,
                true_amplitude=True,
            )

        k2 = xr.zeros_like(Fw)
        for d in Fdims:
            k2 = k2 + Fw[d] ** 2
        dk = [np.diff(Fw[d]).data[0] for d in Fdims]
        C = (np.abs(Fw) ** 2 / k2 * np.prod(dk)).sum(Fdims, skipna=True).real

    else:
        C = 1.0

    if len(dim) == 1:
        return np.abs(dawt) ** 2 * x0 / C
    elif len(dim) == 2:
        return np.abs(dawt) ** 2 * (dawt[s.dims[0]]) ** -1 * x0**2 / C


def wvlt_cross_spectrum(
    da,
    da1,
    s,
    dim=None,
    x0=50e3,
    a=1.0,
    ntheta=16,
    wtype="morlet",
    normalize=True,
    **kwargs
):
    r"""
    Compute continuous wavelet cross spectrum of :math:`da` and :math:`da1`.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to have the cross spectral estimate.
    da1 : `xarray.DataArray`
        The data to have the cross spectral estimate.
    s : `xarray.DataArray`
        Non-dimensional scaling parameter. The dimensionalized length scales are
        :math:`x0\times s`.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    x0 : float
        Length scale of the mother wavelet.
    a : float
        Amplitude of wavelet.
    ntheta : int
        Number of azimuthal angles the wavelet transform is taken over.
    wtype : str
        Type of wavelet.

    Returns
    -------
    cs : `xarray.DataArray`
        The output of the wavelet spectrum, with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if "xo" in kwargs:
        x0 = kwargs.get("xo")
        warnings.warn(_xo_warning, FutureWarning)

    if len(dim) == 1:
        dawt = cwvlt(
            da,
            s,
            dim=dim,
            t0=x0,
            a=a,
            wtype=wtype,
        )
        dawt1 = cwvlt(
            da1,
            s,
            dim=dim,
            t0=x0,
            a=a,
            wtype=wtype,
        )
    elif len(dim) == 2:
        dawt = cwvlt2(da, s, dim=dim, x0=x0, a=a, ntheta=ntheta, wtype=wtype, **kwargs)
        dawt1 = cwvlt2(
            da1, s, dim=dim, x0=x0, a=a, ntheta=ntheta, wtype=wtype, **kwargs
        )
    else:
        raise NotImplementedError(
            "Transformation for three dimensions and higher is not implemented."
        )

    if normalize:
        axis_num = [da.get_axis_num(d) for d in dim]
        N = [da.shape[n] for n in axis_num]
        delta_x = _delta(da, dim)

        Fdims = []
        chunks = dict()
        for d in dim:
            chunks[d] = -1
            Fdims.append("freq_" + d)

        if len(dim) == 1:
            t = da[da.dims[axis_num[0]]] - N[0] / 2.0 * delta_x[0]
            if wtype == "morlet":
                # mother wavelet
                wavelet = _morlet(x0, a, 1.0, t)
            Fw = xrft.fft(
                wavelet.chunk(chunks),
                dim=dim,
                true_phase=True,
                true_amplitude=True,
            )
        elif len(dim) == 2:
            y = da[da.dims[axis_num[-2]]] - N[-2] / 2.0 * delta_x[-2]
            x = da[da.dims[axis_num[-1]]] - N[-1] / 2.0 * delta_x[-1]
            if wtype == "morlet":
                # mother wavelet
                wavelet, phi = _morlet2(x0, ntheta, a, 1.0, y, x, **kwargs)
            Fw = xrft.fft(
                wavelet.isel(angle=0).chunk(chunks),
                dim=dim,
                true_phase=True,
                true_amplitude=True,
            )

        k2 = xr.zeros_like(Fw)
        for d in Fdims:
            k2 = k2 + Fw[d] ** 2
        dk = [np.diff(Fw[d]).data[0] for d in Fdims]
        C = (np.abs(Fw) ** 2 / k2 * np.prod(dk)).sum(Fdims, skipna=True).real

    else:
        C = 1.0

    if len(dim) == 1:
        return (dawt * np.conj(dawt1)).real * x0 / C
    elif len(dim) == 2:
        return (dawt * np.conj(dawt1)).real * (dawt[s.dims[0]]) ** -1 * x0**2 / C
