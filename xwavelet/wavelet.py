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


def _delta(da, dim, spacing_tol):
    """Returns the grid spacing"""

    delta_x = []
    for d in dim:
        diff = _diff_coord(da[d])
        delta = np.abs(diff[0])
        if not np.allclose(diff, diff[0], rtol=spacing_tol):
            raise ValueError(
                "Can't take wavelet transform because "
                "coodinate %s is not evenly spaced" % d
            )
        if delta == 0.0:
            raise ValueError(
                "Can't take wavelet transform because spacing in coordinate %s is zero"
                % d
            )
        delta_x.append(delta)

    return delta_x


def _morlet(xo, ntheta, a, s, y, x, dim):
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
    ko = 1.0 / xo

    # compute morlet wavelet
    th = np.arange(int(ntheta / 2)) * 2.0 * np.pi / ntheta
    th = xr.DataArray(th, dims=["angle"], coords={"angle": th})

    # rotated positions
    yp = np.sin(th) * s ** -1 * y
    xp = np.cos(th) * s ** -1 * x

    arg1 = 2j * np.pi * ko * (yp - xp)
    arg2 = -(x ** 2 + y ** 2) / 2 / s ** 2 / xo ** 2
    m = a * np.exp(arg1) * np.exp(arg2)

    return m, th


def dwvlt(da, s, spacing_tol=1e-3, dim=None, xo=50e3, a=1.0, ntheta=16, wtype="morlet"):
    r"""
    Compute discrete wavelet transform of da. Default is the Morlet wavelet.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed.
    s : `xarray.DataArray`
        One-dimensional array with scaling parameter.
    spacing_tol : float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    xo : float
        Length scale.
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
    wavelet : `xarray.DataArray`
        The wavelet with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    sdim = s.dims[0]

    # the axes along which to take wavelets
    axis_num = [da.get_axis_num(d) for d in dim]

    N = [da.shape[n] for n in axis_num]

    # verify even spacing of input coordinates
    delta_x = _delta(da, dim, spacing_tol)

    # grid parameters
    if len(dim) == 2:
        y = da[da.dims[axis_num[-2]]] - N[-2] / 2.0 * delta_x[-2]
        x = da[da.dims[axis_num[-1]]] - N[-1] / 2.0 * delta_x[-1]
    else:
        raise NotImplementedError(
            "Only two-dimensional transforms are implemented for now."
        )

    if wtype == "morlet":
        wavelet, phi = _morlet(xo, ntheta, a, s, y, x, dim)
    else:
        raise NotImplementedError("Only the Morlet wavelet is implemented for now.")

    dawt = (da * np.conj(wavelet)).sum(dim, skipna=True) * np.prod(delta_x) / s
    dawt = dawt.drop_vars(sdim)
    dawt[sdim] = xo*s

    return dawt


def wvlt_power_spectrum(
    da,
    s,
    spacing_tol=1e-3,
    dim=None,
    xo=50e3,
    a=1.0,
    ntheta=16,
    wtype="morlet",
    normalize=True,
):
    r"""
    Compute discrete wavelet power spectrum of :math:`da`.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to have the spectral estimate.
    s : `xarray.DataArray`
        Scaling parameter.
    spacing_tol : float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    xo : float
        Length scale.
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

    dawt = dwvlt(
        da, s, spacing_tol=spacing_tol, dim=dim, xo=xo, a=a, ntheta=ntheta, wtype=wtype
    )

    if normalize:

        axis_num = [da.get_axis_num(d) for d in dim]
        N = [da.shape[n] for n in axis_num]
        delta_x = _delta(da, dim, spacing_tol)

        y = da[da.dims[axis_num[-2]]] - N[-2] / 2.0 * delta_x[-2]
        x = da[da.dims[axis_num[-1]]] - N[-1] / 2.0 * delta_x[-1]
        if wtype == "morlet":
            # mother wavelet
            wavelet, phi = _morlet(xo, ntheta, a, 1.0, y, x, dim)

        Fdims = []
        chunks = dict()
        for d in dim:
            chunks[d] = -1
            Fdims.append("freq_" + d)

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

    return np.abs(dawt) ** 2 * (dawt[s.dims[0]]) ** -1 * xo ** 2 * C ** -2


def wvlt_cross_spectrum(
    da,
    da1,
    s,
    spacing_tol=1e-3,
    dim=None,
    xo=50e3,
    a=1.0,
    ntheta=16,
    wtype="morlet",
    normalize=True,
):
    r"""
    Compute discrete wavelet cross spectrum of :math:`da` and :math:`da1`.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to have the cross spectral estimate.
    da1 : `xarray.DataArray`
        The data to have the cross spectral estimate.
    s : `xarray.DataArray`
        Scaling parameter.
    spacing_tol : float, optional
        Spacing tolerance. Fourier transform should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    xo : float
        Length scale.
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

    dawt = dwvlt(
        da, s, spacing_tol=spacing_tol, dim=dim, xo=xo, a=a, ntheta=ntheta, wtype=wtype
    )
    dawt1 = dwvlt(
        da1, s, spacing_tol=spacing_tol, dim=dim, xo=xo, a=a, ntheta=ntheta, wtype=wtype
    )

    if normalize:

        axis_num = [da.get_axis_num(d) for d in dim]
        N = [da.shape[n] for n in axis_num]
        delta_x = _delta(da, dim, spacing_tol)

        y = da[da.dims[axis_num[-2]]] - N[-2] / 2.0 * delta_x[-2]
        x = da[da.dims[axis_num[-1]]] - N[-1] / 2.0 * delta_x[-1]
        if wtype == "morlet":
            # mother wavelet
            wavelet, phi = _morlet(xo, ntheta, a, 1.0, y, x, dim)

        Fdims = []
        chunks = dict()
        for d in dim:
            chunks[d] = -1
            Fdims.append("freq_" + d)

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

    return (dawt * np.conj(dawt1)).real * (dawt[s.dims[0]]) ** -1 * xo ** 2 * C ** -2
