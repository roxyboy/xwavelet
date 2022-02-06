import warnings
import operator
import sys
import functools as ft
from functools import reduce

import numpy as np
import xarray as xr
import pandas as pd
import xoa.filter as xoaf


__all__ = [
    "dwvlt",
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
    ko = 1./xo

    # compute morlet wavelet
    th = np.arange(int(ntheta/2))*2.*np.pi/ntheta
    th = xr.DataArray(th, dims=['angle'], coords={'angle':th})

    # rotated positions
    yp = np.sin(th)*s**-1*y
    xp = np.cos(th)*s**-1*x

    arg1 = -2.*np.pi*1j*ko*(xp-yp)
    arg2 = -(x**2+y**2)/2/s**2/xo**2
    m = a*np.exp(arg1)*np.exp(arg2)

    return m


def dwvlt(
    da,
    s,
    spacing_tol=1e-3,
    dim=None,
    xo=50e3,
    a=1.,
    ntheta=16,
    wtype="morlet"
):
    r"""
    Compute discrete wavelet transform of da. Default is the Morlet wavelet.
    Scale :math:`s` is dimensionless.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data to be transformed.
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
    dawt : `xarray.DataArray`
        The output of the wavelet transformation, with appropriate dimensions.
    """

    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    # the axes along which to take wavelets
    axis_num = [
        da.get_axis_num(d) for d in dim
    ]

    N = [da.shape[n] for n in axis_num]

    # verify even spacing of input coordinates
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

    # grid parameters
    if len(dim) == 2:
        x = xr.DataArray((np.arange(1,N[1]+1) - N[1]/2)*delta_x[1],
                         dims=dim[1], coords={dim[1]:da[dim[1]]})
        y = xr.DataArray((np.arange(1,N[0]+1) - N[0]/2)*delta_x[0],
                         dims=dim[0], coords={dim[1]:da[dim[0]]})
        # Lx = N[1]*dx
        # Ly = N[0]*dy
        pass
    else:
        raise NotImplementedError(
            "Only two-dimensional transforms are implemented for now."
        )

    if wtype == 'morlet':
        wavelet = _morlet(xo, ntheta, a, s, y, x, dim)
    else:
        raise NotImplementedError(
            "Only the Morlet wavelet is implemented for now."
        )

    return (xoaf.convolve(da, np.conj(wavelet)).sum(dim, skipna=True)
            * np.prod(delta_x) / s
    )


def wvlt_spectrum(da, s, **kwargs):

    dawt = dwvlt(da, s,
                dim=dim,
                xo=xo,
                a=a,
                ntheta=ntheta,
                wtype=wtype
    )

    return (dawt * np.conj(dawt)).real
