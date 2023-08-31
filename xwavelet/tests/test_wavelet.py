import numpy as np
import pandas as pd
import xarray as xr
import dask.array as dsar

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
from xwavelet.wavelet import (
    dwvlt,
    cwvlt,
    cwvlt2,
    wvlt_power_spectrum,
    wvlt_cross_spectrum,
)


@pytest.fixture
def sample_da_1d():
    time = np.arange(0, 360 * 15 * 86400, 5 * 86400)
    freq0 = (180 * 86400) ** -1
    da = xr.DataArray(
        np.sin(2 * np.pi * freq0 * time), dims=["time"], coords={"time": time}
    )
    return da


@pytest.fixture
def sample_da_2d():
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.arange(11 * 17).reshape(17, 11)
    return xr.DataArray(z, dims=["y", "x"], coords={"y": y, "x": x})


@pytest.fixture
def sample_da_3d():
    x = np.linspace(0, 10, 11)
    y = np.linspace(-4, 4, 17)
    z = np.linspace(-4, 4, 9)
    w = np.arange(11 * 17 * 9).reshape(9, 17, 11)
    return xr.DataArray(w, dims=["z", "y", "x"], coords={"z": z, "y": y, "x": x})


def test_dimensions(sample_da_3d, sample_da_2d, sample_da_1d, x0=1.0):
    s = xr.DataArray(
        np.linspace(0.1, 1.0, 20),
        dims=["scale"],
        coords={"scale": np.linspace(0.1, 1.0, 20)},
    )
    with pytest.raises(ValueError):
        cwvlt(sample_da_2d, s, t0=(180 * 86400))
    with pytest.raises(ValueError):
        cwvlt2(sample_da_1d, s, x0=x0)
    with pytest.raises(NotImplementedError):
        wvlt_power_spectrum(sample_da_3d, s, x0=x0)
    with pytest.raises(NotImplementedError):
        wvlt_cross_spectrum(sample_da_3d, sample_da_3d, s, x0=x0)


def test_convergence(sample_da_2d, sample_da_1d, x0=1.0):
    s = xr.DataArray(
        np.linspace(0.1, 1.0, 20),
        dims=["scale"],
        coords={"scale": np.linspace(0.1, 1.0, 20)},
    )

    npt.assert_almost_equal(
        wvlt_power_spectrum(sample_da_2d, s, x0=x0).values,
        wvlt_cross_spectrum(sample_da_2d, sample_da_2d, s, x0=x0).values,
    )

    npt.assert_almost_equal(
        wvlt_power_spectrum(sample_da_1d, s, x0=(180 * 86400)).values,
        wvlt_cross_spectrum(sample_da_1d, sample_da_1d, s, x0=(180 * 86400)).values,
    )


def test_wtype(sample_da_2d, sample_da_1d, x0=1.0):
    s = xr.DataArray(
        np.linspace(0.1, 1.0, 20),
        dims=["scale"],
        coords={"scale": np.linspace(0.1, 1.0, 20)},
    )
    with pytest.raises(NotImplementedError):
        cwvlt2(sample_da_2d, s, x0=x0, wtype=None)
        cwvlt2(sample_da_2d, s, x0=x0, wtype="boxcar")
    with pytest.raises(NotImplementedError):
        cwvlt(sample_da_1d, s, t0=(180 * 86400) ** -1, wtype=None)
        cwvlt(sample_da_1d, s, t0=(180 * 86400) ** -1, wtype="boxcar")


def test_frequency(sample_da_1d, t0=(180 * 86400)):
    fda = xrft.power_spectrum(
        sample_da_1d,
    )
    s = (
        xr.DataArray(
            fda.freq_time[len(sample_da_1d.time) // 2 + 1 :].data ** -1,
            dims=["scale"],
            coords={
                "scale": fda.freq_time[len(sample_da_1d.time) // 2 + 1 :].data ** -1
            },
        )
        / t0
    )
    wda = wvlt_power_spectrum(sample_da_1d, s, x0=t0)

    npt.assert_equal(
        np.sort(wda.values.argsort()[-3:]),
        np.sort(fda.values[len(sample_da_1d.time) // 2 + 1 :].argsort()[-3:]),
    )


def synthetic_field(N, dL, amp, s):
    """
    Generate a synthetic series of size N by N
    with a spectral slope of s.
    """

    k = np.fft.fftshift(np.fft.fftfreq(N, dL))
    l = np.fft.fftshift(np.fft.fftfreq(N, dL))
    kk, ll = np.meshgrid(k, l)
    K = np.sqrt(kk**2 + ll**2)

    ########
    # amplitude
    ########
    r_kl = np.ma.masked_invalid(
        np.sqrt(amp * 0.5 * (np.pi) ** (-1) * K ** (s - 1.0))
    ).filled(0.0)
    ########
    # phase
    ########
    phi = np.zeros((N, N))

    N_2 = int(N / 2)
    phi_upper_right = 2.0 * np.pi * np.random.random((N_2 - 1, N_2 - 1)) - np.pi
    phi[N_2 + 1 :, N_2 + 1 :] = phi_upper_right.copy()
    phi[1:N_2, 1:N_2] = -phi_upper_right[::-1, ::-1].copy()

    phi_upper_left = 2.0 * np.pi * np.random.random((N_2 - 1, N_2 - 1)) - np.pi
    phi[N_2 + 1 :, 1:N_2] = phi_upper_left.copy()
    phi[1:N_2, N_2 + 1 :] = -phi_upper_left[::-1, ::-1].copy()

    phi_upper_middle = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[N_2:, N_2] = phi_upper_middle.copy()
    phi[1:N_2, N_2] = -phi_upper_middle[1:][::-1].copy()

    phi_right_middle = 2.0 * np.pi * np.random.random(N_2 - 1) - np.pi
    phi[N_2, N_2 + 1 :] = phi_right_middle.copy()
    phi[N_2, 1:N_2] = -phi_right_middle[::-1].copy()

    phi_edge_upperleft = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[N_2:, 0] = phi_edge_upperleft.copy()
    phi[1:N_2, 0] = -phi_edge_upperleft[1:][::-1].copy()

    phi_bot_right = 2.0 * np.pi * np.random.random(N_2) - np.pi
    phi[0, N_2:] = phi_bot_right.copy()
    phi[0, 1:N_2] = -phi_bot_right[1:][::-1].copy()

    phi_corner_leftbot = 2.0 * np.pi * np.random.random() - np.pi

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert phi[N_2 + j, N_2 + i] == -phi[N_2 - j, N_2 - i]

    for i in range(1, N_2):
        for j in range(1, N_2):
            assert phi[N_2 + j, N_2 - i] == -phi[N_2 - j, N_2 + i]

    for i in range(1, N_2):
        assert phi[N_2, N - i] == -phi[N_2, i]
        assert phi[N - i, N_2] == -phi[i, N_2]
        assert phi[N - i, 0] == -phi[i, 0]
        assert phi[0, i] == -phi[0, N - i]
    #########
    # complex fourier amplitudes
    #########
    F_theta = r_kl * np.exp(1j * phi)

    # check that symmetry of FT is satisfied
    theta = np.fft.ifft2(np.fft.ifftshift(F_theta))
    return np.real(theta)


def synthetic_field_xr(
    N, dL, amp, s, other_dim_sizes=None, dim_order=True, chunks=None
):
    theta = xr.DataArray(
        synthetic_field(N, dL, amp, s),
        dims=["y", "x"],
        coords={"y": range(N), "x": range(N)},
    )

    if other_dim_sizes:
        _da = xr.DataArray(
            np.ones(other_dim_sizes),
            dims=["d%d" % i for i in range(len(other_dim_sizes))],
        )
        if dim_order:
            theta = theta + _da
        else:
            theta = _da + theta

    if chunks:
        theta = theta.chunk(chunks)

    return theta


@pytest.mark.parametrize("chunk", [False, True])
def test_isotropic_ps_slope(chunk, N=256, dL=1.0, amp=1e0, slope=-3.0, xo=50):
    """Test the spectral slope of isotropic power spectrum."""

    theta = synthetic_field_xr(
        N,
        dL,
        amp,
        slope,
        other_dim_sizes=[30],
        dim_order=True,
    )

    if chunk:
        theta = theta.chunk({"d0": 5, "y": 64, "x": 64})

    freq_r = xrft.isotropic_power_spectrum(
        theta.chunk({"y": -1, "x": -1}), dim=["y", "x"], truncate=True
    ).freq_r
    s = xr.DataArray(
        freq_r.data[1::2] ** -1 / xo,
        dims=["scale"],
        coords={"scale": freq_r.data[1::2] ** -1 / xo},
    ).chunk({"scale": -1})

    kwargs = {"angle": 2}

    Wtheta = dwvlt(theta, s, dim=["y", "x"], xo=xo, **kwargs)
    npt.assert_allclose(
        Wtheta.values, cwvlt2(theta, s, dim=["y", "x"], x0=xo, **kwargs).values
    )

    iso_ps = (np.abs(Wtheta) ** 2).mean(["d0", "angle"]) * (Wtheta.scale) ** -1
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog(
        (iso_ps.scale.values[2:-1]) ** -1, iso_ps.values[2:-1]
    )
    npt.assert_allclose(a, slope, atol=0.3)

    iso_ps = wvlt_power_spectrum(theta, s, dim=["y", "x"], x0=xo, **kwargs).mean(
        ["d0", "angle"]
    )
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog(
        (iso_ps.scale.values[2:-1]) ** -1, iso_ps.values[2:-1]
    )
    npt.assert_allclose(a, slope, atol=0.3)


#
#     if chunk:
#     	iso_ps = wvlt_power_spectrum(theta, s, dim=["y", "x"], xo=xo, **kwargs).mean(["d0", "angle"])
#     	npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
#     	y_fit, a, b = xrft.fit_loglog(
#         	(iso_ps.scale.values[1:-2]) ** -1, iso_ps.values[1:-2]
#     	)
#     	npt.assert_allclose(a, slope, atol=0.3)
