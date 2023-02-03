import numpy as np
import pandas as pd
import xarray as xr
import cftime
import dask.array as dsar

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
import xwavelet


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
def test_isotropic_ps_slope(chunk, N=512, dL=1.0, amp=1e0, slope=-3.0, xo=5):
    """Test the spectral slope of isotropic power spectrum."""

    theta = synthetic_field_xr(
        N,
        dL,
        amp,
        slope,
        other_dim_sizes=[20],
        dim_order=True,
    )

    if chunk:
        theta = theta.chunk({"d0": 3, "y": 128, "x": 128})

    s = xr.DataArray(
        np.arange(0.5, 10.5, 0.5),
        dims=["scale"],
        coords={"scale": np.arange(0.5, 10.5, 0.5)},
    )

    Wtheta = xwavelet.dwvlt(theta, s, dim=["y", "x"], xo=xo)
    iso_ps = (np.abs(Wtheta) ** 2).mean(["d0", "angle"]) * (Wtheta.scale) ** -1
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog((iso_ps.scale.values[:]) ** -1, iso_ps.values[:])
    npt.assert_allclose(a, slope, atol=0.2)

    iso_ps = xwavelet.wvlt_power_spectrum(theta, s, dim=["y", "x"], xo=xo).mean(
        ["d0", "angle"]
    )
    npt.assert_almost_equal(np.ma.masked_invalid(iso_ps).mask.sum(), 0.0)
    y_fit, a, b = xrft.fit_loglog((xo * iso_ps.scale.values[:]) ** -1, iso_ps.values[:])
    npt.assert_allclose(a, slope, atol=0.2)
