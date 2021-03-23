import cv2
import numpy as np

def FourierShift2D(x, delta):
    """
    FourierShift2D(x, delta)
        Subpixel shifting in python. Based on the original script (FourierShift2D.m)
        by Tim Hutt.

        Original Description
        --------------------
        Shifts x by delta cyclically. Uses the fourier shift theorem.
        Real inputs should give real outputs.
        By Tim Hutt, 26/03/2009
        Small fix thanks to Brian Krause, 11/02/2010

    Parameters
    ----------
    `x`: Numpy Array, required
        The 2D matrix that is to be shifted. Can be real/complex valued.

    `delta`: Iterable, required
        The amount of shift to be done in x and y directions. The 0th index should be
        the shift in the x direction, and the 1st index should be the shift in the y
        direction.

        For e.g., For a shift of +2 in x direction and -3 in y direction,
            delta = [2, -3]

    Returns
    -------
    `y`: The input matrix `x` shifted by the `delta` amount of shift in the
        corresponding directions.
    """
    # The size of the matrix.
    N, M = x.shape

    # FFT of our possibly padded input signal.
    X = np.fft.fft2(x)

    # The mathsy bit. The floors take care of odd-length signals.
    y_arr = np.hstack([
        np.arange(np.floor(N / 2), dtype=np.int),
        np.arange(np.floor(-N / 2), 0, dtype=np.int)
    ])

    x_arr = np.hstack([
        np.arange(np.floor(M / 2), dtype=np.int),
        np.arange(np.floor(-M / 2), 0, dtype=np.int)
    ])

    y_shift = np.exp(-1j * 2 * np.pi * delta[0] * x_arr / N)
    x_shift = np.exp(-1j * 2 * np.pi * delta[1] * y_arr / M)

    y_shift = y_shift[None, :]  # Shape = (1, N)
    x_shift = x_shift[:, None]  # Shape = (M, 1)

    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if np.mod(N, 2) == 0:
        x_shift[N // 2] = np.real(x_shift[N // 2])

    if np.mod(M, 2) == 0:
        y_shift[:, M // 2] = np.real(y_shift[:, M // 2])

    Y = X * (x_shift * y_shift)

    # Invert the FFT.
    y = np.fft.ifft2(Y)

    # There should be no imaginary component (for real input
    # signals) but due to numerical effects some remnants remain.
    if np.isrealobj(x):
        y = np.real(y)

    return y

# from : https://gist.github.com/IAmSuyogJadhav/6b659413dc821d2fb00f290a189da9c1