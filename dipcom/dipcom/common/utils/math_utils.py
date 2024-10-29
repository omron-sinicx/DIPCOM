#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2025 OMRON SINIC X
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Malek Aburub, Cristian C. Beltran-Hernandez

import numpy as np
from numba import jit
import timeit


def cholesky2diag(cholesky):
    return np.diag(cholesky_vector_to_spd(cholesky))


def stiff2cholesky(diag):
    spd_matrix = np.diag(diag)
    return spd_to_cholesky_vector(spd_matrix)

# @jit("float64[:](float64[:,:])", nopython=True)


def spd_to_cholesky_vector(spd_matrix):
    """
        Compute Cholesky decomposition for SPD matrix.
        Then, extract and return its lower triangle as a vector.
    """
    cholesky_matrix = np.linalg.cholesky(spd_matrix).ravel()
    tril_mask = np.array([0, 3, 4, 6, 7, 8])
    return cholesky_matrix[tril_mask]

# @jit("float64[:,:](float64[:])", nopython=True)


def cholesky_vector_to_spd(cholesky_vector):
    """
        Reconstruct Cholesky decomposition matrix from vector.
        Then compute SPD matrix L * L.T
    """
    cholesky_matrix = np.zeros((3, 3), dtype=np.float64)
    mask = np.tril_indices(cholesky_matrix.shape[0], k=0)
    for i in range(6):
        cholesky_matrix[mask[0][i]][mask[1][i]] = cholesky_vector[i]
    return cholesky_matrix @ cholesky_matrix.T


if __name__ == '__main__':
    from sklearn.datasets import make_spd_matrix
    spd = make_spd_matrix(3)
    print("SPD Matrix: \n", spd, type(spd))
    ch = np.linalg.cholesky(spd)
    print("Cholesky decomposition L:\n", ch)
    print("SPD from L @ L.T\n", np.dot(ch, ch.T))

    print("=== Verification ===")
    ch_v = spd_to_cholesky_vector(spd)
    print("cholesky vector\n", ch_v)
    spd_rec = cholesky_vector_to_spd(ch_v)
    print("reconstructed spd\n", spd_rec)