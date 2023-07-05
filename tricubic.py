# -*- coding: utf-8 -*-

__author__  = 'Fabian Gittins'
__date__    = '05/07/2023'

import numpy as np

class tricubic(object):
    """
    A tricubic interpolator in 3 dimensions.
    
    Parameters
    ----------
    X, Y, Z : (nx,), (ny,), (nz,) array_like
        Points defining grid in 3 dimensions.
    F : (nx, ny, nz) array_like
        Values of function on grid to interpolate.

    Methods
    -------
    __call__
    partial_derivative

    Notes
    -----
    Based on Lekien and Marsden (2005), "Tricubic interpolation in three 
    dimensions," Int. J. Numer. Meth. Eng. 63, 455.
    """
    def __init__(self, X, Y, Z, F):
        X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(Z)
        F = np.asarray(F)
        if X.ndim != 1:
            raise ValueError('X must be one dimensional')
        if Y.ndim != 1:
            raise ValueError('Y must be one dimensional')
        if Z.ndim != 1:
            raise ValueError('Z must be one dimensional')
        if F.ndim != 3:
            raise ValueError('F must be three dimensional')
        if X.size != F.shape[0]:
            raise ValueError('X-dimension of F must have same number of '
                             'elements as X')
        if Y.size != F.shape[1]:
            raise ValueError('Y-dimension of F must have same number of '
                             'elements as Y')
        if Z.size != F.shape[2]:
            raise ValueError('Z-dimension of F must have same number of '
                             'elements as Z')

        self.X = X
        self.Y = Y
        self.Z = Z
        self.F = F

        self._Xmin, self._Xmax = X.min(), X.max()
        self._Ymin, self._Ymax = Y.min(), Y.max()
        self._Zmin, self._Zmax = Z.min(), Z.max()

        self._dFdX = self._build_dFdX(F, X)
        self._dFdY = self._build_dFdY(F, Y)
        self._dFdZ = self._build_dFdZ(F, Z)
        self._d2FdXdY = self._build_dFdX(self._dFdY, X)
        self._d2FdXdZ = self._build_dFdX(self._dFdZ, X)
        self._d2FdYdZ = self._build_dFdY(self._dFdZ, Y)
        self._d3FdXdYdZ = self._build_dFdX(self._d2FdYdZ, X)

        self._initialised = False
        self._Xi, self._Xiplus1 = None, None
        self._Yj, self._Yjplus1 = None, None
        self._Zk, self._Zkplus1 = None, None
        self._alpha = None
        self._Binv = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [9, -9, -9, 0, 9, 0, 0, 0, 6, 3, -6, 0, -3, 0, 0, 0, 
                 6, -6, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 4, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-6, 6, 6, 0, -6, 0, 0, 0, -3, -3, 3, 0, 3, 0, 0, 0, 
                 -4, 4, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, -2, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-6, 6, 6, 0, -6, 0, 0, 0, -4, -2, 4, 0, 2, 0, 0, 0, 
                 -3, 3, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, -1, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, -4, -4, 0, 4, 0, 0, 0, 2, 2, -2, 0, -2, 0, 0, 0, 
                 2, -2, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 0, 9, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, 0, -3, 0, 0, 0, 
                 6, -6, 3, 0, -3, 0, 0, 0, 4, 2, 2, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, 0, -6, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 0, 3, 0, 0, 0, 
                 -4, 4, -2, 0, 2, 0, 0, 0, -2, -2, -1, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, 0, -6, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 0, 2, 0, 0, 0, 
                 -3, 3, -3, 0, 3, 0, 0, 0, -2, -1, -2, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 0, 4, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, 0, -2, 0, 0, 0, 
                 2, -2, 2, 0, -2, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [-3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [9, -9, 0, -9, 0, 9, 0, 0, 6, 3, 0, -6, 0, -3, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 3, 0, -3, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 2, 0, 1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-6, 6, 0, 6, 0, -6, 0, 0, -3, -3, 0, 3, 0, 3, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, -2, 0, 2, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, -1, 0, -1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 9, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 6, 3, 0, -6, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 6, -6, 0, 3, 0, -3, 0, 0, 4, 2, 0, 2, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -6, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, -3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -4, 4, 0, -2, 0, 2, 0, 0, -2, -2, 0, -1, 0, -1, 0, 0],
                [9, 0, -9, -9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 6, 0, 3, -6, 0, 0, -3, 0, 6, 0, -6, 3, 0, 0, -3, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 4, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, -9, 0, 0, 9, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 6, 0, 3, -6, 0, 0, -3, 0, 6, 0, -6, 3, 0, 0, -3, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 2, 0, 0, 1, 0],
                [-27, 27, 27, 27, -27, -27, -27, 27, -18, -9, 18, 18, 9, 9, -18, -9, 
                 -18, 18, -9, 18, 9, -18, 9, -9, -18, 18, 18, -9, -18, 9, 9, -9, 
                 -12, -6, -6, 12, -3, 6, 6, 3, -12, -6, 12, -6, 6, -3, 6, 3, 
                 -12, 12, -6, -6, 6, 6, -3, 3, -8, -4, -4, -4, -2, -2, -2, -1],
                [18, -18, -18, -18, 18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 
                 12, -12, 6, -12, -6, 12, -6, 6, 12, -12, -12, 6, 12, -6, -6, 6, 
                 6, 6, 3, -6, 3, -6, -3, -3, 6, 6, -6, 3, -6, 3, -3, -3, 
                 8, -8, 4, 4, -4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1],
                [-6, 0, 6, 6, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 0, -3, 3, 0, 0, 3, 0, -4, 0, 4, -2, 0, 0, 2, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, -2, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 6, 0, 0, -6, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 0, -3, 3, 0, 0, 3, 0, -4, 0, 4, -2, 0, 0, 2, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, -1, 0, 0, -1, 0],
                [18, -18, -18, -18, 18, 18, 18, -18, 12, 6, -12, -12, -6, -6, 12, 6, 
                 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 6, 12, -6, -6, 6, 
                 6, 3, 6, -6, 3, -3, -6, -3, 8, 4, -8, 4, -4, 2, -4, -2, 
                 6, -6, 6, 3, -6, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1],
                [-12, 12, 12, 12, -12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, 
                 -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -4, -8, 4, 4, -4, 
                 -3, -3, -3, 3, -3, 3, 3, 3, -4, -4, 4, -2, 4, -2, 2, 2, 
                 -4, 4, -4, -2, 4, 2, -2, 2, -2, -2, -2, -1, -2, -1, -1, -1],
                [2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-6, 6, 0, 6, 0, -6, 0, 0, -4, -2, 0, 4, 0, 2, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, -3, 0, 3, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, -2, 0, -1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, -4, 0, -4, 0, 4, 0, 0, 2, 2, 0, -2, 0, -2, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 2, 0, -2, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -6, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -4, -2, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -3, 3, 0, -3, 0, 3, 0, 0, -2, -1, 0, -2, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 4, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, 2, 0, -2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, -2, 0, 2, 0, -2, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                [-6, 0, 6, 6, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -4, 0, -2, 4, 0, 0, 2, 0, -3, 0, 3, -3, 0, 0, 3, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -2, 0, -1, -2, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 6, 0, 0, -6, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 -4, 0, -2, 4, 0, 0, 2, 0, -3, 0, 3, -3, 0, 0, 3, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, -2, 0, 0, -1, 0],
                [18, -18, -18, -18, 18, 18, 18, -18, 12, 6, -12, -12, -6, -6, 12, 6, 
                 12, -12, 6, -12, -6, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 
                 8, 4, 4, -8, 2, -4, -4, -2, 6, 3, -6, 6, -3, 3, -6, -3, 
                 6, -6, 3, 6, -3, -6, 3, -3, 4, 2, 2, 4, 1, 2, 2, 1],
                [-12, 12, 12, 12, -12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, 
                 -8, 8, -4, 8, 4, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, 
                 -4, -4, -2, 4, -2, 4, 2, 2, -3, -3, 3, -3, 3, -3, 3, 3, 
                 -4, 4, -2, -4, 2, 4, -2, 2, -2, -2, -1, -2, -1, -2, -1, -1],
                [4, 0, -4, -4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, 0, 2, -2, 0, 0, -2, 0, 2, 0, -2, 2, 0, 0, -2, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, -4, 0, 0, 4, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 2, 0, 2, -2, 0, 0, -2, 0, 2, 0, -2, 2, 0, 0, -2, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                [-12, 12, 12, 12, -12, -12, -12, 12, -8, -4, 8, 8, 4, 4, -8, -4, 
                 -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, 
                 -4, -2, -4, 4, -2, 2, 4, 2, -4, -2, 4, -4, 2, -2, 4, 2, 
                 -3, 3, -3, -3, 3, 3, -3, 3, -2, -1, -2, -2, -1, -1, -2, -1],
                [8, -8, -8, -8, 8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 
                 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 
                 2, 2, 2, -2, 2, -2, -2, -2, 2, 2, -2, 2, -2, 2, -2, -2, 
                 2, -2, 2, 2, -2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

    @staticmethod
    def _build_dFdX(F, X):
        """
        Three-point finite difference formula for 
        :math:`\partial f / \partial x`.

        Parameters
        ----------
        F : (nx, ny, nz) array_like
            Values to finite difference.
        X : (nx,) array_like
            Points defining grid in x-direction.

        Returns
        -------
        dFdX : (nx, ny, nz) array_like
            Estimate of derivative.
        """
        dFdX = np.zeros_like(F)
        for i in range(1, len(X) - 1):
            dFdX[i, :, :] = (F[i + 1, :, :] - F[i - 1, :, :])/(X[i + 1] - X[i - 1])
        dFdX[0, :, :] = (F[2, :, :] - 3*F[1, :, :] + 2*F[0, :, :])/(X[2] - 3*X[1] + 2*X[0])
        dFdX[-1, :, :] = (2*F[-1, :, :] - 3*F[-2, :, :] + F[-3, :, :])/(2*X[-1] - 3*X[-2] + X[-3])
        return dFdX

    @staticmethod
    def _build_dFdY(F, Y):
        """
        Three-point finite difference formula for 
        :math:`\partial f / \partial y`.

        Parameters
        ----------
        F : (nx, ny, nz) array_like
            Values to finite difference.
        Y : (ny,) array_like
            Points defining grid in y-direction.

        Returns
        -------
        dFdY : (nx, ny, nz) array_like
            Estimate of derivative.
        """
        dFdY = np.zeros_like(F)
        for j in range(1, len(Y) - 1):
            dFdY[:, j, :] = (F[:, j + 1, :] - F[:, j - 1, :])/(Y[j + 1] - Y[j - 1])
        dFdY[:, 0, :] = (F[:, 2, :] - 3*F[:, 1, :] + 2*F[:, 0, :])/(Y[2] - 3*Y[1] + 2*Y[0])
        dFdY[:, -1, :] = (2*F[:, -1, :] - 3*F[:, -2, :] + F[:, -3, :])/(2*Y[-1] - 3*Y[-2] + Y[-3])
        return dFdY

    @staticmethod
    def _build_dFdZ(F, Z):
        """
        Three-point finite difference formula for 
        :math:`\partial f / \partial z`.

        Parameters
        ----------
        F : (nx, ny, nz) array_like
            Values to finite difference.
        Z : (nz,) array_like
            Points defining grid in z-direction.

        Returns
        -------
        dFdZ : (nx, ny, nz) array_like
            Estimate of derivative.
        """
        dFdZ = np.zeros_like(F)
        for k in range(1, len(Z) - 1):
            dFdZ[:, :, k] = (F[:, :, k + 1] - F[:, :, k - 1])/(Z[k + 1] - Z[k - 1])
        dFdZ[:, :, 0] = (F[:, :, 2] - 3*F[:, :, 1] + 2*F[:, :, 0])/(Z[2] - 3*Z[1] + 2*Z[0])
        dFdZ[:, :, -1] = (2*F[:, :, -1] - 3*F[:, :, -2] + F[:, :, -3])/(2*Z[-1] - 3*Z[-2] + Z[-3])
        return dFdZ

    def _calculate_coefficients(self, i0, j0, k0):
        """
        Calculate vector of coefficients :math:`\alpha` for interpolation, which 
        is obtained from linear equation :math:`\alpha = B^{-1} b`.

        Parameters
        ----------
        i0, j0, k0 : int
            Indices of position.

        Notes
        -----
        Vector :math:`b` is constructed out of following set:

        .. math:: 
            \left\{ f, \frac{\partial f}{\partial x}, 
            \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}, 
            \frac{\partial^2 f}{\partial x \partial y}, 
            \frac{\partial^2 f}{\partial x \partial z}, 
            \frac{\partial^2 f}{\partial y \partial z}, 
            \frac{\partial^3 f}{\partial x \partial y \partial z} \right\}.

        Each of these quantities is evaluated at 8 corners of cube in following 
        order: `(i0, j0, k0)`, `(i0 + 1, j0, k0)`, `(i0, j0 + 1, k0)`, 
        `(i0, j0, k0 + 1)`, `(i0 + 1, j0 + 1, k0)`, `(i0 + 1, j0, k0 + 1)`, 
        `(i0, j0 + 1, k0 + 1)`, `(i0 + 1, j0 + 1, k0 + 1)`. A different order 
        would require tweaking matrix `Binv`.
        """
        b = np.array([
            #####
            self.F[i0, j0, k0],
            self.F[i0 + 1, j0, k0], 
            self.F[i0, j0 + 1, k0],
            self.F[i0, j0, k0 + 1], 
            self.F[i0 + 1, j0 + 1, k0],
            self.F[i0 + 1, j0, k0 + 1], 
            self.F[i0, j0 + 1, k0 + 1],
            self.F[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self._dFdX[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0 + 1, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0 + 1, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0 + 1, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0, j0 + 1, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self._dFdY[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0, j0, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0 + 1, j0, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0, j0 + 1, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0, j0, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0 + 1, j0 + 1, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0 + 1, j0, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0, j0 + 1, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self._dFdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self._d2FdXdY[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdXdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0 + 1, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0 + 1, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0 + 1, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0, j0 + 1, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d2FdYdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self._d3FdXdYdZ[i0 + 1, j0 + 1, k0 + 1]
        ])

        # flag for first interpolation
        self._initialised = True

        # record location of cube
        self._Xi, self._Xiplus1 = self.X[i0], self.X[i0 + 1]
        self._Yj, self._Yjplus1 = self.Y[j0], self.Y[j0 + 1]
        self._Zk, self._Zkplus1 = self.Z[k0], self.Z[k0 + 1]

        return np.dot(self._Binv, b)
    
    def __call__(self, x, y, z, dx=False, dy=False, dz=False):
        """
        Evaluate tricubic interpolator or (first) partial derivative at 
        position `(x, y, z)`.

        Parameters
        ----------
        x, y, z : float
            Position arguments.
        dx, dy, dz : bool
            Which variable to take derivative with respect to.
        
        Returns
        -------
        f : float
            Interpolated values.
        """
        if x < self._Xmin or self._Xmax < x:
            raise ValueError('x must lie within grid defined by X')
        if y < self._Ymin or self._Ymax < y:
            raise ValueError('y must lie within grid defined by Y')
        if z < self._Zmin or self._Zmax < z:
            raise ValueError('z must lie within grid defined by Z')
        if dx or dy or dz:
            return self.partial_derivative(x, y, z, dx, dy, dz)

        # check if coefficients from last interpolation can be re-used
        if (not self._initialised or self._Xi <= x < self._Xiplus1 
            or self._Yj <= y < self._Yjplus1 or self._Zk <= z < self._Zkplus1):
            # find new origin of cube
            i0 = np.where(self.X <= x)[0][-1]
            j0 = np.where(self.Y <= y)[0][-1]
            k0 = np.where(self.Z <= z)[0][-1]

            # cheap and cheerful fix for evaluations at final grid points
            if x == self._Xmax:
                i0 -= 1
            if y == self._Ymax:
                j0 -= 1
            if z == self._Zmax:
                k0 -= 1

            self._alpha = self._calculate_coefficients(i0, j0, k0)

        # evaluate tricubic function
        f = 0
        for a in range(4):
            for c in range(4):
                for d in range(4):
                    f += (self._alpha[a + 4*c + 16*d]
                          *((x - self._Xi)/(self._Xiplus1 - self._Xi))**a
                          *((y - self._Yj)/(self._Yjplus1 - self._Yj))**c
                          *((z - self._Zk)/(self._Zkplus1 - self._Zk))**d)
        return f
    
    def partial_derivative(self, x, y, z, dx=True, dy=False, dz=False):
        """
        Evaluate (first) partial derivative of tricubic interpolator with 
        respect to `x` (:math:`\partial f / \partial x`), `y` 
        (:math:`\partial f / \partial y`) or `z` 
        (:math:`\partial f / \partial z`) at `(x, y, z)`. Defaults to partial 
        derivative with respect to `x`.

        Parameters
        ----------
        x, y, z : float
            Position arguments.
        dx, dy, dz : bool
            Which variable to take derivative with respect to.
        
        Returns
        -------
        df : float
            Interpolated values of derivative.
        """
        if x < self._Xmin or self._Xmax < x:
            raise ValueError('x must lie within grid defined by X')
        if y < self._Ymin or self._Ymax < y:
            raise ValueError('y must lie within grid defined by Y')
        if z < self._Zmin or self._Zmax < z:
            raise ValueError('z must lie within grid defined by Z')
        if dx + dy + dz > 1:
            raise ValueError('Only one of `dx`, `dy`, `dz` can be True')

        # check if coefficients from last interpolation can be re-used
        if (not self._initialised or self._Xi <= x < self._Xiplus1 
            or self._Yj <= y < self._Yjplus1 or self._Zk <= z < self._Zkplus1):
            # find new origin of cube
            i0 = np.where(self.X <= x)[0][-1]
            j0 = np.where(self.Y <= y)[0][-1]
            k0 = np.where(self.Z <= z)[0][-1]

            # cheap and cheerful fix for evaluations at final grid points
            if x == self._Xmax:
                i0 -= 1
            if y == self._Ymax:
                j0 -= 1
            if z == self._Zmax:
                k0 -= 1

            self._alpha = self._calculate_coefficients(i0, j0, k0)

        # evaluate derivative of tricubic function
        df = 0
        if dx:
            for a in range(1, 4):
                for c in range(4):
                    for d in range(4):
                        df += (self._alpha[a + 4*c + 16*d]
                               *a*((x - self._Xi)/(self._Xiplus1 - self._Xi))**(a - 1)/(self._Xiplus1 - self._Xi)
                               *((y - self._Yj)/(self._Yjplus1 - self._Yj))**c
                               *((z - self._Zk)/(self._Zkplus1 - self._Zk))**d)
        elif dy:
            for a in range(4):
                for c in range(1, 4):
                    for d in range(4):
                        df += (self._alpha[a + 4*c + 16*d]
                               *((x - self._Xi)/(self._Xiplus1 - self._Xi))**a
                               *c*((y - self._Yj)/(self._Yjplus1 - self._Yj))**(c - 1)/(self._Yjplus1 - self._Yj)
                               *((z - self._Zk)/(self._Zkplus1 - self._Zk))**d)
        elif dz:
            for a in range(4):
                for c in range(4):
                    for d in range(1, 4):
                        df += (self._alpha[a + 4*c + 16*d]
                               *((x - self._Xi)/(self._Xiplus1 - self._Xi))**a
                               *((y - self._Yj)/(self._Yjplus1 - self._Yj))**c
                               *d*((z - self._Zk)/(self._Zkplus1 - self._Zk))**(d - 1)/(self._Zkplus1 - self._Zk))
        return df

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # first example: simple one-dimensional cubic function
    f = lambda x: 2*x**3 - 3*x**2 - 3*x + 2

    X = Y = Z = np.linspace(-3, 3, 11)
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    F = f(x)

    interp1 = tricubic(X, Y, Z, F)

    Xnew = np.linspace(-3, 3, 101)
    Fnew = np.zeros((Xnew.size))
    for i, X0 in enumerate(Xnew):
        Fnew[i] = interp1(X0, 0, 0)

    fig, ax = plt.subplots()
    ax.plot(X, F[:, 5, 5], 'o-')
    ax.plot(Xnew, Fnew)

    # second example: two-dimensional Ricker wavelet
    psi = lambda x, y: (1 - (x**2 + y**2))*np.exp(-(x**2 + y**2))

    X = Y = Z = np.linspace(-5, 5, 21)
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    Psi = psi(x, y)

    interp2 = tricubic(X, Y, Z, Psi)

    Xnew = Ynew = np.linspace(-5, 5, 101)
    Psinew = np.zeros((Xnew.size, Ynew.size))
    for i, X0 in enumerate(Xnew):
        for j, Y0 in enumerate(Ynew):
            Psinew[i, j] = interp2(X0, Y0, 0)

    fig, ax = plt.subplots()
    ax.plot(Y, Psi[10, :, 10], 'o-')
    ax.plot(Ynew, Psinew[50, :])

    plt.show()
