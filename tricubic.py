__author__  = 'Fabian Gittins'
__date__    = '26/09/2023'

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
        if not np.all(np.diff(X) > 0) and not np.all(np.diff(X) < 0):
            raise ValueError('X must be either monotonically increasing or '
                             'decreasing')
        if not np.all(np.diff(Y) > 0) and not np.all(np.diff(Y) < 0):
            raise ValueError('Y must be either monotonically increasing or '
                             'decreasing')
        if not np.all(np.diff(Z) > 0) and not np.all(np.diff(Z) < 0):
            raise ValueError('Z must be either monotonically increasing or '
                             'decreasing')

        self.X = X
        self.Y = Y
        self.Z = Z
        self.F = F

        self.__Xmin, self.__Xmax = X.min(), X.max()
        self.__Ymin, self.__Ymax = Y.min(), Y.max()
        self.__Zmin, self.__Zmax = Z.min(), Z.max()

        self.__dFdX = self.__build_dFdX(F, X)
        self.__dFdY = self.__build_dFdY(F, Y)
        self.__dFdZ = self.__build_dFdZ(F, Z)
        self.__d2FdXdY = self.__build_dFdX(self.__dFdY, X)
        self.__d2FdXdZ = self.__build_dFdX(self.__dFdZ, X)
        self.__d2FdYdZ = self.__build_dFdY(self.__dFdZ, Y)
        self.__d3FdXdYdZ = self.__build_dFdX(self.__d2FdYdZ, X)

        self.__initialised = False
        self.__Xi, self.__Xiplus1 = None, None
        self.__Yj, self.__Yjplus1 = None, None
        self.__Zk, self.__Zkplus1 = None, None
        self.__alpha = None
        self.__Binv = np.array([
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
             -30, 30, -9, 30, 9, -30, 9, -9, -18, 18, 18, -9, -18, 9, 9, -9, 
             -18, -12, -6, 18, -3, 12, 6, 3, -12, -6, 12, -6, 6, -3, 6, 3, 
             -20, 20, -6, -10, 6, 10, -3, 3, -12, -8, -4, -6, -2, -4, -2, -1],
            [18, -18, -18, -18, 18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 
             24, -24, 6, -24, -6, 24, -6, 6, 12, -12, -12, 6, 12, -6, -6, 6, 
             12, 12, 3, -12, 3, -12, -3, -3, 6, 6, -6, 3, -6, 3, -3, -3, 
             16, -16, 4, 8, -4, -8, 2, -2, 8, 8, 2, 4, 2, 4, 1, 1],
            [-6, 0, 6, 6, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             -3, 0, -3, 3, 0, 0, 3, 0, -4, 0, 4, -2, 0, 0, 2, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             -2, 0, -2, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 6, 0, 0, -6, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             -3, 0, -3, 3, 0, 0, 3, 0, -4, 0, 4, -2, 0, 0, 2, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, -1, 0, 0, -1, 0],
            [18, -18, -18, -18, 18, 18, 18, -18, 12, 6, -12, -12, -6, -6, 12, 6, 
             21, -21, 9, -21, -9, 21, -9, 9, 12, -12, -12, 6, 12, -6, -6, 6, 
             12, 9, 6, -12, 3, -9, -6, -3, 8, 4, -8, 4, -4, 2, -4, -2, 
             14, -14, 6, 7, -6, -7, 3, -3, 8, 6, 4, 4, 2, 3, 2, 1],
            [-12, 12, 12, 12, -12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, 
             -18, 18, -6, 18, 6, -18, 6, -6, -8, 8, 8, -4, -8, 4, 4, -4, 
             -9, -9, -3, 9, -3, 9, 3, 3, -4, -4, 4, -2, 4, -2, 2, 2, 
             -12, 12, -4, -6, 4, 6, -2, 2, -6, -6, -2, -3, -2, -3, -1, -1],
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
             24, -24, 6, -24, -6, 24, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 
             14, 10, 4, -14, 2, -10, -4, -2, 6, 3, -6, 6, -3, 3, -6, -3, 
             14, -14, 3, 10, -3, -10, 3, -3, 8, 6, 2, 6, 1, 4, 2, 1],
            [-12, 12, 12, 12, -12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, 
             -20, 20, -4, 20, 4, -20, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, 
             -10, -10, -2, 10, -2, 10, 2, 2, -3, -3, 3, -3, 3, -3, 3, 3, 
             -12, 12, -2, -8, 2, 8, -2, 2, -6, -6, -1, -4, -1, -4, -1, -1],
            [4, 0, -4, -4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             2, 0, 2, -2, 0, 0, -2, 0, 2, 0, -2, 2, 0, 0, -2, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, -4, 0, 0, 4, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             2, 0, 2, -2, 0, 0, -2, 0, 2, 0, -2, 2, 0, 0, -2, 0, 
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [-12, 12, 12, 12, -12, -12, -12, 12, -8, -4, 8, 8, 4, 4, -8, -4, 
             -18, 18, -6, 18, 6, -18, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, 
             -10, -8, -4, 10, -2, 8, 4, 2, -4, -2, 4, -4, 2, -2, 4, 2, 
             -11, 11, -3, -7, 3, 7, -3, 3, -6, -5, -2, -4, -1, -3, -2, -1],
            [8, -8, -8, -8, 8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 
             16, -16, 4, -16, -4, 16, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 
             8, 8, 2, -8, 2, -8, -2, -2, 2, 2, -2, 2, -2, 2, -2, -2, 
             10, -10, 2, 6, -2, -6, 2, -2, 5, 5, 1, 3, 1, 3, 1, 1]
        ])

    @staticmethod
    def __build_dFdX(F, X):
        """
        Five-point finite difference formula for 
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
        dFdX[0, :, :] = (-25*F[0, :, :] + 48*F[1, :, :] - 36*F[2, :, :] + 16*F[3, :, :] - 3*F[4, :, :])/(-25*X[0] + 48*X[1] - 36*X[2] + 16*X[3] - 3*X[4])
        dFdX[1, :, :] = (-3*F[0, :, :] - 10*F[1, :, :] + 18*F[2, :, :] - 6*F[3, :, :] + F[4, :, :])/(-3*X[0] - 10*X[1] + 18*X[2] - 6*X[3] + X[4])
        for i in range(2, len(X) - 2):
            dFdX[i, :, :] = (F[i - 2, :, :] - 8*F[i - 1, :, :] + 8*F[i + 1, :, :] - F[i + 2, :, :])/(X[i - 2] - 8*X[i - 1] + 8*X[i + 1] - X[i + 2])
        dFdX[-2, :, :] = (3*F[-5, :, :] + 10*F[-4, :, :] - 18*F[-3, :, :] + 6*F[-2, :, :] - F[-1, :, :])/(3*X[-5] + 10*X[-4] - 18*X[-3] + 6*X[-2] - X[-1])
        dFdX[-1, :, :] = (25*F[-5, :, :] - 48*F[-4, :, :] + 36*F[-3, :, :] - 16*F[-2, :, :] + 3*F[-1, :, :])/(25*X[-5] - 48*X[-4] + 36*X[-3] - 16*X[-2] + 3*X[-1])
        return dFdX

    @staticmethod
    def __build_dFdY(F, Y):
        """
        Five-point finite difference formula for 
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
        dFdY[:, 0, :] = (-25*F[:, 0, :] + 48*F[:, 1, :] - 36*F[:, 2, :] + 16*F[:, 3, :] - 3*F[:, 4, :])/(-25*Y[0] + 48*Y[1] - 36*Y[2] + 16*Y[3] - 3*Y[4])
        dFdY[:, 1, :] = (-3*F[:, 0, :] - 10*F[:, 1, :] + 18*F[:, 2, :] - 6*F[:, 3, :] + F[:, 4, :])/(-3*Y[0] - 10*Y[1] + 18*Y[2] - 6*Y[3] + Y[4])
        for j in range(2, len(Y) - 2):
            dFdY[:, j, :] = (F[:, j - 2, :] - 8*F[:, j - 1, :] + 8*F[:, j + 1, :] - F[:, j + 2, :])/(Y[j - 2] - 8*Y[j - 1] + 8*Y[j + 1] - Y[j + 2])
        dFdY[:, -2, :] = (3*F[:, -5, :] + 10*F[:, -4, :] - 18*F[:, -3, :] + 6*F[:, -2, :] - F[:, -1, :])/(3*Y[-5] + 10*Y[-4] - 18*Y[-3] + 6*Y[-2] - Y[-1])
        dFdY[:, -1, :] = (25*F[:, -5, :] - 48*F[:, -4, :] + 36*F[:, -3, :] - 16*F[:, -2, :] + 3*F[:, -1, :])/(25*Y[-5] - 48*Y[-4] + 36*Y[-3] - 16*Y[-2] + 3*Y[-1])
        return dFdY

    @staticmethod
    def __build_dFdZ(F, Z):
        """
        Five-point finite difference formula for 
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
        dFdZ[:, :, 0] = (-25*F[:, :, 0] + 48*F[:, :, 1] - 36*F[:, :, 2] + 16*F[:, :, 3] - 3*F[:, :, 4])/(-25*Z[0] + 48*Z[1] - 36*Z[2] + 16*Z[3] - 3*Z[4])
        dFdZ[:, :, 1] = (-3*F[:, :, 0] - 10*F[:, :, 1] + 18*F[:, :, 2] - 6*F[:, :, 3] + F[:, :, 4])/(-3*Z[0] - 10*Z[1] + 18*Z[2] - 6*Z[3] + Z[4])
        for k in range(2, len(Z) - 2):
            dFdZ[:, :, k] = (F[:, :, k - 2] - 8*F[:, :, k - 1] + 8*F[:, :, k + 1] - F[:, :, k + 2])/(Z[k - 2] - 8*Z[k - 1] + 8*Z[k + 1] - Z[k + 2])
        dFdZ[:, :, -2] = (3*F[:, :, -5] + 10*F[:, :, -4] - 18*F[:, :, -3] + 6*F[:, :, -2] - F[:, :, -1])/(3*Z[-5] + 10*Z[-4] - 18*Z[-3] + 6*Z[-2] - Z[-1])
        dFdZ[:, :, -1] = (25*F[:, :, -5] - 48*F[:, :, -4] + 36*F[:, :, -3] - 16*F[:, :, -2] + 3*F[:, :, -1])/(25*Z[-5] - 48*Z[-4] + 36*Z[-3] - 16*Z[-2] + 3*Z[-1])
        return dFdZ

    def __calculate_coefficients(self, i0, j0, k0):
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
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*self.__dFdX[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0 + 1, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0 + 1, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0 + 1, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0, j0 + 1, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*self.__dFdY[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0, j0, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0 + 1, j0, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0, j0 + 1, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0, j0, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0 + 1, j0 + 1, k0],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0 + 1, j0, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0, j0 + 1, k0 + 1],
            (self.Z[k0 + 1] - self.Z[k0])*self.__dFdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*self.__d2FdXdY[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdXdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0 + 1, j0, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0 + 1, j0 + 1, k0],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0 + 1, j0, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0, j0 + 1, k0 + 1],
            (self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d2FdYdZ[i0 + 1, j0 + 1, k0 + 1],
            #####
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0 + 1, j0, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0 + 1, j0 + 1, k0],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0 + 1, j0, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0, j0 + 1, k0 + 1],
            (self.X[i0 + 1] - self.X[i0])*(self.Y[j0 + 1] - self.Y[j0])*(self.Z[k0 + 1] - self.Z[k0])*self.__d3FdXdYdZ[i0 + 1, j0 + 1, k0 + 1]
        ])

        # flag for first interpolation
        self.__initialised = True

        # record location of cube
        self.__Xi, self.__Xiplus1 = self.X[i0], self.X[i0 + 1]
        self.__Yj, self.__Yjplus1 = self.Y[j0], self.Y[j0 + 1]
        self.__Zk, self.__Zkplus1 = self.Z[k0], self.Z[k0 + 1]

        return self.__Binv @ b
    
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
        if x < self.__Xmin or self.__Xmax < x:
            raise ValueError('x must lie within grid defined by X')
        if y < self.__Ymin or self.__Ymax < y:
            raise ValueError('y must lie within grid defined by Y')
        if z < self.__Zmin or self.__Zmax < z:
            raise ValueError('z must lie within grid defined by Z')
        if dx or dy or dz:
            return self.partial_derivative(x, y, z, dx, dy, dz)

        # check if coefficients from last interpolation can be re-used
        if (not self.__initialised or not (self.__Xi <= x < self.__Xiplus1 
                                          and self.__Yj <= y < self.__Yjplus1 
                                          and self.__Zk <= z < self.__Zkplus1)):
            # find new origin of cube
            i0 = np.where(self.X <= x)[0][-1]
            j0 = np.where(self.Y <= y)[0][-1]
            k0 = np.where(self.Z <= z)[0][-1]

            # cheap and cheerful fix for evaluations at final grid points
            if x == self.__Xmax:
                i0 -= 1
            if y == self.__Ymax:
                j0 -= 1
            if z == self.__Zmax:
                k0 -= 1

            self.__alpha = self.__calculate_coefficients(i0, j0, k0)

        # evaluate tricubic function
        xi = (x - self.__Xi)/(self.__Xiplus1 - self.__Xi)
        eta = (y - self.__Yj)/(self.__Yjplus1 - self.__Yj)
        zeta = (z - self.__Zk)/(self.__Zkplus1 - self.__Zk)

        etaarray = (1, eta, eta**2, eta**3)
        zetaarray = (1, zeta, zeta**2, zeta**3)

        f = 0
        for c in range(4):
            for d in range(4):
                f += ((self.__alpha[4*c + 16*d] 
                       + xi*(self.__alpha[1 + 4*c + 16*d] 
                             + xi*(self.__alpha[2 + 4*c + 16*d] 
                                   + xi*self.__alpha[3 + 4*c + 16*d])))
                      *etaarray[c]*zetaarray[d])
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
        if x < self.__Xmin or self.__Xmax < x:
            raise ValueError('x must lie within grid defined by X')
        if y < self.__Ymin or self.__Ymax < y:
            raise ValueError('y must lie within grid defined by Y')
        if z < self.__Zmin or self.__Zmax < z:
            raise ValueError('z must lie within grid defined by Z')
        if dx + dy + dz > 1:
            raise ValueError('Only one of `dx`, `dy`, `dz` can be True')

        # check if coefficients from last interpolation can be re-used
        if (not self.__initialised or not (self.__Xi <= x < self.__Xiplus1 
                                          and self.__Yj <= y < self.__Yjplus1 
                                          and self.__Zk <= z < self.__Zkplus1)):
            # find new origin of cube
            i0 = np.where(self.X <= x)[0][-1]
            j0 = np.where(self.Y <= y)[0][-1]
            k0 = np.where(self.Z <= z)[0][-1]

            # cheap and cheerful fix for evaluations at final grid points
            if x == self.__Xmax:
                i0 -= 1
            if y == self.__Ymax:
                j0 -= 1
            if z == self.__Zmax:
                k0 -= 1

            self.__alpha = self.__calculate_coefficients(i0, j0, k0)

        # evaluate derivative of tricubic function
        xi = (x - self.__Xi)/(self.__Xiplus1 - self.__Xi)
        eta = (y - self.__Yj)/(self.__Yjplus1 - self.__Yj)
        zeta = (z - self.__Zk)/(self.__Zkplus1 - self.__Zk)

        xiarray = (1, xi, xi**2)
        etaarray = (1, eta, eta**2, eta**3)
        zetaarray = (1, zeta, zeta**2, zeta**3)

        df = 0
        if dx:
            for a in range(1, 4):
                for d in range(4):
                    df += ((self.__alpha[a + 16*d] 
                            + eta*(self.__alpha[a + 4 + 16*d] 
                                   + eta*(self.__alpha[a + 8 + 16*d] 
                                          + eta*self.__alpha[a + 12 + 16*d])))
                           *a*xiarray[a - 1]/(self.__Xiplus1 - self.__Xi)
                           *zetaarray[d])
        elif dy:
            for c in range(1, 4):
                for d in range(4):
                    df += ((self.__alpha[4*c + 16*d] 
                            + xi*(self.__alpha[1 + 4*c + 16*d] 
                                  + xi*(self.__alpha[2 + 4*c + 16*d] 
                                        + xi*self.__alpha[3 + 4*c + 16*d])))
                           *c*etaarray[c - 1]/(self.__Yjplus1 - self.__Yj)
                           *zetaarray[d])
        elif dz:
            for c in range(4):
                for d in range(1, 4):
                    df += ((self.__alpha[4*c + 16*d] 
                            + xi*(self.__alpha[1 + 4*c + 16*d] 
                                  + xi*(self.__alpha[2 + 4*c + 16*d] 
                                        + xi*self.__alpha[3 + 4*c + 16*d])))
                           *etaarray[c]
                           *d*zetaarray[d - 1]/(self.__Zkplus1 - self.__Zk))
        return df
