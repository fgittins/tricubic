"""
Class definition for tricubic interpolator in three dimensions.

Module includes:
- `Tricubic` : Class for tricubic interpolator.
"""

from importlib.resources import files
from typing import Annotated

import numpy
from numpy.typing import ArrayLike, NDArray


class Tricubic:
    """
    A tricubic interpolator in three dimensions.

    Based on method described in Ref. [1].

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

    References
    ----------
    [1] Lekien and Marsden (2005),
    "Tricubic interpolation in three dimensions,"
    Int. J. Numer. Meth. Eng. 63, 455.
    """

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        F: ArrayLike,
    ) -> None:
        X_array = numpy.asarray(X, dtype=numpy.float64)
        Y_array = numpy.asarray(Y, dtype=numpy.float64)
        Z_array = numpy.asarray(Z, dtype=numpy.float64)
        F_array = numpy.asarray(F, dtype=numpy.float64)
        if X_array.ndim != 1:
            msg = "`X` must be one dimensional"
            raise ValueError(msg)
        if Y_array.ndim != 1:
            msg = "`Y` must be one dimensional"
            raise ValueError(msg)
        if Z_array.ndim != 1:
            msg = "`Z` must be one dimensional"
            raise ValueError(msg)
        if F_array.ndim != 3:
            msg = "`F` must be three dimensional"
            raise ValueError(msg)
        if X_array.size != F_array.shape[0]:
            msg = (
                "First dimension of `F` must have same number of elements as "
                "`X`"
            )
            raise ValueError(msg)
        if Y_array.size != F_array.shape[1]:
            msg = (
                "Second dimension of `F` must have same number of elements as "
                "`Y`"
            )
            raise ValueError(msg)
        if Z_array.size != F_array.shape[2]:
            msg = (
                "Third dimension of `F` must have same number of elements as "
                "`Z`"
            )
            raise ValueError(msg)
        dX = numpy.diff(X)
        if not (numpy.all(dX > 0) or numpy.all(dX < 0)):
            msg = "`X` must be either monotonically increasing or decreasing"
            raise ValueError(msg)
        dY = numpy.diff(Y)
        if not (numpy.all(dY > 0) or numpy.all(dY < 0)):
            msg = "`Y` must be either monotonically increasing or decreasing"
            raise ValueError(msg)
        dZ = numpy.diff(Z)
        if not (numpy.all(dZ > 0) or numpy.all(dZ < 0)):
            msg = "`Z` must be either monotonically increasing or decreasing"
            raise ValueError(msg)

        self.X = X_array
        self.Y = Y_array
        self.Z = Z_array
        self.F = F_array

        self.__Xmin: float = X_array.min()
        self.__Xmax: float = X_array.max()
        self.__Ymin: float = Y_array.min()
        self.__Ymax: float = Y_array.max()
        self.__Zmin: float = Z_array.min()
        self.__Zmax: float = Z_array.max()

        self.__dFdX = self.__build_dFdX(F_array, X_array)
        self.__dFdY = self.__build_dFdY(F_array, Y_array)
        self.__dFdZ = self.__build_dFdZ(F_array, Z_array)
        self.__d2FdXdY = self.__build_dFdX(self.__dFdY, X_array)
        self.__d2FdXdZ = self.__build_dFdX(self.__dFdZ, X_array)
        self.__d2FdYdZ = self.__build_dFdY(self.__dFdZ, Y_array)
        self.__d3FdXdYdZ = self.__build_dFdX(self.__d2FdYdZ, X_array)

        self.__initialised = False

        with files("tricubic").joinpath("binv.npy").open("rb") as file:
            self.__Binv: Annotated[NDArray[numpy.int64], (64, 64)] = (
                numpy.load(file)
            )

    @staticmethod
    def __build_dFdX(
        F: NDArray[numpy.float64],
        X: NDArray[numpy.float64],
    ) -> NDArray[numpy.float64]:
        """
        Finite-difference formula.

        Five-point finite-difference formula for partial derivative with
        respect to `x`.

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
        dFdX = numpy.zeros_like(F)
        dFdX[0, :, :] = (
            -25 * F[0, :, :]
            + 48 * F[1, :, :]
            - 36 * F[2, :, :]
            + 16 * F[3, :, :]
            - 3 * F[4, :, :]
        ) / (-25 * X[0] + 48 * X[1] - 36 * X[2] + 16 * X[3] - 3 * X[4])
        dFdX[1, :, :] = (
            -3 * F[0, :, :]
            - 10 * F[1, :, :]
            + 18 * F[2, :, :]
            - 6 * F[3, :, :]
            + F[4, :, :]
        ) / (-3 * X[0] - 10 * X[1] + 18 * X[2] - 6 * X[3] + X[4])
        for i in range(2, len(X) - 2):
            dFdX[i, :, :] = (
                F[i - 2, :, :]
                - 8 * F[i - 1, :, :]
                + 8 * F[i + 1, :, :]
                - F[i + 2, :, :]
            ) / (X[i - 2] - 8 * X[i - 1] + 8 * X[i + 1] - X[i + 2])
        dFdX[-2, :, :] = (
            3 * F[-5, :, :]
            + 10 * F[-4, :, :]
            - 18 * F[-3, :, :]
            + 6 * F[-2, :, :]
            - F[-1, :, :]
        ) / (3 * X[-5] + 10 * X[-4] - 18 * X[-3] + 6 * X[-2] - X[-1])
        dFdX[-1, :, :] = (
            25 * F[-5, :, :]
            - 48 * F[-4, :, :]
            + 36 * F[-3, :, :]
            - 16 * F[-2, :, :]
            + 3 * F[-1, :, :]
        ) / (25 * X[-5] - 48 * X[-4] + 36 * X[-3] - 16 * X[-2] + 3 * X[-1])
        return dFdX

    @staticmethod
    def __build_dFdY(
        F: NDArray[numpy.float64],
        Y: NDArray[numpy.float64],
    ) -> NDArray[numpy.float64]:
        """
        Finite-difference formula.

        Five-point finite-difference formula for partial derivative with
        respect to `y`.

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
        dFdY = numpy.zeros_like(F)
        dFdY[:, 0, :] = (
            -25 * F[:, 0, :]
            + 48 * F[:, 1, :]
            - 36 * F[:, 2, :]
            + 16 * F[:, 3, :]
            - 3 * F[:, 4, :]
        ) / (-25 * Y[0] + 48 * Y[1] - 36 * Y[2] + 16 * Y[3] - 3 * Y[4])
        dFdY[:, 1, :] = (
            -3 * F[:, 0, :]
            - 10 * F[:, 1, :]
            + 18 * F[:, 2, :]
            - 6 * F[:, 3, :]
            + F[:, 4, :]
        ) / (-3 * Y[0] - 10 * Y[1] + 18 * Y[2] - 6 * Y[3] + Y[4])
        for j in range(2, len(Y) - 2):
            dFdY[:, j, :] = (
                F[:, j - 2, :]
                - 8 * F[:, j - 1, :]
                + 8 * F[:, j + 1, :]
                - F[:, j + 2, :]
            ) / (Y[j - 2] - 8 * Y[j - 1] + 8 * Y[j + 1] - Y[j + 2])
        dFdY[:, -2, :] = (
            3 * F[:, -5, :]
            + 10 * F[:, -4, :]
            - 18 * F[:, -3, :]
            + 6 * F[:, -2, :]
            - F[:, -1, :]
        ) / (3 * Y[-5] + 10 * Y[-4] - 18 * Y[-3] + 6 * Y[-2] - Y[-1])
        dFdY[:, -1, :] = (
            25 * F[:, -5, :]
            - 48 * F[:, -4, :]
            + 36 * F[:, -3, :]
            - 16 * F[:, -2, :]
            + 3 * F[:, -1, :]
        ) / (25 * Y[-5] - 48 * Y[-4] + 36 * Y[-3] - 16 * Y[-2] + 3 * Y[-1])
        return dFdY

    @staticmethod
    def __build_dFdZ(
        F: NDArray[numpy.float64],
        Z: NDArray[numpy.float64],
    ) -> NDArray[numpy.float64]:
        """
        Finite-difference formula.

        Five-point finite-difference formula for partial derivative with
        respect to `z`.

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
        dFdZ = numpy.zeros_like(F)
        dFdZ[:, :, 0] = (
            -25 * F[:, :, 0]
            + 48 * F[:, :, 1]
            - 36 * F[:, :, 2]
            + 16 * F[:, :, 3]
            - 3 * F[:, :, 4]
        ) / (-25 * Z[0] + 48 * Z[1] - 36 * Z[2] + 16 * Z[3] - 3 * Z[4])
        dFdZ[:, :, 1] = (
            -3 * F[:, :, 0]
            - 10 * F[:, :, 1]
            + 18 * F[:, :, 2]
            - 6 * F[:, :, 3]
            + F[:, :, 4]
        ) / (-3 * Z[0] - 10 * Z[1] + 18 * Z[2] - 6 * Z[3] + Z[4])
        for k in range(2, len(Z) - 2):
            dFdZ[:, :, k] = (
                F[:, :, k - 2]
                - 8 * F[:, :, k - 1]
                + 8 * F[:, :, k + 1]
                - F[:, :, k + 2]
            ) / (Z[k - 2] - 8 * Z[k - 1] + 8 * Z[k + 1] - Z[k + 2])
        dFdZ[:, :, -2] = (
            3 * F[:, :, -5]
            + 10 * F[:, :, -4]
            - 18 * F[:, :, -3]
            + 6 * F[:, :, -2]
            - F[:, :, -1]
        ) / (3 * Z[-5] + 10 * Z[-4] - 18 * Z[-3] + 6 * Z[-2] - Z[-1])
        dFdZ[:, :, -1] = (
            25 * F[:, :, -5]
            - 48 * F[:, :, -4]
            + 36 * F[:, :, -3]
            - 16 * F[:, :, -2]
            + 3 * F[:, :, -1]
        ) / (25 * Z[-5] - 48 * Z[-4] + 36 * Z[-3] - 16 * Z[-2] + 3 * Z[-1])
        return dFdZ

    def __calculate_coefficients(
        self,
        i0: int,
        j0: int,
        k0: int,
    ) -> Annotated[NDArray[numpy.float64], (64,)]:
        r"""
        Calculate vector of coefficients.

        Calculate vector of coefficients `alpha` for interpolation, which
        is obtained from linear equation `alpha = Binv b`.

        Parameters
        ----------
        i0, j0, k0 : int
            Indices of position.

        Notes
        -----
        Vector `b` is constructed out of following set:

            {`f`,
            `\partial f / \partial x`,
            `\partial f / \partial y`,
            `\partial f / \partial z`,
            `\partial^2 f / \partial x \partial y`,
            `\partial^2 f / \partial x \partial z`,
            `\partial^2 f / \partial y \partial z`,
            `\partial^3 f / \partial x \partial y \partial z`}.

        Each of these quantities is evaluated at 8 corners of cube in following
        order: `(i0, j0, k0)`, `(i0 + 1, j0, k0)`, `(i0, j0 + 1, k0)`,
        `(i0, j0, k0 + 1)`, `(i0 + 1, j0 + 1, k0)`, `(i0 + 1, j0, k0 + 1)`,
        `(i0, j0 + 1, k0 + 1)`, `(i0 + 1, j0 + 1, k0 + 1)`. A different order
        would require tweaking matrix `Binv`.
        """
        b = numpy.array(
            [
                ###############################################################
                self.F[i0, j0, k0],
                self.F[i0 + 1, j0, k0],
                self.F[i0, j0 + 1, k0],
                self.F[i0, j0, k0 + 1],
                self.F[i0 + 1, j0 + 1, k0],
                self.F[i0 + 1, j0, k0 + 1],
                self.F[i0, j0 + 1, k0 + 1],
                self.F[i0 + 1, j0 + 1, k0 + 1],
                ###############################################################
                (self.X[i0 + 1] - self.X[i0]) * self.__dFdX[i0, j0, k0],
                (self.X[i0 + 1] - self.X[i0]) * self.__dFdX[i0 + 1, j0, k0],
                (self.X[i0 + 1] - self.X[i0]) * self.__dFdX[i0, j0 + 1, k0],
                (self.X[i0 + 1] - self.X[i0]) * self.__dFdX[i0, j0, k0 + 1],
                (self.X[i0 + 1] - self.X[i0])
                * self.__dFdX[i0 + 1, j0 + 1, k0],
                (self.X[i0 + 1] - self.X[i0])
                * self.__dFdX[i0 + 1, j0, k0 + 1],
                (self.X[i0 + 1] - self.X[i0])
                * self.__dFdX[i0, j0 + 1, k0 + 1],
                (self.X[i0 + 1] - self.X[i0])
                * self.__dFdX[i0 + 1, j0 + 1, k0 + 1],
                ###############################################################
                (self.Y[j0 + 1] - self.Y[j0]) * self.__dFdY[i0, j0, k0],
                (self.Y[j0 + 1] - self.Y[j0]) * self.__dFdY[i0 + 1, j0, k0],
                (self.Y[j0 + 1] - self.Y[j0]) * self.__dFdY[i0, j0 + 1, k0],
                (self.Y[j0 + 1] - self.Y[j0]) * self.__dFdY[i0, j0, k0 + 1],
                (self.Y[j0 + 1] - self.Y[j0])
                * self.__dFdY[i0 + 1, j0 + 1, k0],
                (self.Y[j0 + 1] - self.Y[j0])
                * self.__dFdY[i0 + 1, j0, k0 + 1],
                (self.Y[j0 + 1] - self.Y[j0])
                * self.__dFdY[i0, j0 + 1, k0 + 1],
                (self.Y[j0 + 1] - self.Y[j0])
                * self.__dFdY[i0 + 1, j0 + 1, k0 + 1],
                ###############################################################
                (self.Z[k0 + 1] - self.Z[k0]) * self.__dFdZ[i0, j0, k0],
                (self.Z[k0 + 1] - self.Z[k0]) * self.__dFdZ[i0 + 1, j0, k0],
                (self.Z[k0 + 1] - self.Z[k0]) * self.__dFdZ[i0, j0 + 1, k0],
                (self.Z[k0 + 1] - self.Z[k0]) * self.__dFdZ[i0, j0, k0 + 1],
                (self.Z[k0 + 1] - self.Z[k0])
                * self.__dFdZ[i0 + 1, j0 + 1, k0],
                (self.Z[k0 + 1] - self.Z[k0])
                * self.__dFdZ[i0 + 1, j0, k0 + 1],
                (self.Z[k0 + 1] - self.Z[k0])
                * self.__dFdZ[i0, j0 + 1, k0 + 1],
                (self.Z[k0 + 1] - self.Z[k0])
                * self.__dFdZ[i0 + 1, j0 + 1, k0 + 1],
                ###############################################################
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0 + 1, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0 + 1, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0 + 1, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0, j0 + 1, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * self.__d2FdXdY[i0 + 1, j0 + 1, k0 + 1]
                ),
                ###############################################################
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0 + 1, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0 + 1, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0 + 1, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0, j0 + 1, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdXdZ[i0 + 1, j0 + 1, k0 + 1]
                ),
                ###############################################################
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0, j0, k0]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0 + 1, j0, k0]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0, j0 + 1, k0]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0, j0, k0 + 1]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0 + 1, j0 + 1, k0]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0 + 1, j0, k0 + 1]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0, j0 + 1, k0 + 1]
                ),
                (
                    (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d2FdYdZ[i0 + 1, j0 + 1, k0 + 1]
                ),
                ###############################################################
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0 + 1, j0, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0 + 1, j0 + 1, k0]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0 + 1, j0, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0, j0 + 1, k0 + 1]
                ),
                (
                    (self.X[i0 + 1] - self.X[i0])
                    * (self.Y[j0 + 1] - self.Y[j0])
                    * (self.Z[k0 + 1] - self.Z[k0])
                    * self.__d3FdXdYdZ[i0 + 1, j0 + 1, k0 + 1]
                ),
                ###############################################################
            ],
            dtype=numpy.float64,
        )

        # flag for first interpolation
        self.__initialised = True

        # record location of cube
        self.__Xi: float = self.X[i0]
        self.__Xiplus1: float = self.X[i0 + 1]
        self.__Yj: float = self.Y[j0]
        self.__Yjplus1: float = self.Y[j0 + 1]
        self.__Zk: float = self.Z[k0]
        self.__Zkplus1: float = self.Z[k0 + 1]

        return self.__Binv @ b

    def __check_bounds(self, x: float, y: float, z: float) -> None:
        """Check if `x`, `y`, `z` lie within the bounds of the grid."""
        if not (self.__Xmin <= x <= self.__Xmax):
            msg = "`x` must lie within grid defined by `X`"
            raise ValueError(msg)
        if not (self.__Ymin <= y <= self.__Ymax):
            msg = "`y` must lie within grid defined by `Y`"
            raise ValueError(msg)
        if not (self.__Zmin <= z <= self.__Zmax):
            msg = "`z` must lie within grid defined by `Z`"
            raise ValueError(msg)

    def __ensure_coefficients(self, x: float, y: float, z: float) -> None:
        """Ensure that coefficients are calculated for the cube."""
        if not self.__initialised or not (
            self.__Xi <= x < self.__Xiplus1
            and self.__Yj <= y < self.__Yjplus1
            and self.__Zk <= z < self.__Zkplus1
        ):
            # find new origin of cube
            i0: int = numpy.where(x >= self.X)[0][-1]
            j0: int = numpy.where(y >= self.Y)[0][-1]
            k0: int = numpy.where(z >= self.Z)[0][-1]

            # cheap and cheerful fix for evaluations at final grid points
            if x == self.__Xmax:
                i0 -= 1
            if y == self.__Ymax:
                j0 -= 1
            if z == self.__Zmax:
                k0 -= 1

            self.__alpha = self.__calculate_coefficients(i0, j0, k0)

    def __call__(
        self,
        x: float,
        y: float,
        z: float,
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
    ) -> float:
        """
        Evaluate interpolator or its derivatives at given position.

        Only first derivatives are supported.

        Parameters
        ----------
        x, y, z : float
            Position arguments.
        dx : int, optional
            Order of `x` derivative.
        dy : int, optional
            Order of `y` derivative.
        dz : int, optional
            Order of `z` derivative.

        Returns
        -------
        f : float
            Interpolated value.
        """
        if dx or dy or dz:
            return self.partial_derivative(x, y, z, dx, dy, dz)
        self.__check_bounds(x, y, z)

        # check if coefficients from last interpolation can be re-used
        self.__ensure_coefficients(x, y, z)

        # evaluate tricubic function
        C = self.__alpha.reshape((4, 4, 4), order="F")

        xi = (x - self.__Xi) / (self.__Xiplus1 - self.__Xi)
        eta = (y - self.__Yj) / (self.__Yjplus1 - self.__Yj)
        zeta = (z - self.__Zk) / (self.__Zkplus1 - self.__Zk)

        xiarray = numpy.array([1, xi, xi**2, xi**3], dtype=numpy.float64)
        etaarray = numpy.array([1, eta, eta**2, eta**3], dtype=numpy.float64)
        zetaarray = numpy.array(
            [1, zeta, zeta**2, zeta**3], dtype=numpy.float64
        )

        f: float = numpy.einsum("i,j,k,ijk", xiarray, etaarray, zetaarray, C)
        return f

    def partial_derivative(
        self,
        x: float,
        y: float,
        z: float,
        dx: int,
        dy: int,
        dz: int,
    ) -> float:
        """
        Evaluate derivatives at given position.

        Only first derivatives are supported.

        Parameters
        ----------
        x, y, z : float
            Position arguments.
        dx : int
            Order of `x` derivative.
        dy : int
            Order of `y` derivative.
        dz : int
            Order of `z` derivative.

        Returns
        -------
        df : float
            Interpolated value of derivative.
        """
        self.__check_bounds(x, y, z)
        if dx not in (0, 1) or dy not in (0, 1) or dz not in (0, 1):
            msg = "Only first derivatives are supported"
            raise NotImplementedError(msg)
        if dx + dy + dz != 1:
            msg = "One of `dx`, `dy`, `dz` must be `1`"
            raise ValueError(msg)

        # check if coefficients from last interpolation can be re-used
        self.__ensure_coefficients(x, y, z)

        # evaluate derivative of tricubic function
        C = self.__alpha.reshape((4, 4, 4), order="F")

        xi = (x - self.__Xi) / (self.__Xiplus1 - self.__Xi)
        eta = (y - self.__Yj) / (self.__Yjplus1 - self.__Yj)
        zeta = (z - self.__Zk) / (self.__Zkplus1 - self.__Zk)

        df: float = 0.0
        if dx == 1:
            dxiarray = numpy.array([1, 2 * xi, 3 * xi**2], dtype=numpy.float64)
            etaarray = numpy.array(
                [1, eta, eta**2, eta**3], dtype=numpy.float64
            )
            zetaarray = numpy.array(
                [1, zeta, zeta**2, zeta**3], dtype=numpy.float64
            )

            df = numpy.einsum(
                "i,j,k,ijk", dxiarray, etaarray, zetaarray, C[1:, :, :]
            ) / (self.__Xiplus1 - self.__Xi)
        if dy == 1:
            xiarray = numpy.array([1, xi, xi**2, xi**3], dtype=numpy.float64)
            detaarray = numpy.array(
                [1, 2 * eta, 3 * eta**2], dtype=numpy.float64
            )
            zetaarray = numpy.array(
                [1, zeta, zeta**2, zeta**3], dtype=numpy.float64
            )

            df = numpy.einsum(
                "i,j,k,ijk", xiarray, detaarray, zetaarray, C[:, 1:, :]
            ) / (self.__Yjplus1 - self.__Yj)
        if dz == 1:
            xiarray = numpy.array([1, xi, xi**2, xi**3], dtype=numpy.float64)
            etaarray = numpy.array(
                [1, eta, eta**2, eta**3], dtype=numpy.float64
            )
            dzetaarray = numpy.array(
                [1, 2 * zeta, 3 * zeta**2], dtype=numpy.float64
            )

            df = numpy.einsum(
                "i,j,k,ijk", xiarray, etaarray, dzetaarray, C[:, :, 1:]
            ) / (self.__Zkplus1 - self.__Zk)
        return df
