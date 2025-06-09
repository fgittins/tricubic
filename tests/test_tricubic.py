from unittest import TestCase

import numpy

from tricubic import Tricubic


def ftest(x: float, y: float, z: float) -> float:
    return 1 + 2 * x + 3 * y - 4 * z


def gtest(x: float, y: float) -> float:
    return -10 + 0.1 * x - 2 * x**2 + x**3 - 5 * y**2


def htest(x: float, y: float, z: float) -> float:
    return x * numpy.exp(-(x**2) - y**2 - z**2)


class TestTricubic(TestCase):
    rng = numpy.random.default_rng(15092023)

    def test_constant(self) -> None:
        n = 5
        X = Y = Z = numpy.linspace(0, 1, n, dtype=numpy.float64)
        val = self.rng.random()
        F = numpy.full((n, n, n), val)

        f = Tricubic(X, Y, Z, F)

        Xtest = Ytest = Ztest = numpy.linspace(
            0, 1, 5 * n, dtype=numpy.float64
        )
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), val)
                    self.assertAlmostEqual(f(x, y, z, dx=1), 0)
                    self.assertAlmostEqual(f(x, y, z, dy=1), 0)
                    self.assertAlmostEqual(f(x, y, z, dz=1), 0)

    def test_cube_corners(self) -> None:
        n = 11
        X = Y = Z = numpy.linspace(0, 1, n, dtype=numpy.float64)
        F = self.rng.random((n, n, n))

        f = Tricubic(X, Y, Z, F)

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, z in enumerate(Z):
                    self.assertAlmostEqual(f(x, y, z), F[i, j, k])

    def test_linear(self) -> None:
        n = 11
        X = Y = Z = numpy.linspace(-5, 5, n, dtype=numpy.float64)
        F = [[[ftest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        # focus on patch in grid
        Xtest = Ytest = Ztest = numpy.linspace(
            0, 1, 3 * n, dtype=numpy.float64
        )
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), ftest(x, y, z))
                    self.assertAlmostEqual(f(x, y, z, dx=1), 2)
                    self.assertAlmostEqual(f(x, y, z, dy=1), 3)
                    self.assertAlmostEqual(f(x, y, z, dz=1), -4)

    def test_cubic(self) -> None:
        n = 11
        X = Y = Z = numpy.linspace(-5, 5, n, dtype=numpy.float64)
        F = [[[gtest(x, y) for _ in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        # focus on patch in grid
        Xtest = Ytest = Ztest = numpy.linspace(
            -2, -0.5, 3 * n, dtype=numpy.float64
        )
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), gtest(x, y))
                    self.assertAlmostEqual(
                        f(x, y, z, dx=1), 0.1 - 4 * x + 3 * x**2
                    )
                    self.assertAlmostEqual(f(x, y, z, dy=1), -10 * y)
                    self.assertAlmostEqual(f(x, y, z, dz=1), 0)

    def test_ricker_wavelet_like(self) -> None:
        n = 51
        X = Y = Z = numpy.linspace(-2, 2, n, dtype=numpy.float64)
        F = [[[htest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        x: float
        y: float
        z: float
        x, y, z = self.rng.random(3)
        self.assertAlmostEqual(f(x, y, z), htest(x, y, z), places=5)
        self.assertAlmostEqual(
            f(x, y, z, dx=1),
            (1 - 2 * x**2) * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dy=1),
            -2 * x * y * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dz=1),
            -2 * x * z * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )

    def test_errors(self) -> None:
        n = 5
        X = Y = Z = numpy.linspace(0, 1, n, dtype=numpy.float64)
        val = self.rng.random()
        F = numpy.full((n, n, n), val)

        X_reshape = X.reshape(5, 1)
        with self.assertRaises(ValueError):
            Tricubic(X_reshape, Y, Z, F)

        Y_reshape = Y.reshape(5, 1)
        with self.assertRaises(ValueError):
            Tricubic(X, Y_reshape, Z, F)

        Z_reshape = Z.reshape(5, 1)
        with self.assertRaises(ValueError):
            Tricubic(X, Y, Z_reshape, F)

        with self.assertRaises(ValueError):
            Tricubic(X, Y, Z, F[0])

        with self.assertRaises(ValueError):
            Tricubic(X[:-1], Y, Z, F)

        with self.assertRaises(ValueError):
            Tricubic(X, Y[:-1], Z, F)

        with self.assertRaises(ValueError):
            Tricubic(X, Y, Z[:-1], F)

        X_new = X.copy()
        X_new[0], X_new[1] = X_new[1], X_new[0]
        with self.assertRaises(ValueError):
            Tricubic(X_new, Y, Z, F)

        Y_new = Y.copy()
        Y_new[0], Y_new[1] = Y_new[1], Y_new[0]
        with self.assertRaises(ValueError):
            Tricubic(X, Y_new, Z, F)

        Z_new = Z.copy()
        Z_new[0], Z_new[1] = Z_new[1], Z_new[0]
        with self.assertRaises(ValueError):
            Tricubic(X, Y, Z_new, F)

        f = Tricubic(X, Y, Z, F)
        x: float
        y: float
        z: float
        x, y, z = self.rng.random(3)

        with self.assertRaises(ValueError):
            f(2, y, z)

        with self.assertRaises(ValueError):
            f(x, 2, z)

        with self.assertRaises(ValueError):
            f(x, y, 2)

        with self.assertRaises(NotImplementedError):
            f(x, y, z, dx=2)

        with self.assertRaises(ValueError):
            f(x, y, z, dx=1, dy=1)
