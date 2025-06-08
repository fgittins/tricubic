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
                    self.assertAlmostEqual(f(x, y, z, dx=True), 0)
                    self.assertAlmostEqual(f(x, y, z, dy=True), 0)
                    self.assertAlmostEqual(f(x, y, z, dz=True), 0)

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
                    self.assertAlmostEqual(f(x, y, z, dx=True), 2)
                    self.assertAlmostEqual(f(x, y, z, dy=True), 3)
                    self.assertAlmostEqual(f(x, y, z, dz=True), -4)

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
                        f(x, y, z, dx=True), 0.1 - 4 * x + 3 * x**2
                    )
                    self.assertAlmostEqual(f(x, y, z, dy=True), -10 * y)
                    self.assertAlmostEqual(f(x, y, z, dz=True), 0)

    def test_ricker_wavelet_like(self) -> None:
        n = 51
        X = Y = Z = numpy.linspace(-2, 2, n, dtype=numpy.float64)
        F = [[[htest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        x: numpy.float64
        y: numpy.float64
        z: numpy.float64
        x, y, z = self.rng.random(3)
        self.assertAlmostEqual(f(x, y, z), htest(x, y, z), places=5)
        self.assertAlmostEqual(
            f(x, y, z, dx=True),
            (1 - 2 * x**2) * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dy=True),
            -2 * x * y * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dz=True),
            -2 * x * z * numpy.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
