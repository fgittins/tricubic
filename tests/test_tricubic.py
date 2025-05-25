"""Test suite for `Tricubic` interpolator."""

from unittest import TestCase

import numpy as np

from tricubic import Tricubic


class Test(TestCase):
    rng = np.random.default_rng(15092023)

    def test_constant(self):
        n = 5
        X = Y = Z = np.linspace(0, 1, n)
        val = self.rng.random()
        F = np.full((n, n, n), val)

        f = Tricubic(X, Y, Z, F)

        Xtest = Ytest = Ztest = np.linspace(0, 1, 5 * n)
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), val)
                    self.assertAlmostEqual(f(x, y, z, dx=True), 0)
                    self.assertAlmostEqual(f(x, y, z, dy=True), 0)
                    self.assertAlmostEqual(f(x, y, z, dz=True), 0)

    def test_cube_corners(self):
        n = 11
        X = Y = Z = np.linspace(0, 1, n)
        F = self.rng.random((n, n, n))

        f = Tricubic(X, Y, Z, F)

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, z in enumerate(Z):
                    self.assertAlmostEqual(f(x, y, z), F[i, j, k])

    def test_linear(self):
        def ftest(x, y, z):
            return 1 + 2 * x + 3 * y - 4 * z

        n = 11
        X = Y = Z = np.linspace(-5, 5, n)
        F = [[[ftest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        # focus on patch in grid
        Xtest = Ytest = Ztest = np.linspace(0, 1, 3 * n)
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), ftest(x, y, z))
                    self.assertAlmostEqual(f(x, y, z, dx=True), 2)
                    self.assertAlmostEqual(f(x, y, z, dy=True), 3)
                    self.assertAlmostEqual(f(x, y, z, dz=True), -4)

    def test_cubic(self):
        def ftest(x, y, z):
            return -10 + 0.1 * x - 2 * x**2 + x**3 - 5 * y**2

        n = 11
        X = Y = Z = np.linspace(-5, 5, n)
        F = [[[ftest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        # focus on patch in grid
        Xtest = Ytest = Ztest = np.linspace(-2, -0.5, 3 * n)
        for x in Xtest:
            for y in Ytest:
                for z in Ztest:
                    self.assertAlmostEqual(f(x, y, z), ftest(x, y, z))
                    self.assertAlmostEqual(
                        f(x, y, z, dx=True), 0.1 - 4 * x + 3 * x**2
                    )
                    self.assertAlmostEqual(f(x, y, z, dy=True), -10 * y)
                    self.assertAlmostEqual(f(x, y, z, dz=True), 0)

    def test_ricker_wavelet_like(self):
        def ftest(x, y, z):
            return x * np.exp(-(x**2) - y**2 - z**2)

        n = 51
        X = Y = Z = np.linspace(-2, 2, n)
        F = [[[ftest(x, y, z) for z in Z] for y in Y] for x in X]

        f = Tricubic(X, Y, Z, F)

        x, y, z = self.rng.random(
            3,
        )
        self.assertAlmostEqual(f(x, y, z), ftest(x, y, z), places=5)
        self.assertAlmostEqual(
            f(x, y, z, dx=True),
            (1 - 2 * x**2) * np.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dy=True),
            -2 * x * y * np.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
        self.assertAlmostEqual(
            f(x, y, z, dz=True),
            -2 * x * z * np.exp(-(x**2) - y**2 - z**2),
            places=3,
        )
