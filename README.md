# tricubic
Python implementation of a tricubic interpolator in three dimensions. The scheme is based on [Lekien and Marsden (2005), "Tricubic interpolation in three dimensions," Int. J. Numer. Meth. Eng. 63, 455](https://doi.org/10.1002/nme.1296).

## Usage
`tricubic` requires the `numpy` package, so make sure that this is installed on your system.

Here is a simple example to get you started. The interpolator is an object that can be imported as
```
from tricubic import Tricubic
```
We will consider the following function:
$$f(x, y, z) = - x^3 + x + y^2 - z.$$
The `Tricubic` object accepts four inputs `(X, Y, Z, F)`, which are the samples of the three independent variables $(x, y, z)$ and the one dependent variable $f$. These can be generated for our function as
```
import numpy as np

def f(x, y, z): return - x**3 + x + y**2 - z

X, Y, Z = np.linspace(-1, 1, 21)
x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
F = f(x, y, z)
```
Then the interpolator object is initialised as
```
interp = Tricubic(X, Y, Z, F)
```
The interpolator can be called at a point, say $(0.5, -0.1, 0.3)$, for an estimate of the function
```
interp(0.5, -0.1, 0.3)
```
and its derivatives
```
interp(0.5, -0.1, 0.3, dx=True)
interp(0.5, -0.1, 0.3, dy=True)
interp(0.5, -0.1, 0.3, dz=True)
```
Due to the local nature of the interpolation scheme, it does not accept arrays as inputs.

## Testing
To test, run
```
python -m unittest tests.test
```
in the root directory.
