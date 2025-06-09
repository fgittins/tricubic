# tricubic

Python implementation of a tricubic interpolator in three dimensions. The scheme is based on [Lekien and Marsden (2005), "Tricubic interpolation in three dimensions," Int. J. Numer. Meth. Eng. 63, 455](https://doi.org/10.1002/nme.1296).

---

## Usage

`tricubic` requires the `numpy` package, so make sure that this is installed on your system.

Here is a simple example to get you started. The interpolator is an object that can be imported as

```python
from tricubic import Tricubic
```

We will consider the following function:

$$
f(x, y, z) = - x^3 + x + y^2 - z.
$$

The `Tricubic` object accepts four inputs `(X, Y, Z, F)`, which are the samples of the three independent variables $(x, y, z)$ and the one dependent variable $f$. These can be generated for our function as

```python
import numpy

def f(x, y, z):
    return - x**3 + x + y**2 - z

X = Y = Z = numpy.linspace(-1, 1, 21)
x, y, z = numpy.meshgrid(X, Y, Z, indexing='ij')
F = f(x, y, z)
```

Then the interpolator object is initialised as

```python
interp = Tricubic(X, Y, Z, F)
```

The interpolator can be called at a point, say $(0.5, -0.1, 0.3)$, for an estimate of the function

```python
interp(0.5, -0.1, 0.3)
```

and its derivatives

```python
interp(0.5, -0.1, 0.3, dx=1)
interp(0.5, -0.1, 0.3, dy=1)
interp(0.5, -0.1, 0.3, dz=1)
```

Due to the local nature of the interpolation scheme, it does not accept arrays as inputs.

## Installation

You can install `tricubic` easily.

### From source (locally)

Clone the repository and install using `pip`:

```
git clone https://github.com/fgittins/tricubic.git
cd tricubic
pip install .
```

### Directly from GitHub

Or it can be installed from the GitHub repository:

```
pip install git+https://github.com/fgittins/tricubic.git
```

## Testing

To test, run

```
python -m unittest tests.test_tricubic
```

in the root directory. Or you can use `pytest`.
