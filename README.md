# pyregpt - Python wrapper for Regularized Perturbation Theory

## Introduction

**pyregpt** calculates power spectra (density - density, density - velocity, velocity - velocity) and RSD correction terms A and B at 1 and 2 loops.
Bias terms (https://arxiv.org/abs/0902.0991) are computed at 1 loop.
This package is essentially based on [RegPT and RegPTcorr](http://www2.yukawa.kyoto-u.ac.jp/~atsushi.taruya/regpt_code.html) (https://arxiv.org/abs/1208.1191).
The original Fortran code has been refurbished into a more flexible C code, including OpenMP for parallelization.
A basic Python wrapper (based on ctypes) is provided in pyregpt.py.
The integration range and precision are not hardcoded anymore and can be set through set_precision.
All tests are in pyregpt/tests/.
Parts of this code make use of the [Cuba library](http://www.feynarts.de/cuba/).

## Requirements

- numpy
- scipy

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/pyregpt
```

### git

First:
```
git clone https://github.com/adematti/pyregpt.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
python setup.py develop --user
```

## Mac OS

If you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export::

```
export CC=clang
```

Before installing **pyregpt**.

## License

**pyregpt** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/cosmodesi/pyregpt/blob/main/LICENSE).
