pyregpt - Python wrapper for Regularized Perturbation Theory
============================================================

Introduction
------------

**pyregpt** calculates power spectra (density - density, density - velocity, velocity - velocity) and RSD correction terms A and B at 1 and 2 loops.
Bias terms (https://arxiv.org/abs/0902.0991) are computed at 1 loop.
This package is essentially based on RegPT and RegPTcorr http://www2.yukawa.kyoto-u.ac.jp/~atsushi.taruya/regpt_code.html (https://arxiv.org/abs/1208.1191).
The original Fortran code has been refurbished into a more flexible C code, including OpenMP for parallelization if available.
A basic Python wrapper (based on ctypes) is provided in pyregpt.py.
The integration range and precision are not hardcoded anymore can be set through set_precision.
All tests are in pyregpt/tests/.


Installation
------------

First download Cuba-4.1 (http://www.feynarts.de/cuba/) and in Cuba-4.1 run:\
$ ./configure\
Then make shared library (credits to http://github.com/JohannesBuchner/cuba/) following:\
$ sed 's/CFLAGS = -O3 -fomit-frame-pointer/CFLAGS = -O3 -fPIC -fomit-frame-pointer/g' --in-place makefile\
$ make -B libcuba.a\
$ gcc -shared -Wall $(ar xv libcuba.a | sed 's/x - //g') -lm -o libcuba.so\
Export CUBA path:\
$ export CUBA=$(pwd)\
\
To compile the C code, in pyregpt run:\
$ make clean\
$ make

Requirements
------------

- scipy
- numpy
