# Python-Fourier-Transform
Discrete and Fast Fourier Transform naive implementation in Python. Naive implemented by not using any package accept `math`.

**Forward Fourier Transform**
```math
{\displaystyle {\widehat {f}}(\xi )=\int _{-\infty }^{\infty }f(x)\ e^{-i2\pi \xi x}\,dx}
```
for all values of $ξ$ produces the frequency-domain function. The integral can diverge at some frequencies. 


**Inverse Fourier Transform**
```math
{\displaystyle f(x)=\int _{-\infty }^{\infty }{\widehat {f}}(\xi )\ e^{i2\pi \xi x}\,d\xi ,\quad \forall \ x\in \mathbb {R} } 
```
is a representation of $f(x)$ as a weighted summation of complex exponential functions.


**The discrete Fourier transform (DFT) is defined by the formula:**
```math
{\displaystyle X_{k}=\sum _{n=0}^{N-1}x_{n}e^{-{\frac {2\pi i}{N}}nk},}
```
where $k{\displaystyle k}$ is an integer ranging from $0$ to $N−1$

for the inverse DFT and further more mathematical proof, see [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) and [Cooley-Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)


## Naive Implementation

```python
import math

def iexp(n):
    return complex(math.cos(n), math.sin(n))

def is_pow2(n):
    return False if n == 0 else (n == 1 or is_pow2(n >> 1))

def dft(xs):
    "naive dft"
    n = len(xs)
    return [sum((xs[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]

def dftinv(xs):
    "naive dft"
    n = len(xs)
    return [sum((xs[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
            for i in range(n)]

def fft_(xs, n, start=0, stride=1):
    "cooley-turkey fft"
    if n == 1: return [xs[start]]
    hn, sd = n // 2, stride * 2
    rs = fft_(xs, hn, start, sd) + fft_(xs, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(-2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

def fft(xs):
    assert is_pow2(len(xs))
    return fft_(xs, len(xs))

def fftinv_(xs, n, start=0, stride=1):
    "cooley-turkey fft"
    if n == 1: return [xs[start]]
    hn, sd = n // 2, stride * 2
    rs = fftinv_(xs, hn, start, sd) + fftinv_(xs, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(2 * math.pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs

def fftinv(xs):
    assert is_pow2(len(xs))
    n = len(xs)
    return [v / n for v in fftinv_(xs, n)]

if __name__ == "__main__":
    wave = [0, 1, 2, 3, 3, 2, 1, 0]
    dfreq = dft(wave)
    ffreq = fft(wave)
    dwave = dftinv(dfreq)
    fwave= fftinv(ffreq)
    print('dfreq')
    print(dfreq)
    print('ffreq')
    print(ffreq)
    print('[v.real for v in dwave]')
    print([v.real for v in dwave])
    print('[v.real for v in fwave]')
    print([v.real for v in fwave])

```
