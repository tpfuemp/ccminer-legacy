Tested hardware
===============

`--benchmark` with the default intensity, release builds, 60 s runs
(`--time-limit=60`). Fast cards throttle as they heat up, so compare runs
only from the same start temperature (GTX 1080 Ti: blakecoin 50 C, others 55 C):

| Algorithm | GPU | OS | Driver | CUDA 8 build | CUDA 10.1 build |
|---|---|---|---|---|---|
| blake | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 123.6 MH/s (PTX JIT) | 126.1 MH/s |
| blake | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 4059 MH/s | 4164 MH/s |
| blake2s | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 230.4 MH/s (PTX JIT) | 230.9 MH/s |
| blake2s | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 6557 MH/s | 7047 MH/s |
| blakecoin | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 227.8 MH/s (PTX JIT) | 227.6 MH/s |
| blakecoin | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 7255 MH/s | 7484 MH/s |
| bmw512 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 30.4 MH/s (PTX JIT) | 30.4 MH/s |
| bmw512 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1451 MH/s | 1492 MH/s |
| decred | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 247.2 MH/s (PTX JIT) | 246.2 MH/s |
| decred | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 7295 MH/s | 8411 MH/s |
| groestl | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.80 MH/s (PTX JIT) | 1.76 MH/s |
| groestl | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 58.5 MH/s | 59.3 MH/s |
| keccak | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 21.3 MH/s (PTX JIT) | 21.8 MH/s |
| keccak | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1176 MH/s | 1205 MH/s |
| myr-gr | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 3.12 MH/s (PTX JIT) | 3.15 MH/s |
| myr-gr | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 114.0 MH/s | 117.1 MH/s |
| sha256d | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 46.7 MH/s (PTX JIT) | 46.5 MH/s |
| sha256d | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1687 MH/s | 1856 MH/s |
| skein | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 22.2 MH/s (PTX JIT) | 22.6 MH/s |
| skein | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 579 MH/s | 653 MH/s |
| tribus | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 2.22 MH/s (PTX JIT) | 2.24 MH/s |
| tribus | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 100.7 MH/s | 95.5 MH/s |
| whirlpool | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.664 MH/s (PTX JIT) | 0.657 MH/s |
| whirlpool | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 76.2 MH/s | 78.0 MH/s |

Maxwell (`sm_50`, `sm_52`) is untested so far; reports are welcome.
