Tested hardware
===============

`--benchmark` with the default intensity, release builds, 60 s runs
(`--time-limit=60`). Fast cards throttle as they heat up, so compare runs
only from the same start temperature (GTX 1080 Ti: anime, blakecoin,
equihash, heavyhash, hmq1725, hoohash, lbry, rinhash, sha3t, sha512256d,
soterg, timetravel, verthash and whirlpoolx2 50 C, others 55 C):

| Algorithm | GPU | OS | Driver | CUDA 8 build | CUDA 10.1 build |
|---|---|---|---|---|---|
| allium | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 133.4 kH/s (PTX JIT) | 131.1 kH/s |
| allium | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 5.90 MH/s | 6.89 MH/s |
| anime | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.116 MH/s (PTX JIT) | 1.130 MH/s |
| anime | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 44.86 MH/s | 45.44 MH/s |
| bc3hash | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 9.49 kH/s (PTX JIT) | 9.60 kH/s |
| bc3hash | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 114.1 kH/s | 114.8 kH/s |
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
| equihash144 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.64 Sol/s (PTX JIT, lean) | 0.65 Sol/s (lean) |
| equihash144 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 78.9 Sol/s | 82.2 Sol/s |
| equihash192 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 44.3 Sol/s | 45.3 Sol/s |
| groestl | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.80 MH/s (PTX JIT) | 1.76 MH/s |
| groestl | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 58.5 MH/s | 59.3 MH/s |
| heavyhash | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 9.10 MH/s (PTX JIT) | 9.03 MH/s |
| heavyhash | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 388.0 MH/s | 426.5 MH/s |
| hmq1725 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 219.4 kH/s (PTX JIT) | 221.2 kH/s |
| hmq1725 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 9.68 MH/s | 9.74 MH/s |
| hoohash | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 63.8 kH/s (PTX JIT) | 63.5 kH/s |
| hoohash | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1180 kH/s | 1191 kH/s |
| keccak | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 21.3 MH/s (PTX JIT) | 21.8 MH/s |
| keccak | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1176 MH/s | 1205 MH/s |
| lbry | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 9.43 MH/s (PTX JIT) | 9.47 MH/s |
| lbry | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 462.5 MH/s | 484.7 MH/s |
| lyra2v2 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 658.6 kH/s (PTX JIT) | 649.6 kH/s |
| lyra2v2 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 59.5 MH/s | 66.5 MH/s |
| lyra2z330 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 237 H/s (PTX JIT) | 237 H/s |
| lyra2z330 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 3934 H/s | 4021 H/s |
| myr-gr | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 3.12 MH/s (PTX JIT) | 3.15 MH/s |
| myr-gr | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 114.0 MH/s | 117.1 MH/s |
| neoscrypt | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 46.6 kH/s (PTX JIT) | 48.9 kH/s |
| neoscrypt | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1559 kH/s | 1743 kH/s |
| quark | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.083 MH/s (PTX JIT) | 1.096 MH/s |
| quark | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 45.0 MH/s | 46.7 MH/s |
| qubit | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.22 MH/s (PTX JIT) | 1.22 MH/s |
| qubit | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 33.0 MH/s | 31.9 MH/s |
| rinhash | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 25.8 kH/s (PTX JIT) | 27.8 kH/s |
| rinhash | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1209 kH/s | 1203 kH/s |
| sha256d | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 46.7 MH/s (PTX JIT) | 46.5 MH/s |
| sha256d | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1687 MH/s | 1856 MH/s |
| sha3t | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 7.87 MH/s (PTX JIT) | 7.84 MH/s |
| sha3t | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 366.5 MH/s | 415.1 MH/s |
| sha512256d | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 11.03 MH/s (PTX JIT) | 11.03 MH/s |
| sha512256d | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 618.1 MH/s | 655.0 MH/s |
| skein | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 22.2 MH/s (PTX JIT) | 22.6 MH/s |
| skein | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 579 MH/s | 653 MH/s |
| skein2 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 20.04 MH/s (PTX JIT) | 20.02 MH/s |
| skein2 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 589.5 MH/s | 589.4 MH/s |
| soterg | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 872.0 kH/s (PTX JIT) | 873.8 kH/s |
| soterg | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 24.31 MH/s | 23.09 MH/s |
| timetravel | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.214 MH/s (PTX JIT) | 1.220 MH/s |
| timetravel | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 45.9 MH/s | 46.9 MH/s |
| tribus | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 2.22 MH/s (PTX JIT) | 2.24 MH/s |
| tribus | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 100.7 MH/s | 95.5 MH/s |
| verthash | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 38.8 kH/s (PTX JIT) | 37.8 kH/s |
| verthash | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 506.0 kH/s | 506.1 kH/s |
| whirlpool | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.664 MH/s (PTX JIT) | 0.657 MH/s |
| whirlpool | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 76.2 MH/s | 78.0 MH/s |
| whirlpoolx2 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 29.04 MH/s (PTX JIT) | 29.07 MH/s |
| whirlpoolx2 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 590.6 MH/s | 595.0 MH/s |
| x11 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.680 MH/s (PTX JIT) | 0.683 MH/s |
| x11 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 22.5 MH/s | 22.3 MH/s |
| x13 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.511 MH/s (PTX JIT) | 0.518 MH/s |
| x13 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 17.8 MH/s | 17.9 MH/s |
| x16r | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.112 MH/s (PTX JIT) | 1.103 MH/s |
| x16r | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 32.1 MH/s | 34.3 MH/s |
| x16rt | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 679.6 kH/s (PTX JIT) | 688.5 kH/s |
| x16rt | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 24.1 MH/s | 25.6 MH/s |
| x16rv2 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 1.101 MH/s (PTX JIT) | 1.104 MH/s |
| x16rv2 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 31.9 MH/s | 32.8 MH/s |
| x16s | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 0.397 MH/s (PTX JIT) | 0.400 MH/s |
| x16s | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 15.1 MH/s | 15.2 MH/s |
| yescryptr8 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | - | 155 H/s |
| yescryptr8 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | - | 17.4 kH/s |
| yescryptr16 | GTX 660M (`sm_30`) | Windows 10 | 425.31 | - | 37 H/s |
| yescryptr16 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | - | 2466 H/s |
| yescryptr32 | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | - | 940 H/s |

x16r, x16rv2 and x16s benchmark one fixed hash order, the same as tpruvot's
ccminer (x16r and x16rv2: ECHO, then 15 times BLAKE; x16s: the order of a zero
previous block hash). x16rt, timetravel and soterg benchmark the order of the
fixed benchmark block time. When mining, the order changes with the block
(x16rt: every 128 s; soterg: with the block time, bits 0 to 4 and 6 cleared),
so the rate varies: pool runs gave 30 to 60 % of the benchmark rate for
x16rv2, 50 to 110 % for x16rt and 80 to 130 % for soterg.

The equihash rate (solutions per second) swings between the 4 s readings, so
the table gives the mean of all readings of the run. On the GTX 660M, which
finishes only two or three solves per reading, it gives the `--debug` count over
the whole run instead. The GTX 660M has too little VRAM for the full 144/5
solver (2.8 GB), so the lean one runs (1.6 GB, about 30 % of the full rate on a
GTX 1080 Ti); 192/7 needs 5.4 GB and does not run.

Maxwell (`sm_50`, `sm_52`) is untested so far; reports are welcome.
