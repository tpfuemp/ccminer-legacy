# ccminer-legacy

A CUDA miner for older NVIDIA GPUs. It is a legacy fork of
[ccminer-tpfuemp](https://github.com/tpfuemp/ccminer-tpfuemp), which is based on
[ccminer-kudaraidee](https://github.com/Kudaraidee/ccminer-kudaraidee) and
[tpruvot/ccminer](https://github.com/tpruvot/ccminer).

ccminer-tpfuemp targets CUDA 11.8 and Pascal or newer GPUs, and has dropped the
code paths for anything older. This fork keeps its host side (reorganised source
tree, pool handling, REST API) and builds it with **CUDA 8** or **CUDA 10.1**,
so it runs on **Kepler and Maxwell** cards again.

See [CREDITS.txt](CREDITS.txt) for lineage, contributor credits and the donation
address.

Requirements
------------

- **CUDA Toolkit:** two builds are maintained side by side:
  - **CUDA 8.0** runs on the widest range of drivers (376.51 or newer on
    Windows, 375.26 or newer on Linux).
  - **CUDA 10.1** needs driver 418.96 or newer (Linux: 418.39) and generates
    fast Kepler code itself. CUDA 10.1 is the last toolkit that still supports
    Kepler and runs on the last Kepler mobile drivers; CUDA 11 dropped `sm_30`.
- **Build toolchain (Windows):** Visual Studio 2015 (v140), the newest host
  compiler CUDA 8 supports. Solution `ccminer.sln` with the projects
  `ccminer-cuda8.vcxproj` and `ccminer-cuda10.vcxproj`, x64 only.
- **Build toolchain (Linux):** GCC 5 and autotools. CUDA 8 does not accept a
  newer GCC, so on current distributions build in a container.
- **Supported GPU architectures:** Kepler (`sm_30`, `sm_35`, `sm_37`) and
  Maxwell (`sm_50`, `sm_52`). The default build also includes native `sm_61`
  code and `compute_61` PTX so newer cards can JIT-compile it.
- **The CUDA 8 build runs Kepler as JIT-compiled PTX**, because the driver
  JIT generates much faster Kepler code than CUDA 8 itself (sha256d on a
  GTX 660M: 46.7 vs 25 MH/s). It is fast only with a recent driver (425 or
  newer); the first start compiles the kernels, the result is cached. The
  CUDA 10.1 build embeds native Kepler code with the same speed.
- **Not supported:** Fermi (`sm_20/21`), which lacks warp shuffle and funnel
  shift.

**On Pascal or newer hardware?** Use
[ccminer-tpfuemp](https://github.com/tpfuemp/ccminer-tpfuemp); it is faster there
and has many more algorithms.

Supported algorithms
--------------------

Select an algorithm with `-a <name>`. They are ported one at a time from the
pre-refactor history of ccminer-tpfuemp. The miner only accepts the ones
compiled in, both on the command line and when a pool switches algorithm.

| `-a` name | Coin / description |
|---|---|
| `blake` | Blake-256, 14 rounds (Saffroncoin) |
| `blakecoin` | Blake-256, 8 rounds (Blakecoin) |
| `keccak` | Keccak-256 (Maxcoin) |
| `keccakc` | Keccak-256 (CreativeCoin), pool difficulty factor 256 |
| `sha256d` | SHA256d (Bitcoin) |
| `skein` | Skein-512 + SHA256 (Skeincoin) |

Run `ccminer --help` for the authoritative list.

Tested hardware
---------------

`--benchmark` with the default intensity, release builds, 60 s runs
(`--time-limit=60`). Fast cards throttle as they heat up, so compare runs
only from the same start temperature (GTX 1080 Ti: blakecoin 50 C, others 55 C):

| Algorithm | GPU | OS | Driver | CUDA 8 build | CUDA 10.1 build |
|---|---|---|---|---|---|
| sha256d | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 46.7 MH/s (PTX JIT) | 46.5 MH/s |
| sha256d | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1687 MH/s | 1856 MH/s |
| blake | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 123.6 MH/s (PTX JIT) | 126.1 MH/s |
| blake | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 4059 MH/s | 4164 MH/s |
| blakecoin | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 227.8 MH/s (PTX JIT) | 227.6 MH/s |
| blakecoin | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 7255 MH/s | 7484 MH/s |
| skein | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 22.2 MH/s (PTX JIT) | 22.6 MH/s |
| skein | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 579 MH/s | 653 MH/s |
| keccak | GTX 660M (`sm_30`) | Windows 10 | 425.31 | 21.3 MH/s (PTX JIT) | 21.8 MH/s |
| keccak | GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1176 MH/s | 1205 MH/s |

Maxwell (`sm_50`, `sm_52`) is untested so far; reports are welcome.

Building on Windows
-------------------

Install the CUDA 8.0 and/or 10.1 toolkits with their Visual Studio 2015
integration. Then open `ccminer.sln` in Visual Studio 2015 and build the
`Release|x64` configuration. From the command line (VS2015 x64 developer
prompt):

    msbuild ccminer.sln /p:Configuration=Release /p:Platform=x64 /m

This builds `x64\Release-cuda8\ccminer-cuda8.exe` and
`x64\Release-cuda10\ccminer-cuda10.exe`. To build only one, add
`/t:ccminer-cuda8` or `/t:ccminer-cuda10`.

Both projects take their settings and source lists from `ccminer.props`. The
target GPU architectures are the `CudaArchs` property in each project file.

If the CUDA 10.1 Visual Studio integration is not installed, point
`CudaMSBuildDir` (a property or environment variable) at a folder that holds
`CUDA 10.1.props`, `CUDA 10.1.targets`, `CUDA 10.1.xml` and
`Nvda.Build.CudaTasks.v10.1.dll`. The CUDA installer contains them under
`CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions`.

Building on Linux
-----------------

Requires the CUDA 8.0 and/or 10.1 toolkit, GCC 5, autoconf/automake, and the
`curl`, `openssl` and `zlib` development packages. CUDA 8 refuses newer host
compilers, so use Ubuntu 16.04 (GCC 5.4). `docker/Dockerfile` sets that up with
both toolkits:

    docker build -t ccminer-legacy-build docker/
    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
        ccminer-legacy-build ./build.sh
    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
        -e CUDA=10.1 ccminer-legacy-build ./build.sh

`build.sh` runs `autogen.sh` and builds out of tree with the toolkit in
`/usr/local/cuda-$CUDA` (default `CUDA=8.0`), into `build-linux/cuda8/ccminer`
or `build-linux/cuda10/ccminer`. Its arguments go to `configure`:

- `--with-cuda=PATH`: CUDA toolkit prefix (default `/usr/local/cuda`).
- `--with-cuda-arch=LIST`: GPU code to embed. The default is
  `"ptx30 ptx35 ptx37 50 52 61"` with CUDA 8, and `"30 35 37 50 52 61"` with
  newer toolkits.
  `N` embeds `sm_N` code plus `compute_N` PTX, `ptxN` embeds PTX only (for
  testing the driver JIT, e.g. `--with-cuda-arch=ptx52`).
- `--with-nvcc-ccbin=PATH`: host compiler for nvcc, for systems whose default
  GCC is too new for the toolkit (CUDA 8 needs GCC 5 or older, CUDA 10.1 GCC 8
  or older), e.g. `--with-nvcc-ccbin=g++-5`. Use the same version for `CXX`.

- `--with-static-deps=PREFIX`: link curl, OpenSSL and zlib statically from
  PREFIX, plus the C++ runtime. The binary then needs only glibc 2.17 or newer
  and the NVIDIA driver, as the release binaries do. `docker/build-deps.sh`
  builds the libraries (the same versions as for Windows, see
  `compat/BUILD-DEPS.txt`); the Docker image has them in `/opt/ccminer-deps`:

      docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
          ccminer-legacy-build ./build.sh --with-static-deps=/opt/ccminer-deps

Without it the binary links the distribution's curl and OpenSSL 1.0 libraries,
so run it where it was built. With the Docker image and the NVIDIA Container
Toolkit:

    docker run --rm --gpus all -v "$PWD:/src" -w /src \
        ccminer-legacy-build build-linux/cuda8/ccminer -a sha256d --benchmark

Test builds
-----------

To compare native code with the driver JIT, build variants with other GPU code
into their own directories:

    msbuild ccminer.sln /p:Configuration=Release /p:Platform=x64 /t:ccminer-cuda8 ^
        /p:CudaArchs="compute_52,compute_52" /p:BuildTag=-jit52
    TAG=-jit52 ./build.sh --with-cuda-arch=ptx52

These build `x64\Release-cuda8-jit52\ccminer-cuda8.exe` and
`build-linux/cuda8-jit52/ccminer`. `CudaArchs` takes nvcc
`compute_N,sm_N` pairs (`compute_N,compute_N` embeds PTX only). Common
variants: `-sm30` (`compute_30,sm_30` / `30`), `-jit52` and `-jit35` (PTX
only), `-fat` (every arch native plus PTX).

In an existing Linux build directory, `make NVCC_ARCH=...` overrides the
configured GPU code with raw nvcc flags, e.g.
`make NVCC_ARCH="-gencode=arch=compute_52,code=compute_52"` (touch or clean
the `.cu` objects first, make does not track the flags).

`cuobjdump` from the toolkit shows what a binary contains: `--list-elf` the
native code, `--list-ptx` the PTX the driver can JIT-compile (for GPUs of that
architecture or newer):

    cuobjdump --list-elf --list-ptx build-linux/cuda8-jit52/ccminer

Run JIT builds with `CUDA_CACHE_DISABLE=1` when timing or testing them, so the
driver compiles the PTX each time instead of reusing a cached result from an
earlier build:

    CUDA_CACHE_DISABLE=1 build-linux/cuda8-jit52/ccminer -a sha256d --test-vectors
    set CUDA_CACHE_DISABLE=1
    x64\Release-cuda8-jit52\ccminer-cuda8.exe -a sha256d --test-vectors

Test vectors
------------

`ccminer --test-vectors` checks every algo in the build against known block
hashes and exits with 0 when all pass, 3 when one fails. Each vector is first
hashed on the CPU, then every GPU scans a nonce range that ends at the
vector's nonce and must find it, with every reported nonce verified on the
CPU. No pool is needed, and `-d` selects the GPUs as usual:

    ccminer --test-vectors -d 0

The vectors live next to the algo in `algos/<name>/register.cpp`
(`ALGO_TEST_VECTORS`); sha256d uses the Bitcoin genesis block.

Source lists
------------

Both builds keep their own source lists: `ccminer.props` for Windows,
`Makefile.am` and `algos/<name>/algo.mk` for Linux. After adding or removing a
source file, or changing a per-file `maxrregcount`, run `./check-sync.sh`
(any POSIX shell, including Git Bash). It reports every difference between the
two builds and the files on disk.

Monitoring and control
----------------------

The miner API listens on `--api-bind` (default `127.0.0.1:4068`) and speaks two
protocols, chosen with `--api-mode=binary|http|both`:

- **Binary/telnet API**, the upstream protocol, unchanged.
- **HTTP/JSON REST API**. Read-only status, plus optional **runtime control**:
  switch pool or algorithm, pause and resume mining, and apply per-algo
  parameters without restarting. It also exposes **Prometheus** metrics
  (`/metrics`) and a `/health` endpoint for container probes. `--api-http-port`
  serves HTTP on its own port.

Changing things is refused unless explicitly enabled: `--api-control` for
pause/resume, `--api-remote` for pool and algorithm switching. `--api-token`
requires a bearer token and `--api-allow` restricts client addresses, so a
dashboard can get status access without the ability to redirect hashrate.

On mobile and Optimus GPUs, and with older drivers, NVML often reports partial or
no data. Missing values are left out instead of stopping the miner.

Source-code dependencies
------------------------

The build links against OpenSSL, libcurl, zlib and pthreads. For Windows,
prebuilt x64 `.lib` files (VS2015, static CRT) are included in `compat/`, together
with their headers. [compat/BUILD-DEPS.txt](compat/BUILD-DEPS.txt) lists the
versions, source checksums and exact build steps to reproduce them.

Credits & donation
------------------

Lineage, contributor credits and the donation address are in
[CREDITS.txt](CREDITS.txt).
