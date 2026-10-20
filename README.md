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
| `sha256d` | SHA256d (Bitcoin) |

Run `ccminer --help` for the authoritative list.

Tested hardware
---------------

sha256d `--benchmark`, release builds:

| GPU | OS | Driver | CUDA 8 build | CUDA 10.1 build |
|---|---|---|---|---|
| GTX 660M (`sm_30`) | Windows 10 | 425.31 | 46.7 MH/s (PTX JIT) | 46.5 MH/s |
| GTX 1080 Ti (`sm_61`) | Ubuntu 24.04 | 580.173.02 | 1697 MH/s | 1672 MH/s |

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

The binary links the distribution's curl and OpenSSL 1.0 libraries, so run it
where it was built. With the Docker image and the NVIDIA Container Toolkit:

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

Run JIT builds with `CUDA_CACHE_DISABLE=1` when timing or testing them, so the
driver compiles the PTX each time instead of reusing a cached result from an
earlier build.

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
