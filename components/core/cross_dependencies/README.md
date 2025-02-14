# Cross Compile Overview

Intructions to cross compile clp and its dependencies. Helpful in situations where building clp and dependencies on target platform is undesirable. Currently supports android and linux targets from linux x86 host.

- `android` [README](android/README.md)
- `linux` [README](linux/README.md)

# Adding New Targets

Designed to cross compile for android/linux targets from a linux x86 host; however,
cmake files are flexible for other cross compilation host and target platforms.
Add toolchain and cmake toolchain file for new target (note some toolchains conveniently include a cmake toolchain file)
New CMake args may be necessary, check [CMake toolchain documentation][1] for more info.
Remove toolchain specific args entirely to compile for host platform.

[1]: https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#id8
