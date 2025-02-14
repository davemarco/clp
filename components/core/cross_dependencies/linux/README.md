# Linux Cross Compile Overview

Instructions to cross-compile using a Linux toolchain

### Install desired Linux toolchain

* If toolchain is not installed, see existing target scripts in toolchains folder (see x86_64_musl example below):
  ```shell
  cd toolchains
  bash download_x86_64_musl.sh
  ```

* If target toolchains script missing, visit [Bootlin][1] and make new script for toolchain with desired architecture + libc, or just download and move into toolchains directory

Bootlin provides free prebuilt toolchains built from buildroot system

### Build & Install Dependencies

* Create a build directory in linux folder:
  ```shell
   mkdir build
   cd build
  ```

* Generate project files configured for linux target:

  Set toolchain to name of installed directory from Bootlin (ex. x86-64--musl--stable-2023.08-1)

  Set the following to configure paths (normally you should just use paths from sample command
  below):
  - toolchain file path (CMAKE_TOOLCHAIN_FILE)
  - install path (CMAKE_INSTALL_PREFIX)
  - root path (CMAKE_FIND_ROOT_PATH)

  See below for a sample command for x86-64--musl Bootlin toolchain

  ```shell
  toolchain=x86-64--musl--stable-2023.08-1; \
  cmake \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/../toolchains/${toolchain}/share/buildroot/toolchainfile.cmake \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install_${toolchain} \
    -DCMAKE_FIND_ROOT_PATH=$PWD/../install_${toolchain} \
    -S ../../
  ```

* Build dependencies:
  ```shell
  cmake --build . -j
  ```

### Build CLP

* Create a build directory in clp core directory:
  ```shell
   cd ../../../
   mkdir build
   cd build
  ```

* Generate project files configured for linux:

  ```shell
   toolchain=x86-64--musl--stable-2023.08-1; \
   cmake \
    -DCLP_USE_STATIC_LIBS:BOOL=ON \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/../cross_dependencies/linux/toolchains/${toolchain}/share/buildroot/toolchainfile.cmake \
    -DDEP_DIRECTORY=$PWD/../cross_dependencies/linux/install_${toolchain} \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCROSS_COMPILE:BOOL=ON \
    ../
  ```

* Build clp:
  ```shell
  make -j
  ```

[1]: https://toolchains.bootlin.com/

