# Linux Overview

Instructions to cross-compile using local linux


### Build & Install Dependencies

* Create a build directory in linux folder:
  ```shell
   mkdir build
   cd build
  ```

* Generate project files configured for linux target:

  Set the following to configure paths (normally you should just use paths from sample command
  below):
  - toolchain file path (CMAKE_TOOLCHAIN_FILE)
  - install path (CMAKE_INSTALL_PREFIX)
  - root path (CMAKE_FIND_ROOT_PATH)


  ```shell
  cmake \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install \
    -DCMAKE_FIND_ROOT_PATH=$PWD/../install \
    -DCMAKE_INSTALL_RPATH=$PWD/../install/lib \
    -S ../../

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

  ```

    ```shell
   cmake \
    -DDEP_DIRECTORY=$PWD/../cross_dependencies/linux/install \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCROSS_COMPILE:BOOL=ON \
    ../
  ```

* Build clp:
  ```shell
  make -j
  ```

[1]: https://toolchains.bootlin.com/

