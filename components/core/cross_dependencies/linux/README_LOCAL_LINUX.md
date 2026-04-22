# Linux Overview

Instructions to compile using local linux without sudo


### Build & Install Dependencies

* Create a build directory in linux folder:
  ```shell
   mkdir build
   cd build
  ```

* Generate project files configured for linux target:

  Set the following to configure paths (normally you should just use paths from sample command
  below):
  - install path (CMAKE_INSTALL_PREFIX)
  - root path (CMAKE_FIND_ROOT_PATH)

  ```shell
  cmake \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install \
    -DCMAKE_FIND_ROOT_PATH=$PWD/../install \
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

  ```shell
  cmake \
    -DDEP_DIRECTORY=$PWD/../cross_dependencies/linux/install \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCROSS_COMPILE:BOOL=ON \
    -DNVCOMP_DIR=/path/to/nvcomp \
    -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
    -DCLP_S_SEARCH_TIMING=ON \
    ../
  ```

  * Required GPU dependencies:
    - `-DNVCOMP_DIR` must point to an nvcomp installation containing
      `lib/cmake/nvcomp/` and `include/nvcomp/`.
    - `-DCMAKE_CUDA_COMPILER` must point to the `nvcc` binary.

    For example:
    ```shell
    -DNVCOMP_DIR=/groups/mayygrp/GPU_DBMS/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive
    -DCMAKE_CUDA_COMPILER=/groups/mayygrp/GPU_DBMS/cuda-12.2.2/bin/nvcc
    ```

    -DCMAKE_BUILD_TYPE=Debug \

  * Optional timing build flag:
    Enables search timing collection and JSON output in `clp-s`.
    ```shell
    -DCLP_S_SEARCH_TIMING=ON
    ```
  * Optional timing output directory (runtime):
    Writes `search_timing_<archive_id>.json`.
    ```shell
    export CLP_S_SEARCH_TIMING_OUTPUT_DIR=/path/to/output
    ```

* Build clp:
  ```shell
  make -j
  ```

[1]: https://toolchains.bootlin.com/
