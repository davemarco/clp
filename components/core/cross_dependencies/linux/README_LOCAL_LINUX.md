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
    ../
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
