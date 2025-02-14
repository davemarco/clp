# Android Cross-Compile Overview

Instructions to cross-compile using Android NDK

### Install Android NDK

* If NDK is not installed, use this script to install:
  ```shell
  cd ndk
  bash download.sh
  ```

### Build & Install Dependencies

* Create a build directory in android folder:
  ```shell
   mkdir build
   cd build
  ```

* Generate project files configured for android target:

  Set the following to configure desired android target:
  - Android Architecture ([ABI/CMAKE_ANDROID_ARCH_ABI][1])
  - Android API level ([CMAKE_ANDROID_API][2])
  - C++ standard library ([CMAKE_ANDROID_STL_TYPE][3])

  Set the following to configure paths (normally you should just use paths from sample command
  below):
  - toolchain path (CMAKE_TOOLCHAIN_FILE)
  - ndk path (CMAKE_ANDROID_NDK)
  - install path (CMAKE_INSTALL_PREFIX)
  - root path (CMAKE_FIND_ROOT_PATH)

  More information on Android CMake args can be found in [CMake documentation][4]

  See below for a sample command

  ```shell
  ABI=arm64-v8a; \
  cmake \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/../android.toolchain.cmake \
    -DCMAKE_ANDROID_NDK=$PWD/../ndk/android-ndk-r25c \
    -DCMAKE_ANDROID_ARCH_ABI=${ABI} \
    -DCMAKE_ANDROID_API=33 \
    -DCMAKE_ANDROID_STL_TYPE=c++_static \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install_${ABI} \
    -DCMAKE_FIND_ROOT_PATH=$PWD/../install_${ABI} \
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

* Generate project files configured for android target:

  Set android values to values set for dependencies

  ```shell
  ABI=arm64-v8a; \
  cmake \
    -DCLP_USE_STATIC_LIBS:BOOL=ON \
    -DCMAKE_ANDROID_ARCH_ABI=${ABI} \
    -DCMAKE_ANDROID_API=33 \
    -DCMAKE_ANDROID_STL_TYPE=c++_static \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/../cross_dependencies/android/android.toolchain.cmake \
    -DCMAKE_ANDROID_NDK=$PWD/../cross_dependencies/android/ndk/android-ndk-r25c/ \
    -DDEP_DIRECTORY=$PWD/../cross_dependencies/android/install_${ABI} \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCROSS_COMPILE:BOOL=ON \
    ../
  ```

* Build clp:
  ```shell
  make -j
  ```

[1]: https://cmake.org/cmake/help/latest/variable/CMAKE_ANDROID_ARCH_ABI.html
[2]: https://cmake.org/cmake/help/latest/variable/CMAKE_ANDROID_API.html#variable:CMAKE_ANDROID_API
[3]: https://cmake.org/cmake/help/latest/variable/CMAKE_ANDROID_STL_TYPE.html#variable:CMAKE_ANDROID_STL_TYPE
[4]: https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-android-with-the-ndk
