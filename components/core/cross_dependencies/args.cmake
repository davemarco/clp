# Check if building for Android
if (ANDROID)
    list(APPEND CMAKE_ARGS
        "-DCMAKE_ANDROID_NDK:PATH =${CMAKE_ANDROID_NDK}"
        "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
        "-DCMAKE_ANDROID_API=${CMAKE_ANDROID_API}"
        "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}"
    )
endif()

list(APPEND CMAKE_ARGS
  "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}"
  "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
  "-DCMAKE_FIND_ROOT_PATH:PATH=${CMAKE_FIND_ROOT_PATH}"
  "-DCMAKE_FIND_DEBUG_MODE=ON"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  "--compile-no-warning-as-error"
)
