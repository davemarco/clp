#Set Parameters for Android
#Toolchain file will look different for other platforms

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_API ${CMAKE_ANDROID_API}) # API level
set(CMAKE_ANDROID_ARCH_ABI ${CMAKE_ANDROID_ARCH_ABI})
set(CMAKE_ANDROID_NDK ${CMAKE_ANDROID_NDK})
set(CMAKE_ANDROID_STL_TYPE ${CMAKE_ANDROID_STL_TYPE})
