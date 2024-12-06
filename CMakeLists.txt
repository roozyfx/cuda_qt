cmake_minimum_required(VERSION 3.16)

# # Set GCC and G++ paths explicitly to v.13 so nvcc supports it.
# # Although didn't work and had to set gcc/g++ version using:
# # sudo update-alternatives --config g++
# set(CMAKE_C_COMPILER /usr/bin/gcc-13)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
project(CudaQt VERSION 0.3 LANGUAGES CXX CUDA)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CUDA_SEPARABLE_COMPILATION ON) # Optional, for large CUDA projects

# Set GCC/Clang-specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always --std=c++20")
# Add additional compatibility flags for `nvcc`
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --debug --device-debug")
endif()

set(CPP_SOURCES
        main.cpp
        mainwindow.cpp

        mainwindow.h

        mainwindow.ui
        about.ui
)

# include_directories(${CUDAToolkit_INCLUDE_DIRS})
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(${CMAKE_PROJECT_NAME}
        MANUAL_FINALIZATION
        ${CPP_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET ${CMAKE_PROJECT_NAME} APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(${CMAKE_PROJECT_NAME} SHARED
            ${CPP_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(${CMAKE_PROJECT_NAME}
            ${CPP_SOURCES}
        )
    endif()
endif()

set(CUDA_SOURCES
        cudaImageFuncs.cu
)

target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_SOURCES})

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets
            CUDA::cudart CUDA::cuda_driver
        )

# Specify CUDA properties
# Use `g++` as the C++ compiler for standard files and `nvcc` only for CUDA files
# set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 75
# )

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/g++")
# Include CUDA directories if needed
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings")

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
# if(${QT_VERSION} VERSION_LESS 6.1.0)
#   set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.CudaQt)
# endif()
# set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
#     ${BUNDLE_ID_OPTION}
#     MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
#     MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
#     MACOSX_BUNDLE TRUE
#     WIN32_EXECUTABLE TRUE
# )

include(GNUInstallDirs)
install(TARGETS ${CMAKE_PROJECT_NAME}
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${CMAKE_PROJECT_NAME})
endif()

string(REPLACE "-fdiagnostics-color=always" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("// ${PROJECT_NAME} version : ${PROJECT_VERSION}")
message("//===================================================")
message("  CMake version          : ${CMAKE_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CXX      STANDARD  : ${CMAKE_CXX_STANDARD}")
message("  CUDA CXX STANDARD  : ${CMAKE_CUDA_STANDARD}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("")
message("  OpenMP compile flags : ${OpenMP_CXX_FLAGS}")
message("")
