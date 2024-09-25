cmake_minimum_required(VERSION 3.10)
project(slam_components)

################################################################################
# Options
################################################################################

option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)

################################################################################
# CMAKE configuration
################################################################################

add_definitions(-w)
add_definitions(-DCOMPILEDWITHC17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++17" )
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CUDA_NVCC_FLAGS "-rdc=true")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS}") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -fPIC ${AVX_FLAGS}")

if(CUDA_ENABLED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

  # Use a separate stream per thread to allow for concurrent kernel execution
  # between multiple threads on the same device.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

  # Suppress warnings:
  # ptxas warning : Stack size for entry function X cannot be statically determined.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-suppress-stack-size-warning")

  # Fix for Ubuntu 16.04.
  add_definitions("-D_MWAITXINTRIN_H_INCLUDED")
endif()

add_definitions(-DBOOST_EXCEPTION_DISABLE
                -DBOOST_GEOMETRY_OVERLAY_NO_THROW
                -DEIGEN_NO_DEBUG
                )

set(ENV{LD_LIBRARY_PATH} "/usr/local/lib/:$ENV{LD_LIBRARY_PATH}")

message("CMAKE_INSTALL_PREFIX:" ${CMAKE_INSTALL_PREFIX})

################################################################################
# Dependency configuration
################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/find_deps.cmake)

add_subdirectory(slam_components)
add_subdirectory(test)
add_subdirectory(pybind)
