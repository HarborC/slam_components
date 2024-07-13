cmake_minimum_required(VERSION 3.10)
project(slam_components)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_definitions(-w)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND)
  message("OpenCV has found.")
  message("OpenCV_VERSION: ${OpenCV_VERSION}")
else()
  message("ERROR: OpenCV could not be found.")
endif()

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(sparse_map)
add_subdirectory(pybind)