set(CUDA_MIN_VERSION "10.0")
if(CUDA_ENABLED)
    if(CMAKE_VERSION VERSION_LESS 3.17)
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA version ${CUDA_VERSION} installed in "
                    "${CUDA_TOOLKIT_ROOT_DIR} via legacy CMake (<3.17) module. "
                    "Using the legacy CMake module means that any installation of "
                    "COLMAP will require that the CUDA libraries are "
                    "available under LD_LIBRARY_PATH.")
            message(STATUS "Found CUDA ")
            message(STATUS "  Includes : ${CUDA_INCLUDE_DIRS}")
            message(STATUS "  Libraries : ${CUDA_LIBRARIES}")

            enable_language(CUDA)

            macro(declare_imported_cuda_target module)
                add_library(CUDA::${module} INTERFACE IMPORTED)
                target_include_directories(
                    CUDA::${module} INTERFACE ${CUDA_INCLUDE_DIRS})
                target_link_libraries(
                    CUDA::${module} INTERFACE ${CUDA_${module}_LIBRARY} ${ARGN})
            endmacro()

            declare_imported_cuda_target(cudart ${CUDA_LIBRARIES})
            declare_imported_cuda_target(curand ${CUDA_LIBRARIES})
            
            set(CUDAToolkit_VERSION "${CUDA_VERSION_STRING}")
            set(CUDAToolkit_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        endif()
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND ON)
            enable_language(CUDA)
        endif()
    endif()
endif()

if(CUDA_ENABLED AND CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    add_definitions("-DHX_SLAM_CUDA_ENABLED")

    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    # Explicitly set PIC flags for CUDA targets.
    if(NOT IS_MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION}, "
                    "archs: ${CMAKE_CUDA_ARCHITECTURES})")
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif()

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

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIR})

find_package(spdlog REQUIRED)

find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)
find_package(Torch REQUIRED)
# add_definitions(-DC10_USE_GLOG)

include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CUDNN_INCLUDE_PATH: ${CUDNN_INCLUDE_PATH}")
message(STATUS "CUDNN_LIBRARY_PATH: ${CUDNN_LIBRARY_PATH}")
message(STATUS "CUDNN_LIBNAME: ${CUDNN_LIBNAME}")
message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
message(STATUS "CUDNN_VERSION: ${CUDNN_VERSION}")

set(Python_EXECUTABLE /opt/conda/envs/slam4labeling/bin/python)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(general_camera_model REQUIRED)
find_package(foxglove_lib REQUIRED)
find_package(general_dataset REQUIRED)
