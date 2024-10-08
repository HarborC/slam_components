find_package(pybind11 REQUIRED)
set(PYBIND11_LTO_CXX_FLAGS "")

message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Numpy include dir: ${NUMPY_INCLUDE_DIR}")

include_directories("${NUMPY_INCLUDE_DIR}")

pybind11_add_module(pyslamcpts slamcpts_binds.cpp ndarray_converter.cpp)
target_link_libraries(pyslamcpts PUBLIC ${PYTHON_LIBRARIES} sparse_map pthread)
install(TARGETS pyslamcpts
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)