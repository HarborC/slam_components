find_package(foxglove_lib REQUIRED)

# create camera_model
add_executable(seq_data_test seq_data_test.cpp)
target_link_libraries(seq_data_test ${PROJECT_NAME} foxglove_lib)