# # create camera_model
# add_executable(seq_data_test seq_data_test.cpp)
# target_link_libraries(seq_data_test ${PROJECT_NAME})

# add_executable(change_dem change_dem.cpp)
# target_link_libraries(change_dem PUBLIC ${PROJECT_NAME})

add_executable(eruco_val eruco_val.cpp)
target_link_libraries(eruco_val PUBLIC ${PROJECT_NAME})


add_executable(cudnn_test cudnn_test.cpp)
target_link_libraries(cudnn_test PUBLIC ${CUDA_LIBRARY} ${PROJECT_NAME})