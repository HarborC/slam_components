set(PROJ4_LIBRARIES /usr/lib/x86_64-linux-gnu/libproj.so)

add_executable(seq_data_test seq_data_test.cpp)
target_link_libraries(seq_data_test sparse_map ${PROJ4_LIBRARIES})

# add_executable(change_dem change_dem.cpp)
# target_link_libraries(change_dem PUBLIC ${PROJECT_NAME})

add_executable(eruco_val eruco_val.cpp)
target_link_libraries(eruco_val PUBLIC ${PROJECT_NAME})


add_executable(cudnn_test cudnn_test.cpp)
target_link_libraries(cudnn_test PUBLIC ${PROJECT_NAME})

add_executable(test_proj test_proj.cpp)
target_link_libraries(test_proj PUBLIC  ${PROJ4_LIBRARIES})