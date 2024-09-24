add_executable(torch_test torch_test.cpp)
target_link_libraries(torch_test "${TORCH_LIBRARIES}")
set_property(TARGET torch_test PROPERTY CXX_STANDARD 17)

add_executable(torch_script torch_script.cpp)
target_link_libraries(torch_script "${TORCH_LIBRARIES}")
set_property(TARGET torch_script PROPERTY CXX_STANDARD 17)