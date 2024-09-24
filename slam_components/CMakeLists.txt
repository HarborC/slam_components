include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SRC
  # ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_map/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/network/droid_net/*.cpp
)

add_library(${PROJECT_NAME} SHARED ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${CERES_LIBRARIES} general_camera_model ${TORCH_LIBRARIES})

target_include_directories(${PROJECT_NAME} PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

install(
  DIRECTORY include/
  DESTINATION include/
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}-config.cmake
        DESTINATION share/cmake/${PROJECT_NAME}/
     )

add_subdirectory(apps)