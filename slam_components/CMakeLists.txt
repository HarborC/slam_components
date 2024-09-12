find_package(general_camera_model REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_map/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/*.cpp
)

message(STATUS "SRC: ${SRC}")

add_library(${PROJECT_NAME} SHARED ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} general_camera_model)
target_include_directories(${PROJECT_NAME} PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_subdirectory(apps)

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