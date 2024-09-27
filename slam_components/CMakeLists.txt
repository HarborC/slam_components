include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SM_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_map/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
)

add_library(sparse_map STATIC ${SM_SRC})
target_link_libraries(sparse_map PUBLIC 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES} 
  general_camera_model 
  foxglove_lib 
  spdlog::spdlog)
target_include_directories(sparse_map PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(sparse_map PRIVATE cxx_std_17)
set_property(TARGET sparse_map PROPERTY CXX_STANDARD 17)
target_compile_definitions(sparse_map PUBLIC PROJECT_DIR="${PROJECT_SOURCE_DIR}/")

install(TARGETS sparse_map
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS sparse_map
        EXPORT sparse_map-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )

install(EXPORT sparse_map-targets
        FILE sparse_map-config.cmake
        DESTINATION share/cmake/sparse_map/
     )

file(GLOB SRC
  # ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_map/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/network/droid/*
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/network/superpoint/*
  ${CMAKE_CURRENT_SOURCE_DIR}/src/components/network/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES} 
  general_camera_model 
  ${TORCH_LIBRARIES} 
  foxglove_lib 
  general_dataset 
  spdlog::spdlog)
target_include_directories(${PROJECT_NAME} PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PUBLIC PROJECT_DIR="${PROJECT_SOURCE_DIR}/")

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