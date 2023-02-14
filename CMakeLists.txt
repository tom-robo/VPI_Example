cmake_minimum_required(VERSION 3.14)
project(VPI_Example)

add_executable(${PROJECT_NAME}
  ${PROJECT_NAME}.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_HOME_DIRECTORY}/include
    PUBLIC /usr/include/opencv2/
    PUBLIC /usr/include/opencv4/
    PUBLIC /opt/nvidia/vpi2/include/)

target_link_directories(${PROJECT_NAME}
    PUBLIC /opt/nvidia/vpi2/lib64/
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC nvvpi
    PUBLIC opencv_core
    PUBLIC opencv_imgproc
    PUBLIC opencv_imgcodecs
)
