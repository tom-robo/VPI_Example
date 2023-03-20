cmake_minimum_required(VERSION 3.14)
project(VPI_Example LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED) 

add_executable(${PROJECT_NAME}
  ${PROJECT_NAME}.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_HOME_DIRECTORY}/include
    PUBLIC /usr/include/opencv2/
    PUBLIC /usr/include/opencv4/
    PUBLIC /opt/nvidia/vpi2/include/
    PUBLIC /usr/local/cuda/include/)

target_link_directories(${PROJECT_NAME}
    PUBLIC /opt/nvidia/vpi2/lib64/
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC CUDA::cudart)

target_link_libraries(${PROJECT_NAME}
    PUBLIC nvvpi
    PUBLIC opencv_core
    PUBLIC opencv_imgproc
    PUBLIC opencv_imgcodecs
)


# set_target_properties(${PROJECT_NAME} PROPERTIES
#     VERSION ${PROJECT_VERSION}
#     CMAKE_CXX_STANDARD 11
#     CMAKE_CUDA_STANDARD 11
#     CMAKE_CUDA_STANDARD_REQUIRED TRUE
#     CMAKE_CXX_STANDARD_REQUIRED TRUE)