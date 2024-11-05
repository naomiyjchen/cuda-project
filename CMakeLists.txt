cmake_minimum_required(VERSION 3.5)
project (maxk_kernel_test C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Define DEBUG to enable TRACE
# Use with cmake -DCMAKE_BUILD_TYPE=Debug ..
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()


add_executable(${PROJECT_NAME} main.cu spmm_maxk.cu)

# add_executable(${PROJECT_NAME} maxk_kernel.cu)

target_link_libraries(${PROJECT_NAME}  cusparse stdc++fs)
