cmake_minimum_required(VERSION 3.10)
project(ImageProcessor CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Enable optimization flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

add_executable(image_processor
    src/main.cu
    src/image_processor.cu
    include/image_processor.h
)

target_include_directories(image_processor PRIVATE include)
