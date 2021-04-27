cmake_minimum_required(VERSION 3.14)
project(Cuda_NNS CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Cuda_NNS main.cu)

set_target_properties(
        Cuda_NNS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


