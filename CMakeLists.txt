cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_Proj LANGUAGES CXX CUDA)

add_executable(hw main.cu)
set_target_properties(hw 
    PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    ) 
