cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_Proj LANGUAGES CXX CUDA)
     
     
add_executable(hw1 main.cu)
set_target_properties(hw1 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 