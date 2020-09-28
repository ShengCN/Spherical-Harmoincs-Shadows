cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_Proj LANGUAGES CXX CUDA)

FIND_PACKAGE(OpenMP REQUIRED)
IF (OPENMP_FOUND) 
    # SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp") 
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()  

add_executable(hw main.cu)
set_target_properties(hw 
    PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    ) 

if(OpenMP_CXX_FOUND)
    target_link_libraries(hw PUBLIC OpenMP::OpenMP_CXX)
endif()