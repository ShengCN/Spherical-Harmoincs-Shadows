cmake_minimum_required(VERSION 3.8 )
project(CUDA_Proj LANGUAGES CXX CUDA)

# find_package(glfw3 3.3. REQUIRED)
find_package(CUDA  REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
IF (OPENMP_FOUND) 
    # SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp") 
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF() 

add_subdirectory(graphics_lib)

# ImGui
add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD=1)


file(GLOB src
    "${PROJECT_SOURCE_DIR}/*.cu"
    "${PROJECT_SOURCE_DIR}/*.cpp"
)

add_executable(hw ${src})
target_compile_features(hw PUBLIC cxx_std_11)

set_target_properties(hw 
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
) 


target_include_directories(hw PUBLIC
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/graphics_lib/Dep
${CUDA_INCLUDE_DIRS}
)

target_link_libraries(hw PUBLIC
graphics_lib
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hw PUBLIC OpenMP::OpenMP_CXX)
endif()