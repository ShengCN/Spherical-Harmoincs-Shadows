cmake_minimum_required(VERSION 3.8)

project(libgraphics_lib)
find_package(OpenGL REQUIRED)

set (CMAKE_CXX_STANDARD 11)


# ImGui
add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD=1)
file(GLOB render_src
    "${PROJECT_SOURCE_DIR}/Render/*.cpp"
    "${PROJECT_SOURCE_DIR}/Dep/glad/*.c"
    "${PROJECT_SOURCE_DIR}/Utilities/*.cpp"
    "${PROJECT_SOURCE_DIR}/Dep/tinyobjloader/*.cc"
    "${PROJECT_SOURCE_DIR}/Dep/tinyxml2/*.cpp"
    "${PROJECT_SOURCE_DIR}/*.cpp" 
    "${PROJECT_SOURCE_DIR}/Dep/imgui/*.cpp"
    "${PROJECT_SOURCE_DIR}/Dep/imgui/examples/imgui_impl_glfw.cpp"
    "${PROJECT_SOURCE_DIR}/Dep/imgui/examples/imgui_impl_opengl3.cpp")

find_package(Boost COMPONENTS system filesystem REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/Dep/glm)
add_library(graphics_lib SHARED ${render_src})

target_include_directories(graphics_lib PUBLIC 
${PROJECT_SOURCE_DIR}/Dep/stb
${PROJECT_SOURCE_DIR}/Dep/imgui
${PROJECT_SOURCE_DIR}/Dep
${PROJECT_SOURCE_DIR}
${Boost_INCLUDE_DIR})

add_definitions(-DGLM_FORCE_RADIANS)
target_link_libraries(graphics_lib 
${Boost_LIBRARIES})

target_link_libraries(graphics_lib ${OPENGL_gl_LIBRARY})

# install(TARGETS graphics_lib DESTINATION lib)
# install(FILES *.h DESTINATION include)