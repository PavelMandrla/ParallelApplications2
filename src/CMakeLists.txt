cmake_minimum_required(VERSION 3.16)
project(ParallelApplications2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF)

add_subdirectory(../common/ [binary_dir])

add_library(FreeImage STATIC IMPORTED cv7.cu)
set_property(TARGET FreeImage PROPERTY IMPORTED_LOCATION "/usr/lib/libfreeimage.a")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

#find_package(GLFW3 REQUIRED)
#message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

add_executable(PA2 cv9.cu)

set_target_properties(PA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(PA2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_35,code=sm_35;
        -Wno-deprecated-gpu-targets;
        >)

target_link_libraries(PA2
        PG
        CUDA_11
        FreeImage
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        )







