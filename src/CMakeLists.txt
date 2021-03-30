cmake_minimum_required(VERSION 3.16)
project(ParallelApplications2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE OFF)

add_subdirectory(../common/ [binary_dir])

add_library(FreeImage STATIC IMPORTED cv7.cu)
set_property(TARGET FreeImage PROPERTY IMPORTED_LOCATION "/usr/lib/libfreeimage.a")

add_executable(PA2
        arrayUtils.cuh
        unifiedMemTest.cu
        )


target_link_libraries(PA2
        PG
        CUDA_11
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        FreeImage
        )
set_target_properties(PA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(PA2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_35,code=sm_35;
        -Wno-deprecated-gpu-targets;
        >)






