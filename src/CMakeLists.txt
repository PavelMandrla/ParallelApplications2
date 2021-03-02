cmake_minimum_required(VERSION 3.16)
project(PA2_template LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(../common/ [binary_dir])

add_executable(PA2 cv4.cu)

target_link_libraries(PA2
        CUDA_11
        #FreeImage
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        PG
)
set_target_properties(PA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(PA2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_35,code=sm_35;
        -Wno-deprecated-gpu-targets;
        >)






