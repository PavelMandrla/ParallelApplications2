add_library(
        FreeImage
        STATIC
        IMPORTED
    )
set_target_properties(FreeImage
        PROPERTIES
        IMPORTED_LOCATION
        ./linux/libfreeimage.a
        )
target_include_directories(FreeImage INTERFACE ./linux)




#add_library(FreeImage SHARED ./linux/libfreeimage-3.18.0.so)
#target_include_directories(FreeImage PUBLIC ./linux)



#set_target_properties(FreeImage PROPERTIES LINKER_LANGUAGE CXX)

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/linux")
#find_package(FreeImage REQUIRED)
#include_directories(${FREEIMAGE_INCLUDE_DIRS})

#if (${FREEIMAGE_FOUND})
#    message(STATUS "FreeImage was found")
#    message(STATUS ${FREEIMAGE_INCLUDE_DIRS})
#    message(STATUS ${FREEIMAGE_LIBRARIES})
#else()
#    message(FATAL_ERROR "FreeImage was not found")
#endif()
