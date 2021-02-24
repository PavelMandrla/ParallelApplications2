add_library(FreeImage ./linux/libfreeimage.a ./linux/libfreeimageplus.a)
target_include_directories(FreeImage PUBLIC linux)
set_target_properties(FreeImage PROPERTIES LINKER_LANGUAGE CXX)