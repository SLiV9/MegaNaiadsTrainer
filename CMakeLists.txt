cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(meganaiads-trainer)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(torch REQUIRED)

set(CXX_STANDARD 17)
if(WIN32)
	set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
	add_compile_options("/Ox" "/Oi" "/Ot" "/GT" "/GL" "/arch:AVX2" "/fp:fast" "/Z7")
	add_link_options("/LTCG")
else()
	add_compile_options("-pedantic" "-pedantic-errors" "-Wall" "-Wextra" "-g" "-Ofast" "-march=native" "-flto")
endif()

include_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR})
include_directories(src)

add_executable(trainer src/main.cpp
	libs/lodepng/lodepng.cpp
	src/module.cpp src/trainingbrain.cpp src/trainer.cpp)
set_target_properties(trainer PROPERTIES LINK_FLAGS "/DEBUG")
target_link_libraries(trainer ${TORCH_LIBRARIES})

add_library(libmeganaiads EXCLUDE_FROM_ALL SHARED src/lib.cpp
	src/module.cpp)
target_compile_options(libmeganaiads PRIVATE "-fvisibility=hidden" "-fvisibility-inlines-hidden")
target_link_options(libmeganaiads PRIVATE "-nodefaultlibs" "-ffunction-sections" "-fdata-sections" "-Wl,--gc-sections")
target_link_libraries(libmeganaiads ${TORCH_LIBRARIES})
