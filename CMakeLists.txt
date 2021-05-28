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
        src/module.cpp src/brain.cpp
        src/trainer.cpp)
set_target_properties(trainer PROPERTIES LINK_FLAGS "/DEBUG")
target_link_libraries(trainer ${TORCH_LIBRARIES})
