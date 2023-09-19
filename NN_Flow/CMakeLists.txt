cmake_minimum_required(VERSION 3.5)

project(NN_Flow LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/home/ilya/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

################# create executable app #################
add_executable(${PROJECT_NAME} main.cpp)

################# link connect #################
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
