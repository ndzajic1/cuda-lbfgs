cmake_minimum_required(VERSION 3.10)
project(LBFGS_Optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(lbfgs_example ${SOURCES})

target_compile_options(lbfgs_example PRIVATE -O2)