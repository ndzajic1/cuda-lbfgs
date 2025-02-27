cmake_minimum_required(VERSION 3.10)
project(LBFGS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lbfgs
    lbfgs.cpp
    vector_utils.cpp  
    line_search.cpp
    benchmark.cpp
    main.cpp
)

target_include_directories(lbfgs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})