cmake_minimum_required(VERSION 3.9.1)
project(gauss)

set(CMAKE_CXX_STANDARD 14)

#add_executable(main main.cpp)
#add_executable(parallel_task main.cpp sequential.cpp)
add_executable(main source.cpp)
#add_executable(main gauss.c)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(main PUBLIC MPI::MPI_CXX)