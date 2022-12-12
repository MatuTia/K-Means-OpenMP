cmake_minimum_required(VERSION 3.23)
project(Progetto_OpenMP)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)

add_executable(Progetto_OpenMP main.cpp)
target_link_libraries(Progetto_OpenMP OpenMP::OpenMP_CXX)