cmake_minimum_required(VERSION 3.23)
project(bp C)

set(CMAKE_C_STANDARD 23)

add_executable(bp lirmain.c lir.c csv.c)
