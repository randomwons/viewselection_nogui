cmake_minimum_required(VERSION 3.24)
project(Next-Best-View)

set(ENV_PATH "C:/ProgramData/miniforge3/envs/test")
set(Python3_EXECUTABLE "${ENV_PATH}/python.exe")
set(Python3_INCLUDE_DIR "${ENV_PATH}/include")

add_subdirectory(cpp)