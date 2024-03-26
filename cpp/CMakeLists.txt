cmake_minimum_required(VERSION 3.24)
project(DirectionVoxelGrid)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP)

add_subdirectory(thirdparty/pybind11)
pybind11_add_module(${PROJECT_NAME}
                src/binding.cpp
                src/VoxelGrid.cpp
                src/policies.cpp
                src/policies_face.cpp        
)
include_directories("include")
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)