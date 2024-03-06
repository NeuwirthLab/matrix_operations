cmake_minimum_required(VERSION 3.5)
project(matrix_operations)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

option(EIGEN_OPS "Contains the old code" OFF)
option(TACO_OPS "New code, performs desne MV" ON)

if(EIGEN_OPS)
	add_subdirectory(eigen3_ops)
endif()
if(TACO_OPS)
	add_subdirectory(taco_ops)
endif()
