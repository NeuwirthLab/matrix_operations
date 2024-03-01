cmake_minimum_required(VERSION 3.5)

project(matrix_operations)
option(WITH_LIKWID "Likwid marker API" OFF)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(WITH_LIKWID)
	find_package(likwid REQUIRED COMPONENTS marker)
	message(STATUS "USING LIKWID")
	message(STATUS "INC DIR ${LIKWID_INCLUDE_DIRS}")
endif()

add_executable(${PROJECT_NAME} "matrix_ops.cpp")
target_include_directories(${PROJECT_NAME} PRIVATE "../include" "${LIKWID_INCLUDE_DIRS}")
target_compile_features(${PROJECT_NAME} PRIVATE "cxx_std_20")
target_compile_definitions(${PROJECT_NAME} PRIVATE "$<$<BOOL:${WITH_LIKWID}>:LIKWID_PERF>")
target_link_libraries(${PROJECT_NAME} PRIVATE "Eigen3::Eigen" "OpenMP::OpenMP_CXX" "Threads::Threads" "$<$<BOOL:${WITH_LIKWID}>:likwid::likwid>")
