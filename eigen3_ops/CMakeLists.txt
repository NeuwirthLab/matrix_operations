cmake_minimum_required(VERSION 3.5)

option(WITH_LIKWID "Likwid marker API" OFF)
option(WITH_PAPI "PAPI hl API" OFF)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(WITH_LIKWID)
	find_package(likwid REQUIRED COMPONENTS marker)
endif()

if(WITH_PAPI)
	find_package(PAPI REQUIRED)
endif()

add_executable(eigen3_ops "matrix_ops.cpp")
target_include_directories(eigen3_ops PRIVATE "../include" "${LIKWID_INCLUDE_DIRS}" "${PAPI_INCLUDE_DIR}")
target_compile_features(eigen3_ops PRIVATE "cxx_std_20")
target_compile_definitions(eigen3_ops PRIVATE "$<$<BOOL:${WITH_LIKWID}>:LIKWID_PERF>" "$<$<BOOL:${WITH_PAPI}>:PAPI_PERF>")
target_link_libraries(eigen3_ops PRIVATE "Eigen3::Eigen" "OpenMP::OpenMP_CXX" "Threads::Threads" "$<$<BOOL:${WITH_LIKWID}>:likwid::likwid>" "$<$<BOOL:${WITH_PAPI}>:PAPI::PAPI>")
