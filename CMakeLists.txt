cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SoftMaxTest CXX)

# cxx
set(CMAKE_CXX_STANDARD 17)

# use cuda
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "Cannot find CUDA toolkit in this machine")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# use shared cudart
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
message(STATUS "Use CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID}:${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Compiler Path: ${CMAKE_CUDA_COMPILER}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda") # enable lambda in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # enable constexpr expression in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build with CUDA Computation Capability: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(devtest "${CMAKE_SOURCE_DIR}/src/devtest.cc")
target_include_directories(devtest PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(devtest CUDART)

add_executable(softmax_test "${CMAKE_SOURCE_DIR}/src/softmax_test.cu")
target_include_directories(softmax_test PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(softmax_test CUDART)

