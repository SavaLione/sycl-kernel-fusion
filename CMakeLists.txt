#######################################################################
# Copyright (C) 2026 Savelii Pototskii (savalione.com)
# 
# Author: Savelii Pototskii <savelii.pototskii@gmail.com>
# 
# This file is part of sycl-kernel-fusion.
# 
# sycl-kernel-fusion is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# 
# sycl-kernel-fusion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with sycl-kernel-fusion. If not, see <https://www.gnu.org/licenses/>.
#######################################################################
cmake_minimum_required (VERSION 3.28)

project(sycl-kernel-fusion
    VERSION "0.0.1"
    DESCRIPTION "Benchmark suite evaluating memory throughput of unfused vs compile-time fused SYCL pointwise kernels."
    HOMEPAGE_URL "https://github.com/SavaLione/sycl-kernel-fusion"
    LANGUAGES C CXX
)

set(PROJECT_AUTHOR "Savelii Pototskii (savalione.com)")
set(PROJECT_LICENSE "GPL-3.0-or-later")

# Project configuration
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)

# Provide compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress 'argument unused during compilation' warnings from AdaptiveCpp/Clang
add_compile_options(-Qunused-arguments)

# Set a default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Dependencies
find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(benchmark)

# Configuration file generation
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h"
)

# Sources and executable definition
set(SPC_SRC
    src/io.cpp
    src/algorithms.cpp
    src/point.cpp
    src/utils.cpp
    src/xgetopt/xgetopt.c
)

set(SPC_EXECUTABLE_SRC
    src/sycl-kernel-fusion.cpp
    ${SPC_SRC}
)

add_executable(${PROJECT_NAME} ${SPC_EXECUTABLE_SRC})

# Define a SYCL AdaptiveCpp project
add_sycl_to_target(
    TARGET ${PROJECT_NAME}
    SOURCES ${SPC_EXECUTABLE_SRC}
)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
)

# Benchmark executable
set(BENCHMARK_SRC
    benchmarks/bm-common.cpp
    benchmarks/bm-algorithms.cpp
    ${SPC_SRC}
)

add_executable(sycl-kernel-fusion-benchmarks ${BENCHMARK_SRC})

add_sycl_to_target(
    TARGET sycl-kernel-fusion-benchmarks
    SOURCES ${BENCHMARK_SRC}
)

target_include_directories(sycl-kernel-fusion-benchmarks PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(sycl-kernel-fusion-benchmarks PUBLIC "${PROJECT_SOURCE_DIR}/benchmarks")
target_link_libraries(sycl-kernel-fusion-benchmarks PRIVATE benchmark::benchmark)
