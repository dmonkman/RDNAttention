cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

project(rdnattention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Every target with native v_dot2_f32_f16 and v_dot4_i32_i8.
set(RDNATTENTION_TIER1_ARCHS
    gfx906                                  # Radeon VII, Instinct MI50/MI60
    gfx1011 gfx1012                         # RDNA1: Radeon Pro 5600M, RX 5500 XT
    gfx1030 gfx1031 gfx1032                 # RDNA2: RX 6800/6900, 6700, 6600
    gfx1033 gfx1034 gfx1035 gfx1036         # Steam Deck, RX 6400/6500, Rembrandt/Raphael
)

if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${RDNATTENTION_TIER1_ARCHS}")
endif()

# Tier 2/3 targets with no dot-product ALUs at all. The INT8 kernel will
# refuse to compile, so building for them requires turning off INT8
set(RDNATTENTION_NO_DOT_ARCHS
    gfx803 gfx900 gfx902 gfx904 gfx909 gfx90c gfx1010 gfx1013)

option(RDNATTENTION_ENABLE_INT8 "Build the INT8 QK^T kernel (needs native v_dot4_i32_i8)" ON)

if(RDNATTENTION_ENABLE_INT8)
    foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(REGEX REPLACE ":.*$" "" arch_base "${arch}")
        if(arch_base IN_LIST RDNATTENTION_NO_DOT_ARCHS)
            message(FATAL_ERROR
                "Architecture '${arch}' has no native v_dot4_i32_i8, so the INT8 kernel cannot "
                "be built for it. Reconfigure with -DRDNATTENTION_ENABLE_INT8=OFF to build the "
                "fp16 kernel only, or drop '${arch}' from CMAKE_HIP_ARCHITECTURES.")
        endif()
    endforeach()
endif()

enable_language(HIP)
find_package(hip REQUIRED)

set(RDNATTENTION_SOURCES src/c_api.cpp src/rdna/fa2_forward_f16.hip src/rdna/monarch_forward_f16.hip)
set(RDNATTENTION_HIP_SOURCES src/rdna/fa2_forward_f16.hip src/rdna/monarch_forward_f16.hip)
if(RDNATTENTION_ENABLE_INT8)
    list(APPEND RDNATTENTION_SOURCES src/rdna/fa2_forward_int8qk.hip)
    list(APPEND RDNATTENTION_HIP_SOURCES src/rdna/fa2_forward_int8qk.hip)
endif()

add_library(rdnattention SHARED ${RDNATTENTION_SOURCES})
target_include_directories(rdnattention PUBLIC src)
set_source_files_properties(${RDNATTENTION_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
if(NOT RDNATTENTION_ENABLE_INT8)
    target_compile_definitions(rdnattention PRIVATE RDNATTENTION_NO_INT8)
endif()

# Testing hook: force a specific fdot2() lowering instead of letting arch.hpp
# pick one for the target (1=builtin dot, 2=portable, 3=mixed-FMA asm,
# 4=cvt+fma). Lets a dot-capable card execute the fallbacks it would never
# select. For testing, not for production builds.
set(RDNATTENTION_FORCE_DOT_PATH "" CACHE STRING "Force fdot2 lowering: 1=builtin, 2=portable, 3=mixasm, 4=cvtfma")
if(RDNATTENTION_FORCE_DOT_PATH)
    target_compile_definitions(rdnattention PRIVATE RDNA_FORCE_DOT_PATH=${RDNATTENTION_FORCE_DOT_PATH})
    message(STATUS "rdnattention: FORCING fdot2 path ${RDNATTENTION_FORCE_DOT_PATH} (testing build)")
endif()

target_compile_options(rdnattention PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
target_link_libraries(rdnattention PUBLIC hip::host)

if(MSVC)
    target_compile_options(rdnattention PRIVATE /W4)
else()
    target_compile_options(rdnattention PRIVATE -Wall -Wextra)
endif()

target_compile_options(rdnattention PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Werror=unknown-attributes>)
