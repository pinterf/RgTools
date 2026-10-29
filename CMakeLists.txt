# RgTools CMake build: MinGW (GCC, Clang, llvm-mingw), Linux, macOS.
# MSVC uses the solutions: RgTools.slnx (Visual Studio 2026) and RgTools.sln (Visual Studio 2022, XP build).
# Exception: the MSVC ARM64 build, a syntax check of the NEON code (ENABLE_ARM64_WITH_MSVC_TESTING=ON).
#
# Examples:
#   cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_SIMD=OFF   (C only)
cmake_minimum_required(VERSION 3.16...3.31)

project("RgTools" LANGUAGES CXX)
include(GNUInstallDirs)

# Avoid uselessly linking to unused libraries
set(CMAKE_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
set(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Target processor: Intel SIMD or NEON SIMD is turned on automatically.
# CMAKE_GENERATOR_PLATFORM covers the Visual Studio generators (-A ARM64).
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCHID)
if(CMAKE_GENERATOR_PLATFORM)
  string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" ARCHID)
endif()
message(STATUS "Target processor: ${ARCHID}")
set(INTEL_SIMD "OFF")
set(NEON_SIMD "OFF")
if(ARCHID MATCHES "^(x86|x64|i[3-6]86|amd64|x86_64|win32)$")
  set(INTEL_SIMD "ON")
elseif(ARCHID MATCHES "^(aarch64|arm64)$")
  set(NEON_SIMD "ON")
endif()

option(ENABLE_INTEL_SIMD "Enable SIMD intrinsics for Intel processors" "${INTEL_SIMD}")
option(ENABLE_NEON_SIMD "Enable SIMD intrinsics for AArch64 processors" "${NEON_SIMD}")
if(ENABLE_INTEL_SIMD AND ENABLE_NEON_SIMD)
  message(FATAL_ERROR "ENABLE_INTEL_SIMD and ENABLE_NEON_SIMD are mutually exclusive")
endif()

if(MSVC)
  option(ENABLE_ARM64_WITH_MSVC_TESTING "Allow the MSVC ARM64 build (NEON syntax check only, not a release build)" OFF)
  if(NOT (NEON_SIMD AND ENABLE_ARM64_WITH_MSVC_TESTING))
    message(FATAL_ERROR "Use RgTools.slnx (Visual Studio 2026) or RgTools.sln (Visual Studio 2022) for the MSVC "
                        "builds. The MSVC ARM64 syntax check build needs -A ARM64 -DENABLE_ARM64_WITH_MSVC_TESTING=ON.")
  endif()
  add_compile_options(/permissive- /Zc:__cplusplus /W3)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  if(ENABLE_INTEL_SIMD AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    # SSE2 base for 32 bit x86 (x86-64 has it anyway); higher ISA levels are per file or per function
    add_compile_options(-msse2)
  endif()
  if(ENABLE_NEON_SIMD AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC (GNU mode) contracts a*b+c to fmadd also across inlined calls, Clang's default
    # (-ffp-contract=on) only within one expression: with this, the C reference is the same under GCC and Clang.
    add_compile_options(-ffp-contract=fast)
  endif()
  if(WIN32)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--enable-stdcall-fixup")
  elseif(APPLE)
    # macOS uses Clang's linker, doesn't like --no-undefined
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # make sure there are no undefined symbols
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
  endif()
endif()

if(ENABLE_INTEL_SIMD)
  add_compile_definitions(INTEL_INTRINSICS)
elseif(ENABLE_NEON_SIMD)
  add_compile_definitions(NEON_INTRINSICS)
endif()

add_subdirectory("RgTools")

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
