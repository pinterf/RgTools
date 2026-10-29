set(PluginName "RgTools")

if(NOT WIN32)
  string(TOLOWER "${PluginName}" PluginName)
endif()

include("Files.cmake")

# Initialize a variable to control INTEL_INTRINSICS_AVX512 definition
# AVX512 support is only available on Intel architectures and 64-bit builds and specific compilers
# (from AviSynth+ avs_core/CMakeLists.txt)
set(DEFINE_AVX512 FALSE)
if(ENABLE_INTEL_SIMD)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit build
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      # GCC 9.0 is a reasonable safe minimum for VNNI/VBMI/VBMI2/BITALG
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        set(DEFINE_AVX512 TRUE)
      else()
        message(STATUS "Disabling AVX-512: GCC version ${CMAKE_CXX_COMPILER_VERSION} is too old (requires 9.0+).")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # Clang 9.0 is a reasonable safe minimum for VNNI/VBMI/VBMI2/BITALG
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        set(DEFINE_AVX512 TRUE)
      else()
        message(STATUS "Disabling AVX-512: Clang version ${CMAKE_CXX_COMPILER_VERSION} is too old (requires 9.0+).")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      # Assume modern Intel compilers (icc, icx) support it
      set(DEFINE_AVX512 TRUE)
    endif()
  endif()

  # Note: AVX512 related source must be guarded with #ifdef INTEL_INTRINSICS_AVX512 within the
  # already existing #ifdef INTEL_INTRINSICS
  # Though INTEL_INTRINSICS conditional directive can exist for 32 bit and old compiler versions,
  # but INTEL_INTRINSICS_AVX512 exists only on x64 platforms for compilers supporting AVX512 features
endif()

if(NOT DEFINE_AVX512)
  # Remove AVX512 specific files if it's a 32-bit build or not supported compiler version
  list(FILTER RgTools_Sources EXCLUDE REGEX ".*_avx512\\.(cpp|hpp|h)$")
  list(FILTER RgTools_Sources EXCLUDE REGEX ".*_avx512b\\.(cpp|hpp|h)$")
endif()

# Per-file ISA flags by file name suffix (from AviSynth+ avs_core/CMakeLists.txt).
# The files found are appended to the sources, so they can live in a subdirectory as well (aarch64/).
function(handle_arch_flags ARCH_SUFFIX GCC_FLAGS MSVC_FLAGS)
  string(TOLOWER "${ARCH_SUFFIX}" ARCH_SUFFIX_LOWER)
  file(GLOB_RECURSE SRCS_${ARCH_SUFFIX} RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_${ARCH_SUFFIX_LOWER}.cpp")
  if(SRCS_${ARCH_SUFFIX})
    if(MSVC) # here: the MSVC ARM64 syntax check build only
      set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${MSVC_FLAGS}")
    else()
      # GCC, Clang on Linux/macOS, MinGW
      set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
    endif()
    list(APPEND RgTools_Sources ${SRCS_${ARCH_SUFFIX}})
    list(REMOVE_DUPLICATES RgTools_Sources)
    set(RgTools_Sources "${RgTools_Sources}" PARENT_SCOPE)
  endif()
endfunction()

if(ENABLE_NEON_SIMD)
  # =========================================================================
  # AArch64 (ARM64) SIMD Flags
  # =========================================================================
  # Baseline NEON (ASIMD)
  # GCC/Clang: Implicit (or -march=armv8-a) MSVC: /arch:armv8.0
  handle_arch_flags(NEON " -march=armv8-a " " /arch:armv8.0 ")

  # Dot Product (FEAT_DotProd: optional from ARMv8.2-A, compulsory from ARMv8.4-A)
  # armv8.2-a base: GCC 11/12 arm_neon.h define vdotq_u32 for arch=armv8.2-a+dotprod, armv8.1-a+dotprod fails there.
  handle_arch_flags(DP " -march=armv8.2-a+dotprod " " /arch:armv8.2 ")

  # Int8 Matrix Multiply (FEAT_I8MM / ARMv8.2-A) Compulsory from ARMv8.6-A
  handle_arch_flags(I8MM " -march=armv8.2-a+i8mm " " /arch:armv8.2 ")

  # SVE2 (Scalable Vector Extension 2) (FEAT_SVE2 / ARMv8.5-A) Compulsory from ARMv9.0-A
  handle_arch_flags(SVE2 " -march=armv8.5-a+sve2 " " /arch:armv8.5 ")

  # SVE2.1 (FEAT_SVE2p1 / ARMv9.1-A)
  # Needs GCC 15 or a newer Clang (Clang 22 OK, Clang 15 not).
  # Once there are *_sve2_1.cpp files, an older compiler fails on the unknown +sve2p1 modifier:
  # check the flag first (check_cxx_compiler_flag), skip the files and guard their dispatch with a define.
  handle_arch_flags(SVE2_1 " -march=armv9.1-a+sve2p1 " " /arch:armv9.1 ")

elseif(ENABLE_INTEL_SIMD)
  # =========================================================================
  # Intel (x86/x64) SIMD Flags
  # =========================================================================
  # gcc/llvm/clang-like flags, MSVC flags
  handle_arch_flags(SSSE3 " -mssse3 " " /arch:SSE2 ") # no special SSSE3 option in MSVC
  handle_arch_flags(SSE41 " -msse4.1 " " /arch:SSE2 ") # no special SSE4.1 option in MSVC
  handle_arch_flags(AVX " -mavx " " /arch:AVX ")
  handle_arch_flags(AVX2 " -mavx2 -mfma " " /arch:AVX2 ")

  if(DEFINE_AVX512)
    # Similar to AVX2, FMA must be added explicitly for GCC/Clang.
    # We also enable BMI2, as it is standard since AVX2 and useful for
    # AVX-512 mask operations like _bzhi_u64/u32.
    # The core subsets (F, CD, BW, DQ, VL) comprising the CPUF_AVX512_BASE
    # flag are present on all AVX-512-capable architectures.
    #
    # Note: Early Xeon (e.g., Skylake-X/Cascadelake) may exhibit severe
    # thermal throttling even on a single thread. In AviSynth+, AVX512_BASE
    # must be manually enabled in-script unless CPUF_AVX512_FAST is also detected.
    # Functions in *_avx512b.cpp should be dispatched on CPUF_AVX512_BASE.
    set(AVX512_BASE_GCC_FLAGS
      " -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl "
    )

    # CPUF_AVX512_FAST indicates a modern, low-throttling architecture (ICL/RKL/AVX10.x).
    # This set represents the intersection of Icelake and AVX10.1 features:
    # (F, CD, BW, DQ, VL, VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ).
    # Functions in *_avx512.cpp should be dispatched on CPUF_AVX512_FAST.
    set(AVX512_FAST_GCC_FLAGS
      " -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "
    )

    # Note: MSVC's /arch:AVX512 enables the basic subset (F, CD, BW, DQ, VL).
    handle_arch_flags(AVX512B "${AVX512_BASE_GCC_FLAGS}" " /arch:AVX512 ") # *_avx512b.cpp
    handle_arch_flags(AVX512 "${AVX512_FAST_GCC_FLAGS}" " /arch:AVX512 ") # *_avx512.cpp
  endif()
endif()

add_library(${PluginName} SHARED ${RgTools_Sources})

if(DEFINE_AVX512)
  target_compile_definitions(${PluginName} PRIVATE INTEL_INTRINSICS_AVX512)
endif()

set_target_properties(${PluginName} PROPERTIES "OUTPUT_NAME" "${PluginName}")
if(MINGW)
  set_target_properties(${PluginName} PROPERTIES PREFIX "")
  set_target_properties(${PluginName} PROPERTIES IMPORT_PREFIX "")
endif()

# version resource
if(WIN32)
  enable_language(RC)
  target_sources(${PluginName} PRIVATE "rgtools.rc")
endif()

# Automatically group source files according to directory structure
foreach(FILE ${RgTools_Sources})
  get_filename_component(PARENT_DIR "${FILE}" PATH)

  string(REGEX REPLACE "(\\./)" "" GROUP "${PARENT_DIR}")
  string(REPLACE "/" "\\" GROUP "${GROUP}")

  # group into "Source Files" and "Header Files"
  if("${FILE}" MATCHES ".*\\.cpp")
    set(GROUP "Source Files\\${GROUP}")
  elseif("${FILE}" MATCHES ".*\\.h")
    set(GROUP "Header Files\\${GROUP}")
  endif()

  source_group("${GROUP}" FILES "${FILE}")
endforeach()

target_include_directories(${PluginName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# dedicated include dir for avisynth.h
target_include_directories(${PluginName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core test program: the cores and kernels without the AviSynth front ends (no avisynth.h include directory).
# The per-file ISA flags above are source file properties of this directory: they apply here as well.
if(RGTOOLS_BUILD_TESTS)
  set(RgTools_Core_Sources ${RgTools_Sources})
  list(FILTER RgTools_Core_Sources EXCLUDE REGEX "^(avs2x|clense|removegrain|repair|vertical_cleaner|RemoveGrainT)\\.cpp$")
  list(FILTER RgTools_Core_Sources EXCLUDE REGEX "^include/")
  add_executable(rgtools_core_test "${CMAKE_CURRENT_SOURCE_DIR}/../tests/core_test.cpp" ${RgTools_Core_Sources})
  target_include_directories(rgtools_core_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(DEFINE_AVX512)
    target_compile_definitions(rgtools_core_test PRIVATE INTEL_INTRINSICS_AVX512)
  endif()
endif()

install(TARGETS ${PluginName}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")
