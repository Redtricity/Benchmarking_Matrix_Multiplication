cmake_minimum_required(VERSION 3.18) 

if (DEFINED ENV{SCE_ROOT_DIR})
  set(SCE_PS5_CMAKE "$ENV{SCE_ROOT_DIR}/Prospero/Tools/CMake/PS5.cmake")
  set(CMAKE_TOOLCHAIN_FILE ${SCE_PS5_CMAKE})
endif ()

set(CMAKE_CXX_STANDARD 17)

set(n matmul-assignment)
project(${n})

add_executable(${n} ${n}.cpp stats.cpp stats.hpp)
if (DEFINED ENV{SCE_ROOT_DIR})
  target_link_libraries(${n} -lScePosix_stub_weak)
endif ()

#target_include_directories(${name} PRIVATE $ENV{SCE_PROSPERO_SAMPLE_DIR}/sample_code/common/include)

#target_link_directories(${name} PRIVATE $ENV{SCE_PROSPERO_SAMPLE_DIR}/sample_code/common/source/sampleutil/Prospero_Debug) # or Prospero_Release
#target_link_libraries(${name} SceSampleUtil)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${n})
