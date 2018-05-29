##
##  Clover: Quantized 4-bit Linear Algebra Library
##              ______ __
##             / ____// /____  _   __ ___   _____
##            / /    / // __ \| | / // _ \ / ___/
##           / /___ / // /_/ /| |/ //  __// /
##           \____//_/ \____/ |___/ \___//_/
##
##  Copyright 2018 Alen Stojanov       (astojanov@inf.ethz.ch)
##                 Tyler Michael Smith (tyler.smith@inf.ethz.ch)
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##

cmake_minimum_required(VERSION 2.8)
project(Clover)

# ======================================================================================================================
# Setup the executable path
# ======================================================================================================================

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)

# ======================================================================================================================
# Setup extrnal cmake Modules
# ======================================================================================================================

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# ======================================================================================================================
# Set compiler flags
# ======================================================================================================================

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    MESSAGE(STATUS "Make sure that Clang has OpenMP support")

    # clang - CC flags
    find_package(MKL)
    find_package(IPP)
    include_directories(${MKL_INCLUDE_DIRS})
    include_directories(${IPP_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -march=haswell")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    # GNU GCC - CC flags
    find_package(MKL)
    find_package(IPP)
    include_directories(${MKL_INCLUDE_DIRS})
    include_directories(${IPP_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -march=haswell -fopenmp")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

    # Intel ICC - CC flags
    find_package(Matlab)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -xHost -mkl=parallel -ipp -qopenmp")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

    # Microsoft Visual Studio - CC flags
    MESSAGE(STATUS "At this point, Microsoft Visual Studio Compiler is not supported")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

endif()

if (STOCHASTIC_ROUNDING_DISABLED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DCLOVER_STOCHASTIC_ROUNDING_DISABLED=\"1\"")
endif(STOCHASTIC_ROUNDING_DISABLED)


# ======================================================================================================================
# Transfer the compiler flags as macros, so they can be inspected by the runtime
# ======================================================================================================================

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMAKE_C_FLAGS=\"${CMAKE_C_FLAGS}\"")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMAKE_C_COMPILER_ID=\"${CMAKE_C_COMPILER_ID}\"")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMAKE_C_COMPILER=\"${CMAKE_C_COMPILER}\"")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMAKE_C_COMPILER_VERSION=\"${CMAKE_C_COMPILER_VERSION}\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_CXX_FLAGS=\"${CMAKE_CXX_FLAGS}\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_CXX_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_CXX_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\"")

include_directories("include")

set(SOURCE_FILES
        #
        # The CloverVector library
        #
        include/CloverBase.h
        include/CloverRandom.h
        include/CloverVector.h
        include/CloverVector4.h
        include/CloverVector8.h
        include/CloverVector16.h
        include/CloverVector32.h
        include/CloverMatrix.h
        include/CloverMatrix4.h
        include/CloverMatrix8.h
        include/CloverMatrix16.h
        include/CloverMatrix32.h
        include/simdxorshift128plus.h
        #
        # Compile the extra libraries
        #
        lib/cpuid.cpp
        lib/cpuid.h
        lib/perf.cpp
        lib/perf.h
        lib/sysinfo.cpp
        lib/sysinfo.h
        lib/simd_debug.h
        lib/simd_debug.cpp
        lib/cxxopts.h 
        #
        # Entry point
        #
        src/main.cpp
        #
        # The testing infrastructure
        #
        test/random/00_random.h
        test/random/00_random.cpp
		#
		# Performance Testing
		#
        test/performance/00_test.cpp
        test/performance/00_test.h
        test/performance/01_measure.h
        test/performance/02_bit04.h
        test/performance/02_bit04.cpp
        test/performance/02_bit08.h
        test/performance/02_bit08.cpp
        test/performance/02_bit16.h
        test/performance/02_bit16.cpp
        test/performance/02_bit32.h
        test/performance/02_bit32.cpp
        test/performance/03_iht_gd_util.h
        test/performance/03_iht_gd_util.cpp
		#
		# Validation Testing
		#
        test/validate/00_validate.cpp
        test/validate/00_validate.h
        test/validate/01_endianess.h
        test/validate/01_endianess.cpp
        test/validate/02_vector.h
        test/validate/02_vector.cpp
        test/validate/03_matrix.cpp
        test/validate/03_matrix.h
		#
		# Accuracy Testing
		#
        test/accuracy/00_accuracy.cpp
        test/accuracy/00_accuracy.h
        test/accuracy/03_gd_accuracy.h
        test/accuracy/02_iht_accuracy.h
        test/accuracy/01_math.h
        test/accuracy/01_math.cpp
        #
        # Grid search
        #
        test/search/00_search.cpp
        test/search/00_search.h
)

add_executable(clover ${SOURCE_FILES})


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(clover ${MKL_LIBRARIES} ${IPP_LIBRARIES} pthread dl)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(clover ${MKL_LIBRARIES} ${IPP_LIBRARIES} pthread dl)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    target_link_libraries(clover pthread dl)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

endif()

# link math library
target_link_libraries(clover m)


