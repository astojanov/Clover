#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include "cpuid.h"

#if defined(__APPLE__)
#include <mach/machine.h>
#elif defined(__linux__) || defined(linux) || defined(__linux)
#include <unistd.h>
#endif

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

size_t get_system_pagesize ()
{
#if defined(__APPLE__)
    return PAGE_SIZE;
#elif defined(__linux__) || defined(linux) || defined(__linux)
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}


/**
 * The vendor ID of the CPU. For example: GenuineIntel
 *
 * Note that the string must be malloc-ed before return.
 */
char * getVendorID () {

    char * vendorID = (char *) malloc (sizeof(char) * (12 + 1));
    cpuid_t info = cpuid(0);

    memcpy(vendorID + 0, &(info.ebx), sizeof(char) * 4);
    memcpy(vendorID + 4, &(info.edx), sizeof(char) * 4);
    memcpy(vendorID + 8, &(info.ecx), sizeof(char) * 4);

    vendorID[12] = '\0';
    return vendorID;
}

/**
 *  The brand name returns the string of the computer mode.
 *  For example: Intel(R) Core(TM) i7-3720QM CPU @ 2.60GHz
 *
 *  Note that the string must be malloc-ed before return.
 */
char * getBrandName () {

    char * brandName = (char *) malloc (sizeof(char) * 48);

    for (uint32_t i = 0; i <= 2; i += 1) {

        cpuid_t info = cpuid(i + 0x80000002);
        char * str = brandName + i * 16;

        memcpy(str +  0, &(info.eax), sizeof(char) * 4);
        memcpy(str +  4, &(info.ebx), sizeof(char) * 4);
        memcpy(str +  8, &(info.ecx), sizeof(char) * 4);
        memcpy(str + 12, &(info.edx), sizeof(char) * 4);
    }

    return brandName;
}

void print_compiler_and_system_info()
{
    printf("===============================================================\n");
    printf("= Compiler & System info\n");
    printf("===============================================================\n");

    char * cpu_brand = getBrandName ();
    printf("Current CPU          : %s\n", cpu_brand);
    free(cpu_brand);

#ifdef CMAKE_C_COMPILER_ID
    printf("C Compiler ID        : %s\n", STRINGIZE_VALUE_OF(CMAKE_C_COMPILER_ID));
#endif

#ifdef CMAKE_CXX_COMPILER_ID
    printf("CXX Compiler ID      : %s\n", STRINGIZE_VALUE_OF(CMAKE_CXX_COMPILER_ID));
#endif

#ifdef CMAKE_C_COMPILER
    printf("C Compiler Path      : %s\n", STRINGIZE_VALUE_OF(CMAKE_C_COMPILER));
#endif

#ifdef CMAKE_CXX_COMPILER
    printf("CXX Compiler Path    : %s\n", STRINGIZE_VALUE_OF(CMAKE_CXX_COMPILER));
#endif

#ifdef CMAKE_C_COMPILER_VERSION
    printf("C Compiler Version   : %s\n", STRINGIZE_VALUE_OF(CMAKE_C_COMPILER_VERSION));
#endif

#ifdef CMAKE_CXX_COMPILER_VERSION
    printf("CXX Compiler Version : %s\n", STRINGIZE_VALUE_OF(CMAKE_CXX_COMPILER_VERSION));
#endif

#ifdef CMAKE_C_FLAGS
    printf("C Compiler Flags     : %s\n", STRINGIZE_VALUE_OF(CMAKE_C_FLAGS));
#endif

#ifdef CMAKE_CXX_FLAGS
    printf("CXX Compiler Flags   : %s\n", STRINGIZE_VALUE_OF(CMAKE_CXX_FLAGS));
#endif

#if defined(_OPENMP)
    printf("OpenMP support       : ENABLED\n");
#else
    printf("OpenMP support       : DISABLED\n");
#endif

    printf("OS page size         : %zu bytes\n", get_system_pagesize());
    printf("\n");
}