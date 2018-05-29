#ifndef INSTRSET_H
#define INSTRSET_H 120

// Detect 64 bit mode
#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) ) && ! defined(__x86_64__)
#define __x86_64__ 1  // There are many different macros for this, decide on only one
#endif

// Find instruction set from compiler macros if INSTRSET not defined
// Note: Microsoft compilers do not define these macros automatically
#ifndef INSTRSET
#if defined ( __AVX512F__ ) || defined ( __AVX512__ ) // || defined ( __AVX512ER__ )
#define INSTRSET 9
#elif defined ( __AVX2__ )
#define INSTRSET 8
#elif defined ( __AVX__ )
#define INSTRSET 7
#elif defined ( __SSE4_2__ )
#define INSTRSET 6
#elif defined ( __SSE4_1__ )
#define INSTRSET 5
#elif defined ( __SSSE3__ )
#define INSTRSET 4
#elif defined ( __SSE3__ )
#define INSTRSET 3
#elif defined ( __SSE2__ ) || defined ( __x86_64__ )
#define INSTRSET 2
#elif defined ( __SSE__ )
#define INSTRSET 1
#elif defined ( _M_IX86_FP )           // Defined in MS compiler. 1: SSE, 2: SSE2
#define INSTRSET _M_IX86_FP
#else
#define INSTRSET 0
#endif // instruction set defines
#endif // INSTRSET

// Include the appropriate header file for intrinsic functions
#if INSTRSET > 7                       // AVX2 and later
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>                 // x86intrin.h includes header files for whatever instruction
                                       // sets are specified on the compiler command line, such as:
                                       // xopintrin.h, fma4intrin.h
#else
#include <immintrin.h>                 // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__
#elif INSTRSET == 7
#include <immintrin.h>                 // AVX
#elif INSTRSET == 6
#include <nmmintrin.h>                 // SSE4.2
#elif INSTRSET == 5
#include <smmintrin.h>                 // SSE4.1
#elif INSTRSET == 4
#include <tmmintrin.h>                 // SSSE3
#elif INSTRSET == 3
#include <pmmintrin.h>                 // SSE3
#elif INSTRSET == 2
#include <emmintrin.h>                 // SSE2
#elif INSTRSET == 1
#include <xmmintrin.h>                 // SSE
#endif // INSTRSET

#if INSTRSET >= 8 && !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif

// AMD  instruction sets
#if defined (__XOP__) || defined (__FMA4__)
#ifdef __GNUC__
#include <x86intrin.h>                 // AMD XOP (Gnu)
#else
#include <ammintrin.h>                 // AMD XOP (Microsoft)
#endif //  __GNUC__
#elif defined (__SSE4A__)              // AMD SSE4A
#include <ammintrin.h>
#endif // __XOP__

// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__))  && ! defined (__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__

// FMA4 instruction set
#if defined (__FMA4__) && (defined(__GNUC__) || defined(__clang__))
#include <fma4intrin.h> // must have both x86intrin.h and fma4intrin.h, don't know why
#endif // __FMA4__


// Define integer types with known size
#if defined(__GNUC__) || defined(__clang__) || (defined(_MSC_VER) && _MSC_VER >= 1600)
  // Compilers supporting C99 or C++0x have stdint.h defining these integer types
  #include <stdint.h>
#elif defined(_MSC_VER)
  // Older Microsoft compilers have their own definitions
  typedef signed   __int8  int8_t;
  typedef unsigned __int8  uint8_t;
  typedef signed   __int16 int16_t;
  typedef unsigned __int16 uint16_t;
  typedef signed   __int32 int32_t;
  typedef unsigned __int32 uint32_t;
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
  #ifndef _INTPTR_T_DEFINED
    #define _INTPTR_T_DEFINED
    #ifdef  __x86_64__
      typedef int64_t intptr_t;
    #else
      typedef int32_t intptr_t;
    #endif
  #endif
#else
  // This works with most compilers
  typedef signed   char      int8_t;
  typedef unsigned char      uint8_t;
  typedef signed   short int int16_t;
  typedef unsigned short int uint16_t;
  typedef signed   int       int32_t;
  typedef unsigned int       uint32_t;
  typedef long long          int64_t;
  typedef unsigned long long uint64_t;
  #ifdef  __x86_64__
    typedef int64_t intptr_t;
  #else
    typedef int32_t intptr_t;
  #endif
#endif

#include <stdlib.h>                              // define abs(int)
#include <stdbool.h>                             // define abs(int)

#ifdef _MSC_VER                                  // Microsoft compiler or compatible Intel compiler
#include <intrin.h>                              // define _BitScanReverse(int), __cpuid(int[4],int), _xgetbv(int)
#endif // _MSC_VER

// GCC version
#if defined(__GNUC__) && !defined (GCC_VERSION) && !defined (__clang__)
#define GCC_VERSION  ((__GNUC__) * 10000 + (__GNUC_MINOR__) * 100 + (__GNUC_PATCHLEVEL__))
#endif

// Clang version
#if defined (__clang__)
#define CLANG_VERSION  ((__clang_major__) * 10000 + (__clang_minor__) * 100 + (__clang_patchlevel__))
// Problem: The version number is not consistent across platforms
// http://llvm.org/bugs/show_bug.cgi?id=12643
// Apple bug 18746972
#endif

// Fix problem with macros named min and max in WinDef.h
#ifdef _MSC_VER
#if defined (_WINDEF_) && defined(min) && defined(max)
#undef min
#undef max
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

typedef struct {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
} cpuid_t;

cpuid_t cpuid (uint32_t functionnumber);
int64_t xgetbv (int ctr);

#endif // INSTRSET_H
