/**
 *      _________   _____________________  ____  ______
 *     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
 *    / /_  / /| | \__ \ / / / /   / / / / / / / __/
 *   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
 *  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
 *
 *  http://www.inf.ethz.ch/personal/markusp/teaching/
 *  How to Write Fast Numerical Code 263-2300 - ETH Zurich
 *  Copyright (C) 2016  Alen Stojanov      (astojanov@inf.ethz.ch)
 *                      Daniele Spampinato (daniele.spampinato@inf.ethz.ch)
 *                      Singh Gagandeep    (gsingh@inf.ethz.ch)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include "cpuid.h"
#include <immintrin.h>

// Define interface to cpuid instruction.
// input:  eax = function number, ecx = 0
// output: eax = output[0], ebx = output[1], ecx = output[2], edx = output[3]

cpuid_t cpuid (uint32_t functionnumber) {

    cpuid_t info;
    int output[4];

#if defined (_MSC_VER)

    // Microsoft intrinsic function for CPUID
    __cpuidex(output, functionnumber, 0);

#elif defined(__GNUC__) || defined(__clang__) || defined (__INTEL_COMPILER)

    // use inline assembly, Gnu/AT&T syntax
   int a, b, c, d;
   __asm("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "a"(functionnumber),"c"(0) : );
   output[0] = a;
   output[1] = b;
   output[2] = c;
   output[3] = d;

#else
    // unknown platform. try inline assembly with masm/intel syntax
    __asm {
        mov eax, functionnumber
        xor ecx, ecx
        cpuid;
        mov esi, output
        mov [esi],    eax
        mov [esi+4],  ebx
        mov [esi+8],  ecx
        mov [esi+12], edx
    }
#endif

    info.eax = output[0];
    info.ebx = output[1];
    info.ecx = output[2];
    info.edx = output[3];

    return info;
}

// Define interface to xgetbv instruction
int64_t xgetbv (int ctr) {

#if (defined (_MSC_FULL_VER) && _MSC_FULL_VER >= 160040000) || (defined (__INTEL_COMPILER) && __INTEL_COMPILER >= 1200)

    // Microsoft or Intel compiler supporting _xgetbv intrinsic
    // intrinsic function for XGETBV
    return _xgetbv(ctr);

#elif defined(__GNUC__)

    // use inline assembly, Gnu/AT&T syntax
   uint32_t a, d;
   __asm("xgetbv" : "=a"(a),"=d"(d) : "c"(ctr) : );
   return a | (((uint64_t)d) << 32);

#else  // #elif defined (_WIN32)

    // other compiler. try inline assembly with masm/intel/MS syntax
   uint32_t a, d;
    __asm {
        mov ecx, ctr
        _emit 0x0f
        _emit 0x01
        _emit 0xd0 ; // xgetbv
        mov a, eax
        mov d, edx
    }
   return a | (uint64_t(d) << 32);

#endif
}

