/**
 *  Clover: Quantized 4-bit Linear Algebra Library
 *              ______ __
 *             / ____// /____  _   __ ___   _____
 *            / /    / // __ \| | / // _ \ / ___/
 *           / /___ / // /_/ /| |/ //  __// /
 *           \____//_/ \____/ |___/ \___//_/
 *
 *  Copyright 2018 Alen Stojanov       (astojanov@inf.ethz.ch)
 *                 Tyler Michael Smith (tyler.smith@inf.ethz.ch)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CLOVER_BASE_H
#define CLOVER_BASE_H

#include <iostream>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <string>
#include <sstream>
#include <iomanip>
#include "ipp.h"
#include "mkl.h"
#include "../lib/simd_debug.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__APPLE__)
    #include <mach/machine.h>
#elif defined(__linux__) || defined(linux) || defined(__linux)
    #include <unistd.h>
#endif

class CloverBase {

protected:
    //
    // Scalar constants
    //
    const uint32_t clover_1st_bit_set_32  = 0x80000000U;
    const uint32_t clover_1st_bit_off_32  = 0x7FFFFFFFU;

    //
    // Setup SSE constraints
    //
    const __m128  clover_mm_1st_bit_off_ps   = (__m128) _mm_set1_epi32 (clover_1st_bit_off_32);

    #if defined(__AVX__)

        //
        // Setup AVX constants for fast evaluation of an `abs` functions
        //
        const __m256i clover_mm256_1st_bit_off_epi8 = _mm256_set1_epi32 (0x7F7F7F7FU);
        const __m256i clover_mm256_1st_bit_set_epi8 = _mm256_set1_epi8  (-16);
        const __m256  clover_mm256_1st_bit_set_ps   = (__m256) _mm256_set1_epi32 (clover_1st_bit_set_32);
        const __m256  clover_mm256_1st_bit_off_ps   = (__m256) _mm256_set1_epi32 (clover_1st_bit_off_32);

        //
        // Setup AVX constant that represents mask for the first 32-bit chunk
        // of a 256-bit AVX lane
        //
        const __m256i clover_mm256_mask_1st_epi32    = _mm256_setr_epi32(0xFFFFFFFFU, 0, 0, 0, 0, 0, 0, 0);

        //
        // Various other constants
        //
        const __m256i clover_mm256_1_epi16           = _mm256_set1_epi16(1);
        const __m256  clover_mm256_1_ps              = _mm256_set1_ps(1.0f);
        const __m256  clover_mm256_7_ps              = _mm256_set1_ps(7.0f);
        const __m256  clover_mm256_127_ps            = _mm256_set1_ps(127.0f);
        const __m256  clover_mm256_rcp_7_ps          = _mm256_set1_ps(1.0f / 7.0f);
        const __m256  clover_mm256_rcp_127_ps        = _mm256_set1_ps(1.0f / 127.0f);
        const __m256  clover_mm256_rcp_49_ps         = _mm256_set1_ps(1.0f / 49.0f);
        const __m256  clover_mm256_rcp_2pow31_ps     = _mm256_set1_ps(1.0f / 2147483648.0f);

        const __m256i clover_mm256_8bit_perm_lo = _mm256_setr_epi8 (
                0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15,
                0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15
        );
        const __m256i clover_mm256_8bit_perm_hi = _mm256_setr_epi8 (
                2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13,
                2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13
        );

        const __m256i clover_mm256_8bit_restore_perm_lo = _mm256_setr_epi8(
                0, 8, -128, -128, 1, 9, -128, -128, 2, 10, -128, -128, 3, 11, -128, -128,
                -128, -128, 4, 12, -128, -128, 5, 13, -128, -128, 6, 14, -128, -128, 7, 15
        );
        const __m256i clover_mm256_8bit_restore_perm_hi = _mm256_setr_epi8 (
                -128, -128, 0, 8, -128, -128, 1, 9, -128, -128, 2, 10, -128, -128, 3, 11,
                4, 12, -128, -128, 5, 13, -128, -128, 6, 14, -128, -128, 7, 15, -128, -128
        );

        //
        // Calculate the horizontal max in a given AVX vector
        //
        static inline float _mm256_hmaxf32_ps(const __m256 tmp3)
        {
            const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
            const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
            const __m128 tmp6 = _mm_max_ps(tmp4, tmp5);
            const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
            const __m128 tmp8 = _mm_max_ps(tmp6, tmp7);
            const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
            const __m128 tmp0 = _mm_max_ps(tmp8, tmp9);
            //
            // Return the result stored in the first element
            //
            return _mm_cvtss_f32(tmp0);
        }

        //
        // Calculate the horizontal min in a given AVX vector
        //
        static inline float _mm256_hminf32_ps(const __m256 tmp3)
        {
            const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
            const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
            const __m128 tmp6 = _mm_min_ps(tmp4, tmp5);
            const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
            const __m128 tmp8 = _mm_min_ps(tmp6, tmp7);
            const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
            const __m128 tmp0 = _mm_min_ps(tmp8, tmp9);
            //
            // Return the result stored in the first element
            //
            return _mm_cvtss_f32(tmp0);
        }


        //
        // For a given vector __m256 of 8 floats, perform reduction
        //
        static inline float _mm256_haddf32_ps(__m256 acc)
        {
            const __m128 left  = _mm256_extractf128_ps(acc, 1);
            const __m128 right = _mm256_castps256_ps128(acc);
            const __m128 x128  = _mm_add_ps(left, right);
            const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
            const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
            return  _mm_cvtss_f32(x32);
        }

        //
        // Transpose 8x8 registers
        //
        static inline void _mm256_transpose8_epi32(
                __m256i &r0, __m256i &r1, __m256i &r2, __m256i &r3,
                __m256i &r4, __m256i &r5, __m256i &r6, __m256i &r7
        ){
            __m256 u0, u1, u2, u3, u4, u5, u6, u7;
            __m256 s0, s1, s2, s3, s4, s5, s6, s7;

            u0 = (__m256) _mm256_unpacklo_epi32(r0, r1);
            u1 = (__m256) _mm256_unpackhi_epi32(r0, r1);
            u2 = (__m256) _mm256_unpacklo_epi32(r2, r3);
            u3 = (__m256) _mm256_unpackhi_epi32(r2, r3);
            u4 = (__m256) _mm256_unpacklo_epi32(r4, r5);
            u5 = (__m256) _mm256_unpackhi_epi32(r4, r5);
            u6 = (__m256) _mm256_unpacklo_epi32(r6, r7);
            u7 = (__m256) _mm256_unpackhi_epi32(r6, r7);

            s0 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(1,0,1,0));
            s1 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(3,2,3,2));
            s2 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(1,0,1,0));
            s3 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(3,2,3,2));
            s4 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(1,0,1,0));
            s5 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(3,2,3,2));
            s6 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(1,0,1,0));
            s7 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(3,2,3,2));

            r0 = (__m256i) _mm256_permute2f128_ps(s0, s4, 0x20);
            r1 = (__m256i) _mm256_permute2f128_ps(s1, s5, 0x20);
            r2 = (__m256i) _mm256_permute2f128_ps(s2, s6, 0x20);
            r3 = (__m256i) _mm256_permute2f128_ps(s3, s7, 0x20);
            r4 = (__m256i) _mm256_permute2f128_ps(s0, s4, 0x31);
            r5 = (__m256i) _mm256_permute2f128_ps(s1, s5, 0x31);
            r6 = (__m256i) _mm256_permute2f128_ps(s2, s6, 0x31);
            r7 = (__m256i) _mm256_permute2f128_ps(s3, s7, 0x31);
        }

    #endif

    //
    // Structure that holds values and indexes in the matrix / vectors
    //
    union Restorator 
    {
        int32_t i;
        float f;
    };

    typedef struct {
        float  value;
        Restorator bits;
        uint64_t idx;
    } idx_t;
    //
    // Comparison function used for building min-heap
    //
    inline static bool gt_idx_t(const idx_t &a, const idx_t &b) {
        return (a.value > b.value) || std::isnan(a.value);
    }
    inline static bool lt_idx_t(const idx_t &a, const idx_t &b) {
        return a.value < b.value;
    }
    //
    // For a given heap of type idx_t, call "heapify" at a given position
    // pos and heap-length of size k.
    //
    inline static void min_heapify(idx_t * heap, uint32_t pos, const uint32_t k)
    {
        uint32_t smallest = pos;

        while (true) {
            const uint32_t l = pos * 2 + 1;
            const uint32_t r = pos * 2 + 2;

            if (l < k && heap[l].value < heap[smallest].value) {
                smallest = l;
            }
            if (r < k && heap[r].value < heap[smallest].value) {
                smallest = r;
            }
            if (smallest != pos) {
                idx_t tmp = heap[pos];
                heap[pos] = heap[smallest];
                heap[smallest] = tmp;
                pos = smallest;
            } else {
                break;
            }
        }
    }

    static void initializeIPP ()
    {
        std::cout << "======================================================================" << std::endl;
        std::cout << " = Initializing Intel IPP Library" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        ippSetNumThreads(get_OpenMP_threads());
        std::cout << "Intel IPP initialized with " << get_OpenMP_threads() << " threads." << std::endl;
        const IppLibraryVersion *lib;
        IppStatus status;
        Ipp64u mask, emask;

        /* Init IPP library */
        ippInit();
        /* Get IPP library version info */
        lib = ippGetLibVersion();
        printf("%s %s\n", lib->Name, lib->Version);

        /* Get CPU features and features enabled with selected library level */
        status = ippGetCpuFeatures( &mask, 0 );
        if( ippStsNoErr == status ) {
            emask = ippGetEnabledCpuFeatures();
            printf("Features supported by CPU\tby IPP\n");
            printf("-----------------------------------------\n");
            printf("  ippCPUID_MMX        = ");
            printf("%c\t%c\t",( mask & ippCPUID_MMX ) ? 'Y':'N',( emask & ippCPUID_MMX ) ? 'Y':'N');
            printf("Intel(R) Architecture MMX technology supported\n");
            printf("  ippCPUID_SSE        = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSE ) ? 'Y':'N',( emask & ippCPUID_SSE ) ? 'Y':'N');
            printf("Intel(R) Streaming SIMD Extensions\n");
            printf("  ippCPUID_SSE2       = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSE2 ) ? 'Y':'N',( emask & ippCPUID_SSE2 ) ? 'Y':'N');
            printf("Intel(R) Streaming SIMD Extensions 2\n");
            printf("  ippCPUID_SSE3       = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSE3 ) ? 'Y':'N',( emask & ippCPUID_SSE3 ) ? 'Y':'N');
            printf("Intel(R) Streaming SIMD Extensions 3\n");
            printf("  ippCPUID_SSSE3      = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSSE3 ) ? 'Y':'N',( emask & ippCPUID_SSSE3 ) ? 'Y':'N');
            printf("Intel(R) Supplemental Streaming SIMD Extensions 3\n");
            printf("  ippCPUID_MOVBE      = ");
            printf("%c\t%c\t",( mask & ippCPUID_MOVBE ) ? 'Y':'N',( emask & ippCPUID_MOVBE ) ? 'Y':'N');
            printf("The processor supports MOVBE instruction\n");
            printf("  ippCPUID_SSE41      = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSE41 ) ? 'Y':'N',( emask & ippCPUID_SSE41 ) ? 'Y':'N');
            printf("Intel(R) Streaming SIMD Extensions 4.1\n");
            printf("  ippCPUID_SSE42      = ");
            printf("%c\t%c\t",( mask & ippCPUID_SSE42 ) ? 'Y':'N',( emask & ippCPUID_SSE42 ) ? 'Y':'N');
            printf("Intel(R) Streaming SIMD Extensions 4.2\n");
            printf("  ippCPUID_AVX        = ");
            printf("%c\t%c\t",( mask & ippCPUID_AVX ) ? 'Y':'N',( emask & ippCPUID_AVX ) ? 'Y':'N');
            printf("Intel(R) Advanced Vector Extensions instruction set\n");
            printf("  ippAVX_ENABLEDBYOS  = ");
            printf("%c\t%c\t",( mask & ippAVX_ENABLEDBYOS ) ? 'Y':'N',( emask & ippAVX_ENABLEDBYOS ) ? 'Y':'N');
            printf("The operating system supports Intel(R) AVX\n");
            printf("  ippCPUID_AES        = ");
            printf("%c\t%c\t",( mask & ippCPUID_AES ) ? 'Y':'N',( emask & ippCPUID_AES ) ? 'Y':'N');
            printf("Intel(R) AES instruction\n");
            printf("  ippCPUID_SHA        = ");
            printf("%c\t%c\t",( mask & ippCPUID_SHA ) ? 'Y':'N',( emask & ippCPUID_SHA ) ? 'Y':'N');
            printf("Intel(R) SHA new instructions\n");
            printf("  ippCPUID_CLMUL      = ");
            printf("%c\t%c\t",( mask & ippCPUID_CLMUL ) ? 'Y':'N',( emask & ippCPUID_CLMUL ) ? 'Y':'N');
            printf("PCLMULQDQ instruction\n");
            printf("  ippCPUID_RDRAND     = ");
            printf("%c\t%c\t",( mask & ippCPUID_RDRAND ) ? 'Y':'N',( emask & ippCPUID_RDRAND ) ? 'Y':'N');
            printf("Read Random Number instructions\n");
            printf("  ippCPUID_F16C       = ");
            printf("%c\t%c\t",( mask & ippCPUID_F16C ) ? 'Y':'N',( emask & ippCPUID_F16C ) ? 'Y':'N');
            printf("Float16 instructions\n");
            printf("  ippCPUID_AVX2       = ");
            printf("%c\t%c\t",( mask & ippCPUID_AVX2 ) ? 'Y':'N',( emask & ippCPUID_AVX2 ) ? 'Y':'N');
            printf("Intel(R) Advanced Vector Extensions 2 instruction set\n");
            printf("  ippCPUID_AVX512F    = ");
            printf("%c\t%c\t",( mask & ippCPUID_AVX512F ) ? 'Y':'N',( emask & ippCPUID_AVX512F ) ? 'Y':'N');
            printf("Intel(R) Advanced Vector Extensions 3.1 instruction set\n");
            printf("  ippCPUID_AVX512CD   = ");
            printf("%c\t%c\t",( mask & ippCPUID_AVX512CD ) ? 'Y':'N',( emask & ippCPUID_AVX512CD ) ? 'Y':'N');
            printf("Intel(R) Advanced Vector Extensions CD (Conflict Detection) instruction set\n");
            printf("  ippCPUID_AVX512ER   = ");
            printf("%c\t%c\t",( mask & ippCPUID_AVX512ER ) ? 'Y':'N',( emask & ippCPUID_AVX512ER ) ? 'Y':'N');
            printf("Intel(R) Advanced Vector Extensions ER instruction set\n");
            printf("  ippCPUID_ADCOX      = ");
            printf("%c\t%c\t",( mask & ippCPUID_ADCOX ) ? 'Y':'N',( emask & ippCPUID_ADCOX ) ? 'Y':'N');
            printf("ADCX and ADOX instructions\n");
            printf("  ippCPUID_RDSEED     = ");
            printf("%c\t%c\t",( mask & ippCPUID_RDSEED ) ? 'Y':'N',( emask & ippCPUID_RDSEED ) ? 'Y':'N');
            printf("The RDSEED instruction\n");
            printf("  ippCPUID_PREFETCHW  = ");
            printf("%c\t%c\t",( mask & ippCPUID_PREFETCHW ) ? 'Y':'N',( emask & ippCPUID_PREFETCHW ) ? 'Y':'N');
            printf("The PREFETCHW instruction\n");
            printf("  ippCPUID_KNC        = ");
            printf("%c\t%c\t",( mask & ippCPUID_KNC ) ? 'Y':'N',( emask & ippCPUID_KNC ) ? 'Y':'N');
            printf("Intel(R) Xeon Phi(TM) Coprocessor instruction set\n");
        } else {
            std::cout << "Failed! Exiting ...." << std::endl;
            exit(1);
        }

        std::cout << "======================================================================" << std::endl << std::endl;
    }

    static void initializeMKL () {
        std::cout << "======================================================================" << std::endl;
        std::cout << " = Initializing Intel MKL Library" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        char buf[1024];
        mkl_get_version_string(buf, sizeof(buf) / sizeof(char));
        printf("%s\n",buf);
        printf("\n");
        int nthr = get_OpenMP_threads();
        printf("Setting the number of MKL threads to: %d\n", nthr);
        mkl_set_num_threads(nthr);
        std::cout << "======================================================================" << std::endl;
    }



public:

    static inline int get_OpenMP_threads()
    {
        static int nthr = -1;
        if ( nthr == -1 ) {
            #if defined(_OPENMP)
                _Pragma("omp parallel") { nthr = omp_get_num_threads(); }
            #else
                nthr = 1;
            #endif
        }
        return nthr;
    }

    static void initializeLibraries ()
    {
        static bool ippInitialized = false;
        if (!ippInitialized) {
            initializeIPP();
            ippInitialized = true;
        }
        static bool mklInitialized = false;
        if (!mklInitialized) {
            initializeMKL();
            mklInitialized = true;
        }
    }

    CloverBase ()
    {
        initializeLibraries();
    }

    //
    // Shift right with sign extends (its in a way scalar, but uses SSE to achieve this)
    //
    static inline int8_t _mm_srai_epi8_ss(int8_t x, int inm8)
    {
        #if defined(__SSE4_1__)
            const __m128i x0 = _mm_set1_epi8(x);
            const __m128i x1 = _mm_slli_epi16(x0, 8);
            const __m128i x2 = _mm_srai_epi16(x1, 8 + inm8);
            return (int8_t) _mm_extract_epi8(x2, 0);
        #else
            return x >> inm8;
        #endif
    }

    //
    // Gets system page size
    //
    static inline size_t get_system_pagesize ()
    {
        #if defined(__APPLE__)
            return PAGE_SIZE;
        #elif defined(__linux__) || defined(linux) || defined(__linux)
            return (size_t) sysconf(_SC_PAGESIZE);
        #endif
    }


    //
    // Perform horizontal reduction, and make sure that the max is broadcasted in
    // all slots of the 256 bit lane
    //
    static inline __m256 _mm256_hmax_ps(const __m256 &hmax_0)
    {
        const __m256 hmax_1 = _mm256_permute2f128_ps(hmax_0, hmax_0, 3);
        const __m256 hmax_2 = _mm256_max_ps(hmax_0, hmax_1);
        const __m256 hmax_3 = _mm256_permute_ps(hmax_2, 0x4E);
        const __m256 hmax_4 = _mm256_max_ps(hmax_2, hmax_3);
        const __m256 hmax_5 = _mm256_permute_ps(hmax_4, 0xB1);
        const __m256 hmax_6 = _mm256_max_ps(hmax_4, hmax_5);
        return hmax_6;
    }

    //
    // Get horizontal max of 16 x 16-bit signed numbers, and make sure
    // that the max is broadcasted across the 256-bit AVX lane
    //
    static inline __m256i _mm256_hmax_epi16(const __m256i &x)
    {
        const __m256i shuffle = _mm256_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
                                                 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
        //
        // Exchange neighbouring 128-bit chunks
        //
        const __m256i hmax_1 = (__m256i) _mm256_permute2f128_ps((__m256) x, (__m256) x, 3);
        const __m256i hmax_2 = _mm256_max_epi16(x, hmax_1);
        //
        // Exchange neighbouring 64-bit chunks
        //
        const __m256i hmax_3 = (__m256i) _mm256_permute_ps((__m256) hmax_2, 0x4E);
        const __m256i hmax_4 = _mm256_max_epi16(hmax_2, hmax_3);
        //
        // Exchange neighbouring 32-bit chunks
        //
        const __m256i hmax_5 = (__m256i) _mm256_permute_ps((__m256) hmax_4, 0xB1);;
        const __m256i hmax_6 = _mm256_max_epi16(hmax_4, hmax_5);
        //
        // Exchange neighbouring 16-bit chunks
        //
        const __m256i hmax_7 = _mm256_shuffle_epi8(hmax_6, shuffle);
        const __m256i hmax_8 = _mm256_max_epi16(hmax_6, hmax_7);
        //
        // Return the result
        //
        return hmax_8;
    }

    //
    // Calculates a * b + c using FMA instructions
    //
    static inline float _mm_fmadd_ss(const float &a, const float &b, const float &c)
    {
        const __m128 va = _mm_set1_ps(a);
        const __m128 vb = _mm_set1_ps(b);
        const __m128 vc = _mm_set1_ps(c);
        const __m128 result = _mm_fmadd_ps(va, vb, vc);
        return _mm_cvtss_f32(result);
    }

    //
    // Convert float into FP16C (half-precision float)
    //
    static inline uint16_t _mm_cvtss_sh(const float &value)
    {
        const __m128  value_ps = _mm_set1_ps(value);
        const __m128i f16c     = _mm_cvtps_ph(value_ps, 0);
        const uint16_t result  = (uint16_t)_mm_extract_epi16(f16c, 0);
        return result;
    }

    //
    // Convert FP16C (half-precision float) into float
    //
    static inline float _mm_cvtsh_ss(const uint16_t &value)
    {
        const __m128i f16c     = _mm_set1_epi16(value);
        const __m128  value_ps = _mm_cvtph_ps(f16c);
        const float result     = _mm_cvtss_f32(value_ps);
        return result;
    }

    //
    // Convert a floating point number into its hex string representation
    //
    static std::string float2hex(float f)
    {
        char buf[32];
        sprintf(buf, "0x%08x", *(uint32_t *)&f);
        return std::string(buf);
    }
};

#endif /* CLOVER_BASE_H */
