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

#include <immintrin.h>
#include <cassert>
#include <simdxorshift128plus.h>
#include "01_math.h"

//
// Create a random array
//
void create_array_of_random_values(float * arr, uint64_t n, __m256i &key1, __m256i &key2)
{

    assert(n % 128 == 0);

    const uint32_t clover_1st_bit_off_32       = 0x7FFFFFFFU;
    const __m256   clover_mm256_1st_bit_off_ps = (__m256) _mm256_set1_epi32 (clover_1st_bit_off_32);
    const __m256   clover_mm256_rcp_2pow31_ps  = _mm256_set1_ps(1.0f / 2147483648.0f);

    for (uint64_t i = 0; i < n; i += 8)
    {
        const __m256i rnd_0 = avx_xorshift128plus(key1, key2);
        const __m256i rnd_1 = _mm256_and_si256 (rnd_0, (__m256i) clover_mm256_1st_bit_off_ps);
        const __m256  rnd_2 = _mm256_cvtepi32_ps(rnd_1);
        const __m256  rnd_3 = _mm256_mul_ps(rnd_2, clover_mm256_rcp_2pow31_ps);

        _mm256_store_ps(arr + i, rnd_3);
    }
}
