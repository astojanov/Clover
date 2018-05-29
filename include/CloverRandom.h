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

#ifndef CLOVER_RANDOM_H
#define CLOVER_RANDOM_H

#include <cstdint>
#include <immintrin.h>
#include "simdxorshift128plus.h"
#include "CloverBase.h"

class CloverRandom : public CloverBase {

protected:
    //
    // Random keys for XORshift algorithm
    //
    __m256i random_key1, random_key2;
    __m256i * random_key1_perthread;
    __m256i * random_key2_perthread;

public:

    //
    // If the __RDRND__ is available during compilation
    // use the hardware defined function to obtain a
    // random number
    //
    static inline uint64_t get_random_uint64 ()
    {
        unsigned long long rnd1;
        int ret = 0;
        while (ret == 0) {
            ret = _rdrand64_step(&rnd1);
        }
        return (uint64_t) rnd1;
    }

    //
    // If the __RDRND__ is available during compilation
    // use the hardware defined function to obtain a
    // random number
    //
    static inline int32_t get_random_int32 ()
    {
        unsigned int rnd1;

        int ret = 0;
        while (ret == 0) {
            ret = _rdrand32_step(&rnd1);
        }
        uint32_t rnd0 = (uint32_t) rnd1;
        return * (int32_t *) &rnd0;
    }


    static inline float get_random_float ()
    {
        unsigned int i_rnd;
        int ret = 0;
        while (ret == 0) {
            ret = _rdrand32_step(&i_rnd);
        }
        const float f_rnd = (float) i_rnd;
        return f_rnd * (1.0f / 4294967296.0f);
    }


    void setRandomKeys(__m256i key1, __m256i key2)
    {
        random_key1 = key1;
        random_key2 = key2;
    }

    CloverRandom ()
    {
        //
        // Initialize the random numbers per vector, using the hardware random number generator
        //
        avx_xorshift128plus_init(get_random_uint64(), get_random_uint64(), random_key1, random_key2);
        //
        // Allocate sufficient keys for each thread
        //
        const int n = get_OpenMP_threads();
        random_key1_perthread = new __m256i[n];
        random_key2_perthread = new __m256i[n];
        //
        // Then initialize random keys that could be used per thread.
        //
        for (int i = 0; i < n; i++) {
            avx_xorshift128plus_init(get_random_uint64(), get_random_uint64(), random_key1_perthread[i], random_key2_perthread[i]);
        }
    }

    ~CloverRandom()
    {
        delete [] random_key1_perthread;
        delete [] random_key2_perthread;
    }
};


#endif /* CLOVER_RANDOM_H */
