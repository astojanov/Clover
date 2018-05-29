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

#ifndef CLOVER_VECTOR_H
#define CLOVER_VECTOR_H

#include <cstdint>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>
#include <immintrin.h>

#include "simdxorshift128plus.h"
#include "CloverRandom.h"
#include "CloverBase.h"

#define CLOVER_VECTOR_BLOCK    64
#define CLOVER_VECTOR_SIZE_PAD (CLOVER_VECTOR_BLOCK * 2)

class CloverVector : public CloverRandom {

private:
    //
    // Memory space for the min_heaps
    //
    idx_t    * min_heaps_space;
    uint64_t   min_heaps_bytes;

protected:
    //
    // The initial length of the vector
    //
    const uint64_t length;
    //
    // The length + pad of the vector. Length is always padded with
    // a number defined by CLOVER_VECTOR_SIZE_PAD
    //
    const uint64_t length_pad;

    //
    // Memory management for the min-heaps
    //
    inline idx_t * get_min_heaps_mem (uint64_t size)
    {
        uint64_t bytes = size * sizeof(idx_t);
        if (min_heaps_bytes < bytes) {
            if (min_heaps_bytes != 0) {
                free(min_heaps_space);
            }
            min_heaps_space = (idx_t *) malloc(bytes);
            if (min_heaps_space == NULL) {
                std::cout << "We ran out of memory, while allocating thresholding memory. Exiting ..." << std::endl;
                exit(1);
            }
            min_heaps_bytes = bytes;
        }
        return min_heaps_space;
    }

public:

    CloverVector (uint64_t s) :
        length(s),
        length_pad(s % CLOVER_VECTOR_SIZE_PAD ? s + CLOVER_VECTOR_SIZE_PAD - (s % CLOVER_VECTOR_SIZE_PAD) : s )
    {
        min_heaps_space = NULL;
        min_heaps_bytes = 0;
    }

    inline uint64_t size () const
    {
        return length;
    }

    inline uint64_t size_pad () const
    {
        return length_pad;
    }

    virtual inline uint64_t    getBitsLength () const = 0;
    virtual inline uint64_t    getBytes      () const = 0;
    virtual inline std::string toString      () const = 0;

    ~ CloverVector ()
    {
        if (min_heaps_bytes != 0) {
            delete [] min_heaps_space;
        }
    }
};


#endif /* CLOVER_VECTOR_H */

