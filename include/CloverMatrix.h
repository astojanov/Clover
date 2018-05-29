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

#ifndef CLOVER_MATRIX_H
#define CLOVER_MATRIX_H

#include "CloverBase.h"
#include "CloverRandom.h"
#include "CloverVector.h"

/**
 * Matrix size is defined as M (rows) x N (cols). Size must be multiple
 * of CLOVER_VECTOR_SIZE_PAD. Each matrix contains its own random keys
 * using AVX XOR shift algorithm.
 */
class CloverMatrix : public CloverRandom {

protected:
    //
    // The length of the matrix
    //
    const uint64_t rows; // M
    const uint64_t cols; // N

public:

    CloverMatrix (uint64_t h, uint64_t w) :
            rows(h % CLOVER_VECTOR_SIZE_PAD ? h + CLOVER_VECTOR_SIZE_PAD - (h % CLOVER_VECTOR_SIZE_PAD) : h),
            cols(w % CLOVER_VECTOR_SIZE_PAD ? w + CLOVER_VECTOR_SIZE_PAD - (w % CLOVER_VECTOR_SIZE_PAD) : w)
    {
        // Do nothing
    }

    uint64_t getRows() const
    {
        return rows;
    }

    uint64_t getCols() const
    {
        return cols;
    }

    uint64_t size () const
    {
        return cols * rows;
    }

    virtual inline uint64_t    getBytes () const = 0;
    virtual inline std::string toString () const = 0;
};

#endif /* CLOVER_MATRIX_H */
