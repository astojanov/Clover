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

#ifndef CLOVER_MATH_H
#define CLOVER_MATH_H

#include <cstdint>
#include <cmath>

//
// Calculate the norm2
//
template <class QVector>
double norm2(const QVector &v) {
    double result = 0;
    for (uint64_t i = 0; i < v.size(); i += 1) {
        result += (double) v.get(i) * (double) v.get(i);
    }
    return sqrt(result);
}

template <class QMatrix>
double norm2(const QMatrix &m, uint64_t row) {
    double result = 0.0;
    for (uint64_t j = 0; j < m.getCols(); j += 1) {
        result += (double) m.get(row, j) * (double) m.get(row, j);
    }
    return sqrt(result);
}

void create_array_of_random_values(float * arr, uint64_t n, __m256i &key1, __m256i &key2);


#endif /* CLOVER_MATH_H */
