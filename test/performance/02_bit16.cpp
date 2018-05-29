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


#include <CloverVector16.h>
#include <CloverMatrix16.h>
#include "02_bit16.h"

// ===============================================================================================================
// = Vector operations
// ===============================================================================================================

measurement_t measure_vector_quantize_16(uint64_t size)
{
    return measure_vector_quantize<CloverVector16>(size);
}

measurement_t measure_vector_quantize_parallel_16(uint64_t size)
{
    return measure_vector_quantize_parallel<CloverVector16>(size);
}

measurement_t measure_vector_get_16(uint64_t size)
{
    return measure_vector_get<CloverVector16>(size);
}

measurement_t measure_vector_dot_16(uint64_t size)
{
    return measure_vector_dot<CloverVector16>(size);
}

measurement_t measure_vector_dot_parallel_16(uint64_t size)
{
    return measure_vector_dot_parallel<CloverVector16>(size);
}

measurement_t measure_vector_scaleandadd_16(uint64_t size)
{
    return measure_vector_scaleandadd<CloverVector16>(size);
};

measurement_t measure_vector_scaleandadd_parallel_16(uint64_t size)
{
    return measure_vector_scaleandadd_parallel<CloverVector16>(size);
};

measurement_t measure_vector_threshold_16(uint64_t size, uint64_t k)
{
    return measure_vector_threshold<CloverVector16>(size, k);
}

measurement_t measure_vector_threshold_parallel_16(uint64_t size, uint64_t k)
{
    return measure_vector_threshold_parallel<CloverVector16>(size, k);
}

// ===============================================================================================================
// = Matrix operations
// ===============================================================================================================

measurement_t measure_matrix_quantize_16(uint64_t size)
{
    return measure_matrix_quantize<CloverMatrix16>(size);
}

measurement_t measure_matrix_transpose_16(uint64_t size)
{
    return measure_matrix_transpose<CloverMatrix16>(size);
}

measurement_t measure_matrix_transpose_parallel_16(uint64_t size)
{
    return measure_matrix_transpose_parallel<CloverMatrix16>(size);
}

measurement_t measure_matrix_MVM_16(uint64_t size)
{
    return measure_matrix_MVM<CloverMatrix16, CloverVector16>(size);
}

measurement_t measure_matrix_MVM_parallel_16(uint64_t size)
{
    return measure_matrix_MVM_parallel<CloverMatrix16, CloverVector16>(size);
}

// ===============================================================================================================
// = Quantized Linear Algebra Application
// ===============================================================================================================

measurement_t measure_IHT_or_GD_16(
        problem_type_t problem_type,
        CloverVector32 &x_32, CloverMatrix32 &Phi_32, CloverVector32 &y_32, uint64_t &K, float &mu,
        uint64_t &iterations
) {
    return measure_IHT_or_GD<CloverMatrix16, CloverVector16>(problem_type, x_32, Phi_32, y_32, K, mu,
                                                             iterations);
}
