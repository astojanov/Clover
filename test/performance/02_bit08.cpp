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


#include <CloverVector8.h>
#include <CloverMatrix8.h>
#include "02_bit04.h"

// ===============================================================================================================
// = Vector operations
// ===============================================================================================================

measurement_t measure_vector_quantize_08(uint64_t size)
{
    return measure_vector_quantize<CloverVector8>(size);
}

measurement_t measure_vector_quantize_parallel_08(uint64_t size)
{
    return measure_vector_quantize_parallel<CloverVector8>(size);
}

measurement_t measure_vector_get_08(uint64_t size)
{
    return measure_vector_get<CloverVector8>(size);
}

measurement_t measure_vector_dot_08(uint64_t size)
{
    return measure_vector_dot<CloverVector8>(size);
}

measurement_t measure_vector_dot_parallel_08(uint64_t size)
{
    return measure_vector_dot_parallel<CloverVector8>(size);
}

measurement_t measure_vector_scaleandadd_08(uint64_t size)
{
    return measure_vector_scaleandadd<CloverVector8>(size);
}

measurement_t measure_vector_scaleandadd_parallel_08(uint64_t size)
{
    return measure_vector_scaleandadd_parallel<CloverVector8>(size);
}

measurement_t measure_vector_threshold_08(uint64_t size, uint64_t k)
{
    return measure_vector_threshold<CloverVector8>(size, k);
}

measurement_t measure_vector_threshold_parallel_08(uint64_t size, uint64_t k)
{
    return measure_vector_threshold_parallel<CloverVector8>(size, k);
}

// ===============================================================================================================
// = Matrix operations
// ===============================================================================================================

measurement_t measure_matrix_quantize_08(uint64_t size)
{
    return measure_matrix_quantize<CloverMatrix8>(size);
}

measurement_t measure_matrix_MVM_08(uint64_t size)
{
    return measure_matrix_MVM<CloverMatrix8, CloverVector8>(size);
}

measurement_t measure_matrix_transpose_08(uint64_t size)
{
    return measure_matrix_transpose<CloverMatrix8>(size);
}

measurement_t measure_matrix_transpose_parallel_08(uint64_t size)
{
    return measure_matrix_transpose_parallel<CloverMatrix8>(size);
}

measurement_t measure_matrix_MVM_parallel_08(uint64_t size)
{
    return measure_matrix_MVM_parallel<CloverMatrix8, CloverVector8>(size);
}

// ===============================================================================================================
// = Quantized Linear Algebra Application
// ===============================================================================================================

measurement_t measure_IHT_or_GD_08(
        problem_type_t problem_type,
        CloverVector32 &x_32, CloverMatrix32 &Phi_32, CloverVector32 &y_32, uint64_t &K, float &mu, uint64_t &iterations
) {
    return measure_IHT_or_GD<CloverMatrix8, CloverVector8>(
            problem_type, x_32, Phi_32, y_32, K, mu, iterations
    );
}
