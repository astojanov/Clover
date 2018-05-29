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

#include <CloverVector4.h>
#include <CloverMatrix4.h>
#include "02_bit04.h"

// ===============================================================================================================
// = Vector operations
// ===============================================================================================================

measurement_t measure_vector_quantize_04(uint64_t size)
{
    return measure_vector_quantize<CloverVector4>(size);
}

measurement_t measure_vector_quantize_parallel_04(uint64_t size)
{
    return measure_vector_quantize_parallel<CloverVector4>(size);
}

measurement_t measure_vector_get_04(uint64_t size)
{
    return measure_vector_get<CloverVector4>(size);
}

measurement_t measure_vector_dot_04(uint64_t size)
{
    return measure_vector_dot<CloverVector4>(size);
}

measurement_t measure_vector_dot_parallel_04(uint64_t size)
{
    return measure_vector_dot_parallel<CloverVector4>(size);
}

measurement_t measure_vector_scaleandadd_04(uint64_t size)
{
    return measure_vector_scaleandadd<CloverVector4>(size);
};

measurement_t measure_vector_scaleandadd_parallel_04(uint64_t size)
{
    return measure_vector_scaleandadd_parallel<CloverVector4>(size);
};

measurement_t measure_vector_threshold_04(uint64_t size, uint64_t k)
{
    return measure_vector_threshold<CloverVector4>(size, k);
}

measurement_t measure_vector_threshold_parallel_04(uint64_t size, uint64_t k)
{
    return measure_vector_threshold_parallel<CloverVector4>(size, k);
}


// ===============================================================================================================
// = Matrix operations
// ===============================================================================================================

measurement_t measure_matrix_quantize_04(uint64_t size)
{
    return measure_matrix_quantize<CloverMatrix4>(size);
}

measurement_t measure_matrix_MVM_04(uint64_t size)
{
    return measure_matrix_MVM<CloverMatrix4, CloverVector4>(size);
}

measurement_t measure_matrix_MVM_mixed_mat04_vec08(uint64_t size)
{
    return measure_matrix_MVM<CloverMatrix4, CloverVector8>(size);
}

measurement_t measure_matrix_MVM_mixed_mat04_vec32(uint64_t size)
{
    return measure_matrix_MVM<CloverMatrix4, CloverVector32>(size);
}

measurement_t measure_matrix_MVM_parallel_04(uint64_t size)
{
    return measure_matrix_MVM_parallel<CloverMatrix4, CloverVector4>(size);
}

measurement_t measure_matrix_MVM_parallel_mixed_mat04_vec08(uint64_t size)
{
    return measure_matrix_MVM_parallel<CloverMatrix4, CloverVector8>(size);
}

measurement_t measure_matrix_MVM_parallel_mixed_mat04_vec32(uint64_t size)
{
    return measure_matrix_MVM_parallel<CloverMatrix4, CloverVector32>(size);
}

measurement_t measure_matrix_transpose_04(uint64_t size)
{
    return measure_matrix_transpose<CloverMatrix4>(size);
}

measurement_t measure_matrix_transpose_parallel_04(uint64_t size)
{
    return measure_matrix_transpose_parallel<CloverMatrix4>(size);
}

// ===============================================================================================================
// = Quantized Linear Algebra Application
// ===============================================================================================================

measurement_t measure_IHT_or_GD_mixed_mat4_vec8(
        problem_type_t problem_type,
        CloverVector32 &x_32,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        uint64_t &K, float &mu,
        uint64_t &iterations
) {
    return measure_IHT_or_GD<CloverMatrix4, CloverVector8>(
            problem_type, x_32, Phi_32, y_32, K, mu, iterations
    );
}

measurement_t measure_IHT_or_GD_04(
        problem_type_t problem_type,
        CloverVector32 &x_32,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        uint64_t &K, float &mu,
        uint64_t &iterations
) {
    return measure_IHT_or_GD<CloverMatrix4, CloverVector4>(
            problem_type, x_32, Phi_32, y_32, K, mu, iterations
    );
}