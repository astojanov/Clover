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

#ifndef CLOVER_MATRIX4_H
#define CLOVER_MATRIX4_H

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "CloverMatrix.h"
#include "CloverMatrix32.h"
#include "CloverVector4.h"
#include "CloverVector8.h"
#include "CloverBase.h"

/**
 *
 *  CloverMatrix4 is a quantized matrix that contains M x N values.
 *  It is stored in a row-major order. One scalar value is being used
 *  for 4096 values, corresponding to one block of 64 x 64 elements,
 *  as illustrated bellow:
 *
 *              h_block
 *             64x 4-bit
 *            --------------------------  .......  --------------
 *            |           |            |           |            |
 *   v_block  | scales[0] | scales[1]  |           |            |
 *  64x 4-bit |           |            |           |            |
 *            |           |            |           |            |
 *            --------------------------  .......  --------------
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            --------------------------  .......  --------------
 *            .           .            . .         |            |
 *            .           .            .   .       |            |
 *            .           .            .     .     |            |
 *            .           .            .       .   |            |
 *            --------------------------  .......  --------------
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            |           |            |           |            |
 *            ---------------------------------------------------
 *
 *
 **/
class CloverMatrix4 : public CloverMatrix {

protected:
    int8_t * values;
    float  * scales;

    inline void allocate()
    {
        uint64_t length   = rows * cols;
        uint64_t h_blocks = rows >> 6;
        uint64_t v_blocks = cols >> 6;

        uint64_t value_bytes = (length % 2 == 0 ? length : length + 1) >> 1;
        uint64_t scale_bytes = h_blocks * v_blocks * sizeof(float);

        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes + scale_bytes);
        if (ret == 0) {
            scales = (float *) (values + value_bytes);
        } else {
            std::cout << "Could not allocate memory for CloverVector4. Exiting ..." << std::endl;
            exit(1);
        }
    }

public:

    // ===============================================================================================================
    // = Support methods and functions: Constructor / Destructor / Getters / Setters / toString etc.
    // ===============================================================================================================

    CloverMatrix4 (uint64_t h, uint64_t w) : CloverMatrix(h, w)
    {
        allocate();
    }

    uint64_t getBitsLength () const {
        return 4;
    }


    inline uint64_t getBytes () const
    {
        uint64_t length   = rows * cols;
        uint64_t v_blocks = rows >> 6;
        uint64_t h_blocks = cols >> 6;

        uint64_t value_bytes = (length % 2 == 0 ? length : length + 1) >> 1;
        uint64_t scale_bytes = h_blocks * v_blocks * sizeof(float);

        return value_bytes + scale_bytes;
    }

    inline float get(uint64_t i, uint64_t j) const
    {
        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        const uint64_t b_i = i >> 6;
        const uint64_t b_j = j >> 6;

        const float scale = scales[b_i * h_blocks + b_j] / 7.0f;

        const uint64_t pos = i * cols + j;
        const uint64_t idx = pos >> 1;
        const int8_t qu_p = values[idx];

        const int8_t qu_1 = _mm_srai_epi8_ss(qu_p << (pos % 2) * 4, 4);
        return scale * (float) qu_1;
    }

    inline std::string toString () const
    {
        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        std::stringstream sout;

        for (uint64_t i = 0; i < rows; i += 1) {
            for (uint64_t j = 0; j < cols; j += 1) {
                sout << std::setw(7) << std::fixed << std::setprecision(2) << get(i, j) << " ";
            }
            sout << ";" << std::endl;
        }

        for (uint64_t i = 0; i < v_blocks; i += 1) {
            for (uint64_t j = 0; j < h_blocks; j += 1) {
                sout << std::setw(7) << std::fixed << std::setprecision(2) << scales[i * h_blocks + j] << " ";
            }
            sout << ";" << std::endl;
        }

        return sout.str();
    }

    ~CloverMatrix4()
    {
        free(values);
    }

    // ===============================================================================================================
    // = End of support method and functions
    // ===============================================================================================================

    // ===============================================================================================================
    // = Scalar methods
    // ===============================================================================================================

    inline void quantize_scalar(const CloverMatrix32 &m)
    {
        if (m.getRows() != rows || m.getCols() != cols) {
            std::cout << "Matrices do not have the same size. Exiting ..." << std::endl;
            exit(1);
        }
        //
        // Setup data endpoints
        //
        const float * u = m.getData();
        int8_t * r      = values;
        float * sr      = scales;

        const uint64_t h_blocks = cols >> 6;
        const uint64_t v_blocks = rows >> 6;

        //
        // Note that b_i is the vertical block index
        // while b_j is the horizontal block index
        //
        for (uint64_t b_j = 0; b_j < h_blocks; b_j += 1) {
            for (uint64_t b_i = 0; b_i < v_blocks; b_i += 1) {
                //
                // Determine the block index & offset
                //
                uint64_t block_index  = (b_i * h_blocks) + b_j;
                uint64_t block_offset = (b_i << 6) * cols + (b_j << 6);
                //
                // Now find the max
                //
                float max = 0;
                for (uint64_t i = 0; i < 64; i += 1) {
                    for (uint64_t j = 0; j < 64; j += 1) {
                        const uint64_t idx = block_offset + i * cols + j;
                        const float fmax = fabsf(u[idx]);
                        if (fmax > max) {
                            max = fmax;
                        }
                    }
                }
                //
                // Define the scale, and store it on the right place
                //
                sr[block_index] = max;
                const float scaled_rcp_max = 7.0f / max;
                //
                // Perform the quantization
                //
                for (uint64_t i = 0; i < 64; i += 1) {
                    for (uint64_t j = 0; j < 64; j += 2) {

                        const uint64_t idx0 = block_offset + i * cols + j + 0;
                        const uint64_t idx1 = block_offset + i * cols + j + 1;

                        const float u1 = u[idx0];
                        const float u2 = u[idx1];

                        #ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                            const float rnd_noise1 = 0;
                            const float rnd_noise2 = 0;
                        #else
                            const float rnd_noise1 = get_random_float();
                            const float rnd_noise2 = get_random_float();
                        #endif

                        const int8_t  u_sgn1  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u1 >> 31) << 1);
                        const int8_t  u_sgn2  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u2 >> 31) << 1);

                        const uint32_t u_abs1 = clover_1st_bit_off_32 & *(uint32_t *) &u1;
                        const uint32_t u_abs2 = clover_1st_bit_off_32 & *(uint32_t *) &u2;

                        const float v_abs1 = *(float *) &u_abs1;
                        const float v_abs2 = *(float *) &u_abs2;

                        const int8_t q_abs1 = (int8_t) floorf(_mm_fmadd_ss(v_abs1, scaled_rcp_max, rnd_noise1));
                        const int8_t q_abs2 = (int8_t) floorf(_mm_fmadd_ss(v_abs2, scaled_rcp_max, rnd_noise2));

                        const int8_t q_1 = (q_abs1 * u_sgn1) << 4;
                        const int8_t q_2 = (q_abs2 * u_sgn2) & (int8_t) 0xF;

                        r[idx0 >> 1] = q_1 | q_2;
                    }
                }
            }
        }

    }

    inline void restore_scalar(CloverMatrix32 &other) const
    {
        const uint64_t blocks = cols >> 6;

        for (int i = 0; i < rows; i += 1) {

            const int8_t * u = values + i * cols / 2;
            float * r        = other.getData() + i * cols;
            float * s        = scales + (i >> 6) * blocks;

            for (uint64_t b = 0; b < blocks; b += 1) {
                //
                // Define the start and end point of the chunk
                //
                const uint64_t offset = b * 64;
                //
                // Adjust the scale
                //
                const float scaled_rcp_max = s[b] / 7.0f;
                //
                // Perform de-quantization of the block
                //
                for (uint64_t idx = 0; idx < 64; idx += 2) {
                    const uint64_t i0 = idx + offset;
                    const uint64_t i1 = i0 >> 1;

                    const int8_t qu_p = u[i1];
                    const int8_t qu_1 = _mm_srai_epi8_ss(qu_p, 4);
                    const int8_t qu_2 = _mm_srai_epi8_ss(qu_p << 4, 4);

                    r[i0 + 0] = scaled_rcp_max * (float) qu_1;
                    r[i0 + 1] = scaled_rcp_max * (float) qu_2;
                }
            }
        }
    }

    /**
     * Pure precision 4-bit Matrix and 4-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides a scalar implementation.
     *
     * @param productVector 4-bit vector that multiplies the current matrix
     * @param resultVector  4-bit vector that represents the result of the multiplication
     */
    inline void mvm_scalar(const CloverVector4 &productVector, CloverVector4 &resultVector)
    {
        if (productVector.size() != getCols() || resultVector.size() != getRows()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        const int8_t * u  = values;
        const int8_t * v  = productVector.getData();
        const float * sv  = productVector.getScales();
        int8_t * r        = resultVector.getData();
        float * sr        = resultVector.getScales();

        const uint64_t h_blocks = cols >> 6;
        const uint64_t v_blocks = rows >> 6;

        for (uint64_t b_i = 0; b_i < v_blocks; b_i += 1) {

            float max = 0;
            float block_values[64];
            uint64_t row_scales = b_i * h_blocks;

            for (uint64_t i = 0; i < 64; i += 1) {
                //
                // Create a vector from the given row in the matrix
                //
                const uint64_t row_offset = ((b_i << 6) + i) * cols;
                CloverVector4 rowVector(cols, values + (row_offset >> 1), scales + row_scales);
                //
                // Perform dot product on the two vectors
                //
                const float dot_product = rowVector.dot(productVector);
                block_values[i] = dot_product;
                //
                // Look for the maximum
                //
                const float fmax = fabsf(dot_product);
                if (fmax > max) {
                    max = fmax;
                }
            }

            //
            // Define the scale, and store it on the right place
            //
            sr[b_i] = max;
            const float scaled_rcp_max = 7.0f / max;

            for (uint64_t i = 0; i < 64; i += 2) {

                const uint64_t idx0 = (b_i << 6) + i;

                const float u1 = block_values[i + 0];
                const float u2 = block_values[i + 1];

                #ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                    const float rnd_noise1 = 0;
                    const float rnd_noise2 = 0;
                #else
                    const float rnd_noise1 = get_random_float();
                    const float rnd_noise2 = get_random_float();
                #endif

                const int8_t  u_sgn1  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u1 >> 31) << 1);
                const int8_t  u_sgn2  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u2 >> 31) << 1);

                const uint32_t u_abs1 = clover_1st_bit_off_32 & *(uint32_t *) &u1;
                const uint32_t u_abs2 = clover_1st_bit_off_32 & *(uint32_t *) &u2;

                const float v_abs1 = *(float *) &u_abs1;
                const float v_abs2 = *(float *) &u_abs2;

                const int8_t q_abs1 = (int8_t) floorf(_mm_fmadd_ss(v_abs1, scaled_rcp_max, rnd_noise1));
                const int8_t q_abs2 = (int8_t) floorf(_mm_fmadd_ss(v_abs2, scaled_rcp_max, rnd_noise2));

                const int8_t q_1 = (q_abs1 * u_sgn1) << 4;
                const int8_t q_2 = (q_abs2 * u_sgn2) & (int8_t) 0xF;

                r[idx0 >> 1] = q_1 | q_2;
            }
        }
    }

    /**
     * Mixed precision 4-bit Matrix and 8-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides scalar implementation.
     *
     * @param productVector 8-bit vector that multiplies the current matrix
     * @param resultVector  8-bit vector that represents the result of the multiplication
     */
    inline void mvm_scalar(const CloverVector8 &productVector, CloverVector8 &resultVector)
    {
        CloverVector32 resultVector32(rows);
        for (uint64_t i = 0; i < rows; i += 1) {
            double sum = 0;
            for (uint64_t j = 0; j < cols; j += 1) {
                sum += (double) get(i, j) * (double) productVector.get(j);
            }
            resultVector32.set(i, (float) sum);
        }
        resultVector.quantize(resultVector32);
    }

    /**
     * Mixed precision 4-bit Matrix and 32-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector
     *
     * @param productVector 32-bit vector that multiplies the current matrix
     * @param resultVector  32-bit vector that represents the result of the multiplication
     */
    inline void mvm_scalar(const CloverVector32 &productVector, CloverVector32 &resultVector)
    {
        for (uint64_t i = 0; i < rows; i += 1) {
            double sum = 0;
            for (uint64_t j = 0; j < cols; j += 1) {
                sum += (double) get(i, j) * (double) productVector.get(j);
            }
            resultVector.set(i, (float) sum);
        }
    }


    inline void transpose_scalar(CloverMatrix4 &other)
    {
        const int8_t * u = values;
        int8_t * v = other.values;

        const int8_t hi_mask = -16;
        const int8_t lo_mask = +15;

        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        for(uint64_t i = 0; i < rows; i += 2)
        {
            const uint64_t k_i = i >> 1;

            const uint64_t u_idx1_base = k_i * cols;
            const uint64_t u_idx2_base = k_i * cols + (cols >> 1);

            const uint64_t v_idx1_base = k_i;
            const uint64_t v_idx2_base = k_i + (rows >> 1);

            for(uint64_t j = 0; j < cols; j += 2)
            {
                const uint64_t k_j = j >> 1;

                const uint64_t u_idx1 = u_idx1_base + k_j;
                const uint64_t u_idx2 = u_idx2_base + k_j;

                const uint64_t v_idx1 = v_idx1_base + k_j * rows;
                const uint64_t v_idx2 = v_idx2_base + k_j * rows;

                const int8_t q1 = u[u_idx1];
                const int8_t q2 = u[u_idx2];

                const int8_t q11 = q1 & hi_mask;
                const int8_t q12 = q1 & lo_mask;
                const int8_t q21 = q2 & hi_mask;
                const int8_t q22 = q2 & lo_mask;

                const int8_t t11 = q11;
                const int8_t t12 = (q21 >> 4) & lo_mask;
                const int8_t t21 = q12 << 4;
                const int8_t t22 = q22;

                const int8_t t1 = t11 | t12;
                const int8_t t2 = t21 | t22;

                v[v_idx1] = t1;
                v[v_idx2] = t2;
            }
        }

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );

        //
        // Transpose the scales using MKL
        //
        // mkl_somatcopy ('R', 'T', v_blocks, h_blocks, 1, scales, 1, other.scales, 1);
    }

    // ===============================================================================================================
    // = End of scalar methods
    // ===============================================================================================================

    // ===============================================================================================================
    // = SIMD / Vectorized methods
    // ===============================================================================================================

    inline void quantize(const CloverMatrix32 &m)
    {
        if (m.getRows() != rows || m.getCols() != cols) {
            std::cout << "Matrices do not have the same size. Exiting ..." << std::endl;
            exit(1);
        }

        const uint64_t h_blocks = cols >> 6;
        const uint64_t v_blocks = rows >> 6;

        const float * u = m.getData();

        for (uint64_t b_j = 0; b_j < h_blocks; b_j += 1) {
            for (uint64_t b_i = 0; b_i < v_blocks; b_i += 1) {
                //
                // Determine the block index & offset
                //
                const uint64_t block_index  = (b_i * h_blocks) + b_j;
                const uint64_t block_offset = (b_i << 6) * cols + (b_j << 6);

                const float * u0 = u + block_offset;

                __m256 max_acc_1 = (__m256) _mm256_setzero_si256();
                __m256 max_acc_2 = (__m256) _mm256_setzero_si256();

                for (uint64_t i = 0; i < 64; i += 1)
                {
                    const float * u1 = u0 + i * cols;

                    const __m256 u_1 = _mm256_loadu_ps(u1 + 0);
                    const __m256 u_2 = _mm256_loadu_ps(u1 + 8);
                    const __m256 u_3 = _mm256_loadu_ps(u1 + 16);
                    const __m256 u_4 = _mm256_loadu_ps(u1 + 24);
                    const __m256 u_5 = _mm256_loadu_ps(u1 + 32);
                    const __m256 u_6 = _mm256_loadu_ps(u1 + 40);
                    const __m256 u_7 = _mm256_loadu_ps(u1 + 48);
                    const __m256 u_8 = _mm256_loadu_ps(u1 + 56);
                    //
                    // Get the absolute values of each
                    //
                    const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);
                    //
                    // Find the maximum
                    //
                    const __m256 m1 = _mm256_max_ps(u_abs_1, u_abs_2);
                    const __m256 m2 = _mm256_max_ps(u_abs_3, u_abs_4);
                    const __m256 m3 = _mm256_max_ps(u_abs_5, u_abs_6);
                    const __m256 m4 = _mm256_max_ps(u_abs_7, u_abs_8);
                    const __m256 m5 = _mm256_max_ps(m1, m2);
                    const __m256 m6 = _mm256_max_ps(m3, m4);
                    //
                    // Accumulate the max
                    //
                    max_acc_1 = _mm256_max_ps(m5, max_acc_1);
                    max_acc_2 = _mm256_max_ps(m6, max_acc_2);
                }

                //
                // Perform horizontal reduction, and make sure that the max is broadcasted in
                // all slots of the 256 bit lane
                //
                const __m256 hmax_0 = _mm256_max_ps(max_acc_1, max_acc_2);
                const __m256 hmax_1 = _mm256_permute2f128_ps(hmax_0, hmax_0, 3);
                const __m256 hmax_2 = _mm256_max_ps(hmax_0, hmax_1);
                const __m256 hmax_3 = _mm256_permute_ps(hmax_2, 0x4E);
                const __m256 hmax_4 = _mm256_max_ps(hmax_2, hmax_3);
                const __m256 hmax_5 = _mm256_permute_ps(hmax_4, 0xB1);
                const __m256 hmax_6 = _mm256_max_ps(hmax_4, hmax_5);

                //
                // Normalize if zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_6, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_7 = _mm256_add_ps(cndOne, hmax_6);

                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_7);

                //
                // Store the scale to the right place
                //
                _mm256_maskstore_ps(scales + block_index, clover_mm256_mask_1st_epi32, hmax_7);

                //
                // Get the starting position of the resulting memory space:
                //
                int8_t * r0 = values + (block_offset >> 1);

                for (uint64_t i = 0; i < 64; i += 1) {

                    const float * u1 = u0 + i * cols;
                    int8_t * r = r0 + ((i * cols) >> 1);

                    //
                    // Reload the initial values again
                    //
                    const __m256 u_1 = _mm256_loadu_ps(u1 + 0);
                    const __m256 u_2 = _mm256_loadu_ps(u1 + 8);
                    const __m256 u_3 = _mm256_loadu_ps(u1 + 16);
                    const __m256 u_4 = _mm256_loadu_ps(u1 + 24);
                    const __m256 u_5 = _mm256_loadu_ps(u1 + 32);
                    const __m256 u_6 = _mm256_loadu_ps(u1 + 40);
                    const __m256 u_7 = _mm256_loadu_ps(u1 + 48);
                    const __m256 u_8 = _mm256_loadu_ps(u1 + 56);

                    //
                    // Get the absolute values of each
                    //
                    const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
                    const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

                    const __m256  rnd_1 = _mm256_setzero_ps();
                    const __m256  rnd_2 = _mm256_setzero_ps();
                    const __m256  rnd_3 = _mm256_setzero_ps();
                    const __m256  rnd_4 = _mm256_setzero_ps();
                    const __m256  rnd_5 = _mm256_setzero_ps();
                    const __m256  rnd_6 = _mm256_setzero_ps();
                    const __m256  rnd_7 = _mm256_setzero_ps();
                    const __m256  rnd_8 = _mm256_setzero_ps();

#else
                    //
                    // Get the first set of 32 random numbers
                    //
                    const __m256i rnd_xor1 = avx_xorshift128plus(random_key1, random_key2);

                    const __m256i rnd_i8_1 = _mm256_and_si256 (rnd_xor1, clover_mm256_1st_bit_off_epi8);
                    const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1, 8);
                    const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
                    const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

                    const __m256 rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
                    const __m256 rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
                    const __m256 rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
                    const __m256 rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

                    const __m256 rnd_1 = _mm256_mul_ps(rnd_f8_1, clover_mm256_rcp_2pow31_ps);
                    const __m256 rnd_2 = _mm256_mul_ps(rnd_f8_2, clover_mm256_rcp_2pow31_ps);
                    const __m256 rnd_3 = _mm256_mul_ps(rnd_f8_3, clover_mm256_rcp_2pow31_ps);
                    const __m256 rnd_4 = _mm256_mul_ps(rnd_f8_4, clover_mm256_rcp_2pow31_ps);

                    //
                    // Get the second set of 32 random numbers
                    //
                    const __m256i rnd_xor2 = avx_xorshift128plus(random_key1, random_key2);

                    const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
                    const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
                    const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
                    const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

                    const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
                    const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
                    const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
                    const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

                    const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
                    const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
                    const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
                    const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);
#endif


                    //
                    // Calculate the projected values
                    //
                    const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
                    const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
                    const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
                    const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);

                    const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
                    const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
                    const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
                    const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);

                    //
                    // Truncate
                    //
                    const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
                    const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
                    const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
                    const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);

                    const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
                    const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
                    const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
                    const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);

                    //
                    // Reassemble the signs
                    //
                    __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
                    __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
                    __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
                    __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
                    __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
                    __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
                    __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
                    __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);

                    //
                    // Transpose the 8x8 registers (this might actually run faster if done right)
                    //
                    _mm256_transpose8_epi32(q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8);

                    q_1 = _mm256_slli_epi32(q_1, 28);
                    q_2 = _mm256_slli_epi32(q_2, 28);
                    q_3 = _mm256_slli_epi32(q_3, 28);
                    q_4 = _mm256_slli_epi32(q_4, 28);
                    q_5 = _mm256_slli_epi32(q_5, 28);
                    q_6 = _mm256_slli_epi32(q_6, 28);
                    q_7 = _mm256_slli_epi32(q_7, 28);
                    q_8 = _mm256_slli_epi32(q_8, 28);

                    q_1 = _mm256_srli_epi32(q_1, 6 * 4);
                    q_2 = _mm256_srli_epi32(q_2, 7 * 4);
                    q_3 = _mm256_srli_epi32(q_3, 4 * 4);
                    q_4 = _mm256_srli_epi32(q_4, 5 * 4);
                    q_5 = _mm256_srli_epi32(q_5, 2 * 4);
                    q_6 = _mm256_srli_epi32(q_6, 3 * 4);
                    q_7 = _mm256_srli_epi32(q_7, 0 * 4);
                    q_8 = _mm256_srli_epi32(q_8, 1 * 4);

                    const __m256i t1 = _mm256_or_si256(q_1, q_2);
                    const __m256i t2 = _mm256_or_si256(q_3, q_4);
                    const __m256i t3 = _mm256_or_si256(q_5, q_6);
                    const __m256i t4 = _mm256_or_si256(q_7, q_8);
                    const __m256i t5 = _mm256_or_si256(t1, t2);
                    const __m256i t6 = _mm256_or_si256(t3, t4);
                    const __m256i t7 = _mm256_or_si256(t5, t6);

                    _mm256_storeu_si256((__m256i *) r, t7);
                }
            }
        }
    }


    /**
     * Pure precision 4-bit Matrix and 4-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides SIMD version.
     *
     * @param productVector 4-bit vector that multiplies the current matrix
     * @param resultVector  4-bit vector that represents the result of the multiplication
     */
    inline void mvm(const CloverVector4 &productVector, CloverVector4 &resultVector)
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        const int8_t * v        = productVector.getData();
        const float * sv        = productVector.getScales();
        const uint64_t h_blocks = cols >> 6;

        for (uint64_t i = 0; i < rows; i += 64) {

            //
            // We process 64 rows in one go. Once we have the 64 values
            // we can quantize the same values.
            //
            float block_values[64];

            //
            // 128-bit SSE variable to keep the max element
            //
            __m128 max_ss = _mm_setzero_ps();

            //
            // Lets's calculate 64 dot products:
            //
            for (uint64_t i_block = 0; i_block < 64; i_block += 1) {

                const uint64_t m_i  = i_block >> 3;
                const uint64_t m_j  = i_block & 0x7;
                const uint64_t m_ji = (m_j << 3) | m_i;

                const int8_t * u = values + (i + i_block) * cols / 2;
                const float * su = scales + ((i + i_block) >> 6) * h_blocks;

                __m256 dot_product_acc_1 = _mm256_setzero_ps();
                __m256 dot_product_acc_2 = _mm256_setzero_ps();

                for (uint64_t b = 0; b < h_blocks; b += 2)
                {
                    const uint64_t offset_1 = b * 32;
                    const uint64_t b1       = b + 1;
                    const uint64_t b2       = b + 32;
                    const uint64_t offset_2 = offset_1 + 32;
                    const uint64_t offset_3 = offset_1 + 64;

                    const __m256i qu_1 = _mm256_loadu_si256( (__m256i *) (u + offset_1) );
                    const __m256i qu_2 = _mm256_loadu_si256( (__m256i *) (u + offset_2) );
                    const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset_1) );
                    const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset_2) );

                    const __m256 su_1 = _mm256_broadcast_ss(su + b);
                    const __m256 su_2 = _mm256_broadcast_ss(su + b1);
                    const __m256 sv_1 = _mm256_broadcast_ss(sv + b);
                    const __m256 sv_2 = _mm256_broadcast_ss(sv + b1);

                    const __m256 su_scaled_1  = _mm256_mul_ps(su_1, clover_mm256_rcp_49_ps);
                    const __m256 su_scaled_2  = _mm256_mul_ps(su_2, clover_mm256_rcp_49_ps);
                    const __m256 scaled_rcp_1 = _mm256_mul_ps(su_scaled_1, sv_1);
                    const __m256 scaled_rcp_2 = _mm256_mul_ps(su_scaled_2, sv_2);

                    _mm_prefetch((char *)(u + offset_3), _MM_HINT_T0);
                    _mm_prefetch((char *)(v + offset_3), _MM_HINT_T0);
                    _mm_prefetch((char *)(su + b2), _MM_HINT_T0);
                    _mm_prefetch((char *)(sv + b2), _MM_HINT_T0);

                    const __m256i qu_lo_shift_1 = _mm256_slli_epi16(qu_1, 4);
                    const __m256i qv_lo_shift_1 = _mm256_slli_epi16(qv_1, 4);
                    const __m256i qu_lo_shift_2 = _mm256_slli_epi16(qu_2, 4);
                    const __m256i qv_lo_shift_2 = _mm256_slli_epi16(qv_2, 4);

                    const __m256i qu_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_1);
                    const __m256i qv_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_1);
                    const __m256i qu_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_1);
                    const __m256i qv_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_1);
                    const __m256i qu_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_2);
                    const __m256i qv_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_2);
                    const __m256i qu_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_2 );
                    const __m256i qv_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_2);
                    //
                    // Get absolute values of u vectors
                    //
                    const __m256i au_hi_1 = _mm256_sign_epi8(qu_hi_1, qu_hi_1);
                    const __m256i au_lo_1 = _mm256_sign_epi8(qu_lo_1, qu_lo_1);
                    const __m256i au_hi_2 = _mm256_sign_epi8(qu_hi_2, qu_hi_2);
                    const __m256i au_lo_2 = _mm256_sign_epi8(qu_lo_2, qu_lo_2);
                    //
                    // Sign the values of the v vectors
                    //
                    const __m256i sv_hi_1 = _mm256_sign_epi8(qv_hi_1, qu_hi_1);
                    const __m256i sv_lo_1 = _mm256_sign_epi8(qv_lo_1, qu_lo_1);
                    const __m256i sv_hi_2 = _mm256_sign_epi8(qv_hi_2, qu_hi_2);
                    const __m256i sv_lo_2 = _mm256_sign_epi8(qv_lo_2, qu_lo_2);
                    //
                    // Perform multiplicatiton and create 16-bit values
                    //
                    const __m256i dot_hi_1 = _mm256_maddubs_epi16 (au_hi_1, sv_hi_1);
                    const __m256i dot_lo_1 = _mm256_maddubs_epi16 (au_lo_1, sv_lo_1);
                    const __m256i dot_hi_2 = _mm256_maddubs_epi16 (au_hi_2, sv_hi_2);
                    const __m256i dot_lo_2 = _mm256_maddubs_epi16 (au_lo_2, sv_lo_2);

                    const __m256i dot_hi_shift_1 = _mm256_srai_epi16 (dot_hi_1, 8);
                    const __m256i dot_lo_shift_1 = _mm256_srai_epi16 (dot_lo_1, 8);
                    const __m256i dot_hi_shift_2 = _mm256_srai_epi16 (dot_hi_2, 8);
                    const __m256i dot_lo_shift_2 = _mm256_srai_epi16 (dot_lo_2, 8);

                    const __m256i dot_16_1 = _mm256_add_epi16(dot_hi_shift_1, dot_lo_shift_1);
                    const __m256i dot_16_2 = _mm256_add_epi16(dot_hi_shift_2, dot_lo_shift_2);

                    const __m256i dot_32_1 = _mm256_madd_epi16(dot_16_1, clover_mm256_1_epi16);
                    const __m256i dot_32_2 = _mm256_madd_epi16(dot_16_2, clover_mm256_1_epi16);

                    const __m256  dot_f_1  = _mm256_cvtepi32_ps(dot_32_1);
                    const __m256  dot_f_2  = _mm256_cvtepi32_ps(dot_32_2);

                    //
                    // Perform dot product on the block
                    //
                    dot_product_acc_1 = _mm256_fmadd_ps(scaled_rcp_1, dot_f_1, dot_product_acc_1);
                    dot_product_acc_2 = _mm256_fmadd_ps(scaled_rcp_2, dot_f_2, dot_product_acc_2);
                }
                //
                // Perform horizontal addition
                //
                const __m256 vacc   = _mm256_add_ps(dot_product_acc_1, dot_product_acc_2);
                const __m128 hadd_0 = _mm256_extractf128_ps(vacc, 1);
                const __m128 hadd_1 = _mm256_castps256_ps128(vacc);
                const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));
                //
                // Store the result at the right place
                //
                _mm_store_ss(block_values + m_ji, hadd_4);
                //
                // Now find the maximum
                //
                const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                max_ss = _mm_max_ss(habs, max_ss);
            }

            int8_t * r = resultVector.getData() + i / 2;
            float * sr = resultVector.getScales() + (i >> 6);

            //
            // Reload the memory blocks
            //
            const __m256 u_1 = _mm256_loadu_ps(block_values +  0);
            const __m256 u_2 = _mm256_loadu_ps(block_values +  8);
            const __m256 u_3 = _mm256_loadu_ps(block_values + 16);
            const __m256 u_4 = _mm256_loadu_ps(block_values + 24);
            const __m256 u_5 = _mm256_loadu_ps(block_values + 32);
            const __m256 u_6 = _mm256_loadu_ps(block_values + 40);
            const __m256 u_7 = _mm256_loadu_ps(block_values + 48);
            const __m256 u_8 = _mm256_loadu_ps(block_values + 56);
            //
            // Get the absolute values of each
            //
            const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);

            const __m256 hmax_6 = (__m256) _mm256_broadcastd_epi32( (__m128i) max_ss);

            //
            // Avoid zero
            //
            const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_6, _mm256_setzero_si256());
            const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
            const __m256  hmax_7 = _mm256_add_ps(cndOne, hmax_6);

            //
            // Finally we have the scale
            //
            const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_7);

            //
            // Store the scale to the right place
            //
            _mm256_maskstore_ps(sr, clover_mm256_mask_1st_epi32, hmax_7);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

            const __m256  rnd_1 = _mm256_setzero_ps();
            const __m256  rnd_2 = _mm256_setzero_ps();
            const __m256  rnd_3 = _mm256_setzero_ps();
            const __m256  rnd_4 = _mm256_setzero_ps();
            const __m256  rnd_5 = _mm256_setzero_ps();
            const __m256  rnd_6 = _mm256_setzero_ps();
            const __m256  rnd_7 = _mm256_setzero_ps();
            const __m256  rnd_8 = _mm256_setzero_ps();

#else
            //
            // Get the first set of 32 random numbers
            //
            const __m256i rnd_xor1 = avx_xorshift128plus(random_key1, random_key2);

            const __m256i rnd_i8_1 = _mm256_and_si256(rnd_xor1, clover_mm256_1st_bit_off_epi8);
            const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1,  8);
            const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
            const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

            const __m256  rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
            const __m256  rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
            const __m256  rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
            const __m256  rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

            const __m256  rnd_1 = _mm256_mul_ps (rnd_f8_1, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_2 = _mm256_mul_ps (rnd_f8_2, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_3 = _mm256_mul_ps (rnd_f8_3, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_4 = _mm256_mul_ps (rnd_f8_4, clover_mm256_rcp_2pow31_ps);

            //
            // Get the second set of 32 random numbers
            //
            const __m256i rnd_xor2 = avx_xorshift128plus(random_key1, random_key2);

            const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
            const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
            const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
            const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

            const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
            const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
            const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
            const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

            const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);
#endif
            //
            // Calculate the projected values
            //
            const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
            const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
            const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
            const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);

            const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
            const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
            const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
            const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);

            //
            // Truncate
            //
            const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
            const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
            const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
            const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);

            const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
            const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
            const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
            const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);

            //
            // Reassemble the signs
            //
            __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
            __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
            __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
            __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
            __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
            __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
            __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
            __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);

            q_1 = _mm256_slli_epi32(q_1, 28);
            q_2 = _mm256_slli_epi32(q_2, 28);
            q_3 = _mm256_slli_epi32(q_3, 28);
            q_4 = _mm256_slli_epi32(q_4, 28);
            q_5 = _mm256_slli_epi32(q_5, 28);
            q_6 = _mm256_slli_epi32(q_6, 28);
            q_7 = _mm256_slli_epi32(q_7, 28);
            q_8 = _mm256_slli_epi32(q_8, 28);

            q_1 = _mm256_srli_epi32(q_1, 6 * 4);
            q_2 = _mm256_srli_epi32(q_2, 7 * 4);
            q_3 = _mm256_srli_epi32(q_3, 4 * 4);
            q_4 = _mm256_srli_epi32(q_4, 5 * 4);
            q_5 = _mm256_srli_epi32(q_5, 2 * 4);
            q_6 = _mm256_srli_epi32(q_6, 3 * 4);
            q_7 = _mm256_srli_epi32(q_7, 0 * 4);
            q_8 = _mm256_srli_epi32(q_8, 1 * 4);

            const __m256i t1 = _mm256_or_si256(q_1, q_2);
            const __m256i t2 = _mm256_or_si256(q_3, q_4);
            const __m256i t3 = _mm256_or_si256(q_5, q_6);
            const __m256i t4 = _mm256_or_si256(q_7, q_8);
            const __m256i t5 = _mm256_or_si256(t1, t2);
            const __m256i t6 = _mm256_or_si256(t3, t4);
            const __m256i t7 = _mm256_or_si256(t5, t6);

            _mm256_storeu_si256((__m256i *)r, t7);
        }

    }

    /**
     * Mixed precision 4-bit Matrix and 8-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides SIMD version.
     *
     * @param productVector 8-bit vector that multiplies the current matrix
     * @param resultVector  8-bit vector that represents the result of the multiplication
     */
    inline void mvm(const CloverVector8 &productVector, CloverVector8 &resultVector)
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        const int8_t * v        = productVector.getData();
        const float * sv        = productVector.getScales();
        const uint64_t h_blocks = cols >> 6;

        for (uint64_t row = 0; row < rows; row += 64) {

            //
            // We process 64 rows in one go. Once we have the 64 values
            // we can quantize the same values.
            //
            float block_values[64];

            //
            // 128-bit SSE variable to keep the max element
            //
            __m128 max_ss = _mm_setzero_ps();

            //
            // Lets's calculate 64 dot products:
            //
            for (uint64_t i = 0; i < 64; i += 1) {

                const uint64_t rowIdx = row + i;
                const int8_t * u = values + ((rowIdx * cols) >> 1);
                const float * su = scales + (row >> 6) * h_blocks;

                __m256 dot_product_acc = _mm256_setzero_ps();

                for (uint64_t b = 0; b < h_blocks; b += 1)
                {
                    const uint64_t offset_1 = b * 32;
                    const uint64_t offset_2 = b * 64;
                    const uint64_t offset_3 = b * 64 + 32;

                    //
                    // Load 64 4-bit elements. Permutation defined as:
                    // [ 1  0  3  2  5  4  7  6  9  8  11  10 .... 63 62]
                    //
                    // Each element is in the range [-7, 7]
                    //
                    const __m256i qu_64   = _mm256_loadu_si256( (__m256i *) (u + offset_1) );
                    //
                    // Load twice 32 8-bit elements. Permutation defined as:
                    // [ 0  1  2  3  4  5  6  7  8 ... 31]
                    // [32 33 34 35 36 37 38 39 40 ... 63]
                    //
                    // Elements are in the range [-127, 127]
                    //
                    const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset_2) );
                    const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset_3) );
                    //
                    // Setup the scales
                    //
                    const __m256 scale_u   = _mm256_broadcast_ss(su + b);
                    const __m256 scale_v   = _mm256_broadcast_ss(sv + b);
                    const __m256 su_scaled = _mm256_mul_ps(scale_u, clover_mm256_rcp_7_ps);
                    const __m256 sv_scaled = _mm256_mul_ps(scale_v, clover_mm256_rcp_127_ps);
                    const __m256 scale     = _mm256_mul_ps(su_scaled, sv_scaled);
                    //
                    // Keep the pre-fetcher busy
                    //
                    _mm_prefetch((char *)(u + offset_1 + 32), _MM_HINT_T0);
                    _mm_prefetch((char *)(v + offset_3 + 64), _MM_HINT_T0);
                    _mm_prefetch((char *)(su + b + 64), _MM_HINT_T0);
                    _mm_prefetch((char *)(sv + b + 64), _MM_HINT_T0);

                    //
                    // Split the 64-bit elements into two registers of 32-bit elements of 8-bit chunks.
                    // Permutation defined as follows:
                    //
                    // qu_32_hi = [ 1  3  5  7  9 11 13 15 17 ... 63]
                    // qu_32_lo = [ 0  2  4  6  8 10 12 14 16 ... 62]
                    //
                    // Elements are shifted by 4 bits, thus are in the range [-7, 7] * 16 = [-112, 112]
                    //
                    const __m256i qu_lo_shift = _mm256_slli_epi16(qu_64, 4);
                    const __m256i qu_32_lo    = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_64);
                    const __m256i qu_32_hi    = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift);

                    //
                    // Now put them in the permutation defined by qv:
                    // qu_1 = [ 0  1  2  3  4  5  6  7  8 ... 31]
                    // qu_2 = [32 33 34 35 36 37 38 39 40 ... 63]
                    //
                    // Elements are shifted by 4 bits, thus are in the range [-7, 7] * 16 = [-112, 112]
                    //
                    const __m256i unpack_lo_u = _mm256_unpacklo_epi8(qu_32_lo, qu_32_hi);
                    const __m256i unpack_hi_u = _mm256_unpackhi_epi8(qu_32_lo, qu_32_hi);
                    const __m256i qu_1 = _mm256_permute2f128_si256(unpack_lo_u, unpack_hi_u, 0x20);
                    const __m256i qu_2 = _mm256_permute2f128_si256(unpack_lo_u, unpack_hi_u, 0x31);

                    //
                    // Get absolute values of u vectors
                    //
                    const __m256i av_1 = _mm256_sign_epi8(qv_1, qv_1);
                    const __m256i av_2 = _mm256_sign_epi8(qv_2, qv_2);
                    //
                    // Sign the values of the v vectors
                    //
                    const __m256i su_1 = _mm256_sign_epi8(qu_1, qv_1);
                    const __m256i su_2 = _mm256_sign_epi8(qu_2, qv_2);
                    //
                    // Perform multiplication and create 16-bit values
                    // each value is in the range [-112*127*2, +112*127*2]
                    //
                    const __m256i dot_16_1 = _mm256_maddubs_epi16 (av_1, su_1);
                    const __m256i dot_16_2 = _mm256_maddubs_epi16 (av_2, su_2);
                    //
                    // Now, convert to 32-bit values range: [-112*127*4, +112*127*4]
                    //
                    const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
                    const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);
                    const __m256i dot_32_s = _mm256_add_epi32(dot_32_1, dot_32_2);
                    //
                    // Shift back the 4-bits from the initial shifts
                    //
                    const __m256i dot_32   = _mm256_srai_epi32(dot_32_s, 4);
                    //
                    // Convert to float
                    //
                    const __m256  dot_f32  = _mm256_cvtepi32_ps(dot_32);
                    //
                    // Perform dot product on the block
                    //
                    dot_product_acc = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc);
                }
                //
                // Perform horizontal addition
                //
                const __m256 vacc   = dot_product_acc;
                const __m128 hadd_0 = _mm256_extractf128_ps(vacc, 1);
                const __m128 hadd_1 = _mm256_castps256_ps128(vacc);
                const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));
                //
                // Store the result at the right place
                //
                _mm_store_ss(block_values + i, hadd_4);
                //
                // Now find the maximum
                //
                const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                max_ss = _mm_max_ss(habs, max_ss);
            }


            //
            // Now start with the re-quantization process
            //

            int8_t * r = resultVector.getData() + row;
            float * sr = resultVector.getScales() + (row >> 6);

            //
            // Reload the memory blocks
            //
            const __m256 u_1 = _mm256_loadu_ps(block_values +  0);
            const __m256 u_2 = _mm256_loadu_ps(block_values +  8);
            const __m256 u_3 = _mm256_loadu_ps(block_values + 16);
            const __m256 u_4 = _mm256_loadu_ps(block_values + 24);
            const __m256 u_5 = _mm256_loadu_ps(block_values + 32);
            const __m256 u_6 = _mm256_loadu_ps(block_values + 40);
            const __m256 u_7 = _mm256_loadu_ps(block_values + 48);
            const __m256 u_8 = _mm256_loadu_ps(block_values + 56);
            //
            // Get the absolute values of each
            //
            const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
            const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);

            const __m256 hmax_6 = (__m256) _mm256_broadcastd_epi32( (__m128i) max_ss);

            //
            // Avoid zero
            //
            const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_6, _mm256_setzero_si256());
            const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
            const __m256  hmax_7 = _mm256_add_ps(cndOne, hmax_6);
            //
            // Finally we have the scale
            //
            const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_7);
            //
            // Store the scale to the right place
            //
            _mm256_maskstore_ps(sr, clover_mm256_mask_1st_epi32, hmax_7);


#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

            const __m256 rnd_1 = _mm256_setzero_ps();
            const __m256 rnd_2 = _mm256_setzero_ps();
            const __m256 rnd_3 = _mm256_setzero_ps();
            const __m256 rnd_4 = _mm256_setzero_ps();
            const __m256 rnd_5 = _mm256_setzero_ps();
            const __m256 rnd_6 = _mm256_setzero_ps();
            const __m256 rnd_7 = _mm256_setzero_ps();
            const __m256 rnd_8 = _mm256_setzero_ps();

            //
            // Meanwhile, keep busy the pre-fetcher
            //
            // _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);
#else
            //
            // Get the first set of 32 random numbers
            //
            const __m256i rnd_xor1 = avx_xorshift128plus(random_key1, random_key2);

            const __m256i rnd_i8_1 = _mm256_and_si256(rnd_xor1, clover_mm256_1st_bit_off_epi8);
            const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1,  8);
            const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
            const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

            const __m256  rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
            const __m256  rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
            const __m256  rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
            const __m256  rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

            const __m256  rnd_1 = _mm256_mul_ps (rnd_f8_1, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_2 = _mm256_mul_ps (rnd_f8_2, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_3 = _mm256_mul_ps (rnd_f8_3, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_4 = _mm256_mul_ps (rnd_f8_4, clover_mm256_rcp_2pow31_ps);

            //
            // Meanwhile, keep busy the pre-fetcher
            //
//            _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);

            //
            // Get the second set of 32 random numbers
            //
            const __m256i rnd_xor2 = avx_xorshift128plus(random_key1, random_key2);

            const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
            const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
            const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
            const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

            const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
            const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
            const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
            const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

            const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
            const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);

#endif
            //
            // Calculate the projected values
            //
            const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
            const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
            const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
            const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);
            const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
            const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
            const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
            const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);
            //
            // Truncate
            //
            const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
            const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
            const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
            const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);
            const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
            const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
            const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
            const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);
            //
            // Reassemble the signs
            //
            const __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
            const __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
            const __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
            const __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
            const __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
            const __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
            const __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
            const __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);
            //
            // Start Packing
            //
            const __m256i left1 = _mm256_slli_epi32(q_1, 24);
            const __m256i left2 = _mm256_slli_epi32(q_2, 24);
            const __m256i left3 = _mm256_slli_epi32(q_3, 24);
            const __m256i left4 = _mm256_slli_epi32(q_4, 24);
            const __m256i left5 = _mm256_slli_epi32(q_5, 24);
            const __m256i left6 = _mm256_slli_epi32(q_6, 24);
            const __m256i left7 = _mm256_slli_epi32(q_7, 24);
            const __m256i left8 = _mm256_slli_epi32(q_8, 24);

            const __m256i right1 = _mm256_srli_epi32(left1, 24);
            const __m256i right2 = _mm256_srli_epi32(left2, 16);
            const __m256i right3 = _mm256_srli_epi32(left3, 24);
            const __m256i right4 = _mm256_srli_epi32(left4, 16);
            const __m256i right5 = _mm256_srli_epi32(left5, 24);
            const __m256i right6 = _mm256_srli_epi32(left6, 16);
            const __m256i right7 = _mm256_srli_epi32(left7, 24);
            const __m256i right8 = _mm256_srli_epi32(left8, 16);
            //
            // Combine the 8-bit chunks into 16-bit chunks
            //
            const __m256i pack16_1 = _mm256_or_si256(right1, right2);
            const __m256i pack16_2 = _mm256_or_si256(right3, right4);
            const __m256i pack16_3 = _mm256_or_si256(right5, right6);
            const __m256i pack16_4 = _mm256_or_si256(right7, right8);
            //
            // Interleave them across the 128-bit barrier
            //
            const __m256i interleave_lo_1 = _mm256_permute2f128_si256(pack16_1, pack16_2, 0x20);
            const __m256i interleave_hi_1 = _mm256_permute2f128_si256(pack16_1, pack16_2, 0x31);
            const __m256i interleave_lo_2 = _mm256_permute2f128_si256(pack16_3, pack16_4, 0x20);
            const __m256i interleave_hi_2 = _mm256_permute2f128_si256(pack16_3, pack16_4, 0x31);
            //
            // Permute them into the 128-lanes
            //
            const __m256i permute_lo_1 = _mm256_shuffle_epi8(interleave_lo_1, clover_mm256_8bit_perm_lo);
            const __m256i permute_hi_1 = _mm256_shuffle_epi8(interleave_hi_1, clover_mm256_8bit_perm_hi);
            const __m256i permute_lo_2 = _mm256_shuffle_epi8(interleave_lo_2, clover_mm256_8bit_perm_lo);
            const __m256i permute_hi_2 = _mm256_shuffle_epi8(interleave_hi_2, clover_mm256_8bit_perm_hi);
            //
            // Assemble the final package
            //
            const __m256i pack8_lo = _mm256_or_si256(permute_lo_1, permute_hi_1);
            const __m256i pack8_hi = _mm256_or_si256(permute_lo_2, permute_hi_2);

            _mm256_storeu_si256((__m256i *)(r +  0), pack8_lo);
            _mm256_storeu_si256((__m256i *)(r + 32), pack8_hi);

        }
    }

    /**
     * Mixed precision 4-bit Matrix and 32-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides parallel version.
     *
     * @param productVector 32-bit vector that multiplies the current matrix
     * @param resultVector  32-bit vector that represents the result of the multiplication
     */
    inline void mvm(const CloverVector32 &productVector, CloverVector32 &resultVector)
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        const int8_t * u0       = values;
        const float * v         = productVector.getData();
        float * r               = resultVector.getData();
        const uint64_t h_blocks = cols >> 6;

        for (uint64_t i = 0; i < rows; i += 1) {
            const int8_t *u = u0 + ((i * cols) >> 1);
            const float *su = scales + (i >> 6) * h_blocks;

            __m256 acc_1 = _mm256_setzero_ps();
            __m256 acc_2 = _mm256_setzero_ps();
            __m256 acc_3 = _mm256_setzero_ps();
            __m256 acc_4 = _mm256_setzero_ps();

            for (uint64_t b = 0; b < h_blocks; b += 1)
            {
                const uint64_t offset0 = b * 64;
                const uint64_t offset1 = b * 32;

                const __m256i qu_64 = _mm256_loadu_si256((__m256i *) (u + offset1));

                const __m256i qu_1 = _mm256_slli_epi32(qu_64, 4 * 6);
                const __m256i qu_2 = _mm256_slli_epi32(qu_64, 4 * 7);
                const __m256i qu_3 = _mm256_slli_epi32(qu_64, 4 * 4);
                const __m256i qu_4 = _mm256_slli_epi32(qu_64, 4 * 5);
                const __m256i qu_5 = _mm256_slli_epi32(qu_64, 4 * 2);
                const __m256i qu_6 = _mm256_slli_epi32(qu_64, 4 * 3);
                const __m256i qu_7 = _mm256_slli_epi32(qu_64, 4 * 0);
                const __m256i qu_8 = _mm256_slli_epi32(qu_64, 4 * 1);

                const float su_ss = su[b] / 7.0f;
                const __m256 scale = _mm256_set1_ps(su_ss);

                __m256i q_1 = _mm256_srai_epi32(qu_1, 28);
                __m256i q_2 = _mm256_srai_epi32(qu_2, 28);
                __m256i q_3 = _mm256_srai_epi32(qu_3, 28);
                __m256i q_4 = _mm256_srai_epi32(qu_4, 28);
                __m256i q_5 = _mm256_srai_epi32(qu_5, 28);
                __m256i q_6 = _mm256_srai_epi32(qu_6, 28);
                __m256i q_7 = _mm256_srai_epi32(qu_7, 28);
                __m256i q_8 = _mm256_srai_epi32(qu_8, 28);

                _mm256_transpose8_epi32(q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8);

                const __m256 fu_1 = _mm256_cvtepi32_ps(q_1);
                const __m256 fu_2 = _mm256_cvtepi32_ps(q_2);
                const __m256 fu_3 = _mm256_cvtepi32_ps(q_3);
                const __m256 fu_4 = _mm256_cvtepi32_ps(q_4);
                const __m256 fu_5 = _mm256_cvtepi32_ps(q_5);
                const __m256 fu_6 = _mm256_cvtepi32_ps(q_6);
                const __m256 fu_7 = _mm256_cvtepi32_ps(q_7);
                const __m256 fu_8 = _mm256_cvtepi32_ps(q_8);

                const __m256  v_1 = _mm256_loadu_ps(v + offset0 +  0);
                const __m256  v_2 = _mm256_loadu_ps(v + offset0 +  8);
                const __m256  v_3 = _mm256_loadu_ps(v + offset0 + 16);
                const __m256  v_4 = _mm256_loadu_ps(v + offset0 + 24);
                const __m256  v_5 = _mm256_loadu_ps(v + offset0 + 32);
                const __m256  v_6 = _mm256_loadu_ps(v + offset0 + 40);
                const __m256  v_7 = _mm256_loadu_ps(v + offset0 + 48);
                const __m256  v_8 = _mm256_loadu_ps(v + offset0 + 56);

                const __m256 f_1 = _mm256_mul_ps(fu_1, scale);
                const __m256 f_2 = _mm256_mul_ps(fu_2, scale);
                const __m256 f_3 = _mm256_mul_ps(fu_3, scale);
                const __m256 f_4 = _mm256_mul_ps(fu_4, scale);
                const __m256 f_5 = _mm256_mul_ps(fu_5, scale);
                const __m256 f_6 = _mm256_mul_ps(fu_6, scale);
                const __m256 f_7 = _mm256_mul_ps(fu_7, scale);
                const __m256 f_8 = _mm256_mul_ps(fu_8, scale);

                acc_1 = _mm256_fmadd_ps(v_1, f_1, acc_1);
                acc_2 = _mm256_fmadd_ps(v_2, f_2, acc_2);
                acc_3 = _mm256_fmadd_ps(v_3, f_3, acc_3);
                acc_4 = _mm256_fmadd_ps(v_4, f_4, acc_4);

                acc_1 = _mm256_fmadd_ps(v_5, f_5, acc_1);
                acc_2 = _mm256_fmadd_ps(v_6, f_6, acc_2);
                acc_3 = _mm256_fmadd_ps(v_7, f_7, acc_3);
                acc_4 = _mm256_fmadd_ps(v_8, f_8, acc_4);
            }

            const __m256 sum_1 = _mm256_add_ps(acc_1, acc_2);
            const __m256 sum_2 = _mm256_add_ps(acc_3, acc_4);
            const __m256 sum_3 = _mm256_add_ps(sum_1, sum_2);

            r[i] = _mm256_haddf32_ps(sum_3);
        }

    }

    inline void transpose(CloverMatrix4 &other)
    {
        uint32_t scatter[8];

        int8_t * u = values;
        int8_t * v = other.values;

        const uint32_t h_stride = (uint32_t)(cols >> 1);
        const uint32_t v_stride = (uint32_t)(rows >> 1);

        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        const __m256i bit_masks = _mm256_setr_epi32 (
                0x000000F0, 0x0000000F, 0x0000F000, 0x00000F00,
                0x00F00000, 0x000F0000, 0xF0000000, 0x0F000000
        );

        const __m256i l_masks_0 = _mm256_setr_epi32 ( 0,  4,  0,  0,  0,  0,  0,  0);
        const __m256i l_masks_1 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  0,  0);
        const __m256i l_masks_2 = _mm256_setr_epi32 ( 8, 12,  0,  4,  0,  0,  0,  0);
        const __m256i l_masks_3 = _mm256_setr_epi32 ( 4,  8,  0,  0,  0,  0,  0,  0);
        const __m256i l_masks_4 = _mm256_setr_epi32 (16, 20,  8, 12,  0,  4,  0,  0);
        const __m256i l_masks_5 = _mm256_setr_epi32 (12, 16,  4,  8,  0,  0,  0,  0);
        const __m256i l_masks_6 = _mm256_setr_epi32 (24, 28, 16, 20,  8, 12,  0,  4);
        const __m256i l_masks_7 = _mm256_setr_epi32 (20, 24, 12, 16,  4,  8,  0,  0);

        const __m256i r_masks_0 = _mm256_setr_epi32 ( 0,  0,  8,  4, 16, 12, 24, 20);
        const __m256i r_masks_1 = _mm256_setr_epi32 ( 4,  0, 12,  8, 20, 16, 28, 24);
        const __m256i r_masks_2 = _mm256_setr_epi32 ( 0,  0,  0,  0,  8,  4, 16, 12);
        const __m256i r_masks_3 = _mm256_setr_epi32 ( 0,  0,  4,  0, 12,  8, 20, 16);
        const __m256i r_masks_4 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  8,  4);
        const __m256i r_masks_5 = _mm256_setr_epi32 ( 0,  0,  0,  0,  4,  0, 12,  8);
        const __m256i r_masks_6 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  0,  0);
        const __m256i r_masks_7 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  4,  0);

        for(uint64_t i = 0; i < rows; i += 8)
        {
            for(uint64_t j = 0; j < cols; j += 8)
            {
                int8_t * src = u + ((i * cols + j) >> 1);
                int8_t * dst = v + ((j * rows + i) >> 1);

                __m256i S0 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 0) );
                __m256i S1 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 1) );
                __m256i S2 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 2) );
                __m256i S3 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 3) );
                __m256i S4 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 4) );
                __m256i S5 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 5) );
                __m256i S6 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 6) );
                __m256i S7 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 7) );

                const __m256i M0 = _mm256_and_si256(S0, bit_masks);
                const __m256i M1 = _mm256_and_si256(S1, bit_masks);
                const __m256i M2 = _mm256_and_si256(S2, bit_masks);
                const __m256i M3 = _mm256_and_si256(S3, bit_masks);
                const __m256i M4 = _mm256_and_si256(S4, bit_masks);
                const __m256i M5 = _mm256_and_si256(S5, bit_masks);
                const __m256i M6 = _mm256_and_si256(S6, bit_masks);
                const __m256i M7 = _mm256_and_si256(S7, bit_masks);

                const __m256i L0 = _mm256_sllv_epi32(M0, l_masks_0);
                const __m256i L1 = _mm256_sllv_epi32(M1, l_masks_1);
                const __m256i L2 = _mm256_sllv_epi32(M2, l_masks_2);
                const __m256i L3 = _mm256_sllv_epi32(M3, l_masks_3);
                const __m256i L4 = _mm256_sllv_epi32(M4, l_masks_4);
                const __m256i L5 = _mm256_sllv_epi32(M5, l_masks_5);
                const __m256i L6 = _mm256_sllv_epi32(M6, l_masks_6);
                const __m256i L7 = _mm256_sllv_epi32(M7, l_masks_7);

                const __m256i R0 = _mm256_srlv_epi32(L0, r_masks_0);
                const __m256i R1 = _mm256_srlv_epi32(L1, r_masks_1);
                const __m256i R2 = _mm256_srlv_epi32(L2, r_masks_2);
                const __m256i R3 = _mm256_srlv_epi32(L3, r_masks_3);
                const __m256i R4 = _mm256_srlv_epi32(L4, r_masks_4);
                const __m256i R5 = _mm256_srlv_epi32(L5, r_masks_5);
                const __m256i R6 = _mm256_srlv_epi32(L6, r_masks_6);
                const __m256i R7 = _mm256_srlv_epi32(L7, r_masks_7);

                const __m256i T0 = _mm256_or_si256(R0, R1);
                const __m256i T1 = _mm256_or_si256(R2, R3);
                const __m256i T2 = _mm256_or_si256(R4, R5);
                const __m256i T3 = _mm256_or_si256(R6, R7);
                const __m256i T4 = _mm256_or_si256(T0, T1);
                const __m256i T5 = _mm256_or_si256(T2, T3);
                const __m256i T6 = _mm256_or_si256(T4, T5);

                _mm256_storeu_si256((__m256i *) scatter, T6);
                *(uint32_t *)(dst + v_stride * 0) = scatter[0];
                *(uint32_t *)(dst + v_stride * 1) = scatter[1];
                *(uint32_t *)(dst + v_stride * 2) = scatter[2];
                *(uint32_t *)(dst + v_stride * 3) = scatter[3];
                *(uint32_t *)(dst + v_stride * 4) = scatter[4];
                *(uint32_t *)(dst + v_stride * 5) = scatter[5];
                *(uint32_t *)(dst + v_stride * 6) = scatter[6];
                *(uint32_t *)(dst + v_stride * 7) = scatter[7];

            }
        }

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );
        //
        // Transpose the scales using MKL
        //
        // mkl_somatcopy ('R', 'T', v_blocks, h_blocks, 1, scales, h_blocks, other.scales, v_blocks);
    }

    // ===============================================================================================================
    // = End of SIMD / Vectorized methods
    // ===============================================================================================================

    // ===============================================================================================================
    // = Parallelized and vectorized methods
    // ===============================================================================================================

    /**
    * Pure precision 4-bit Matrix and 4bit Vector - Matrix Vector Multiplication
    * This method will multiply the current matrix with the product vector and store
    * the result into the result vector. It provides parallel version.
    *
    * @param productVector 4-bit vector that multiplies the current matrix
    * @param resultVector  4-bit vector that represents the result of the multiplication
    */
    inline void mvm_parallel(const CloverVector4 &productVector, CloverVector4 &resultVector)
    {
#if defined(_OPENMP)
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        _Pragma("omp parallel") {
            const int8_t * v        = productVector.getData();
            const float * sv        = productVector.getScales();
            const uint64_t h_blocks = cols >> 6;

            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            __m256i my_key1 = random_key1_perthread[tid];
            __m256i my_key2 = random_key2_perthread[tid];

            uint64_t n_rowblocks = (rows - 1) / 64 + 1;
            uint64_t rowblocks_per_thread = (n_rowblocks - 1) / nt + 1;
            uint64_t start = 64 * rowblocks_per_thread * tid;
            uint64_t end = std::min(rows, start + 64 * rowblocks_per_thread);

            for (uint64_t i = start; i < end; i += 64) {
                //
                // We process 64 rows in one go. Once we have the 64 values
                // we can quantize the same values.
                //
                float block_values[64];

                //
                // 128-bit SSE variable to keep the max element
                //
                __m128 max_ss = _mm_setzero_ps();

                //
                // Lets's calculate 64 dot products:
                //
                for (int i_block = 0; i_block < 64; i_block += 1) {

                    const uint64_t m_i = i_block >> 3;
                    const uint64_t m_j = i_block & 0x7;
                    const uint64_t m_ji = (m_j << 3) | m_i;

                    const int8_t * u = values + (i + i_block) * cols / 2;
                    const float * su = scales + ((i + i_block) >> 6) * h_blocks;

                    __m256 dot_product_acc_1 = _mm256_setzero_ps();
                    __m256 dot_product_acc_2 = _mm256_setzero_ps();

                    for (uint64_t b = 0; b < h_blocks; b += 2)
                    {
                        const uint64_t offset_1 = b * 32;
                        const uint64_t b1       = b + 1;
                        const uint64_t b2       = b + 32;
                        const uint64_t offset_2 = offset_1 + 32;
                        const uint64_t offset_3 = offset_1 + 64;

                        const __m256i qu_1 = _mm256_loadu_si256( (__m256i *) (u + offset_1) );
                        const __m256i qu_2 = _mm256_loadu_si256( (__m256i *) (u + offset_2) );
                        const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset_1) );
                        const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset_2) );

                        const __m256 su_1 = _mm256_broadcast_ss(su + b);
                        const __m256 su_2 = _mm256_broadcast_ss(su + b1);
                        const __m256 sv_1 = _mm256_broadcast_ss(sv + b);
                        const __m256 sv_2 = _mm256_broadcast_ss(sv + b1);

                        const __m256 su_scaled_1  = _mm256_mul_ps(su_1, clover_mm256_rcp_49_ps);
                        const __m256 su_scaled_2  = _mm256_mul_ps(su_2, clover_mm256_rcp_49_ps);
                        const __m256 scaled_rcp_1 = _mm256_mul_ps(su_scaled_1, sv_1);
                        const __m256 scaled_rcp_2 = _mm256_mul_ps(su_scaled_2, sv_2);

                        _mm_prefetch((char *)(u + offset_3), _MM_HINT_T0);
                        _mm_prefetch((char *)(v + offset_3), _MM_HINT_T0);
                        _mm_prefetch((char *)(su + b2), _MM_HINT_T0);
                        _mm_prefetch((char *)(sv + b2), _MM_HINT_T0);

                        const __m256i qu_lo_shift_1 = _mm256_slli_epi16(qu_1, 4);
                        const __m256i qv_lo_shift_1 = _mm256_slli_epi16(qv_1, 4);
                        const __m256i qu_lo_shift_2 = _mm256_slli_epi16(qu_2, 4);
                        const __m256i qv_lo_shift_2 = _mm256_slli_epi16(qv_2, 4);

                        const __m256i qu_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_1);
                        const __m256i qv_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_1);
                        const __m256i qu_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_1);
                        const __m256i qv_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_1);
                        const __m256i qu_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_2);
                        const __m256i qv_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_2);
                        const __m256i qu_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_2 );
                        const __m256i qv_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_2);
                        //
                        // Get absolute values of u vectors
                        //
                        const __m256i au_hi_1 = _mm256_sign_epi8(qu_hi_1, qu_hi_1);
                        const __m256i au_lo_1 = _mm256_sign_epi8(qu_lo_1, qu_lo_1);
                        const __m256i au_hi_2 = _mm256_sign_epi8(qu_hi_2, qu_hi_2);
                        const __m256i au_lo_2 = _mm256_sign_epi8(qu_lo_2, qu_lo_2);
                        //
                        // Sign the values of the v vectors
                        //
                        const __m256i sv_hi_1 = _mm256_sign_epi8(qv_hi_1, qu_hi_1);
                        const __m256i sv_lo_1 = _mm256_sign_epi8(qv_lo_1, qu_lo_1);
                        const __m256i sv_hi_2 = _mm256_sign_epi8(qv_hi_2, qu_hi_2);
                        const __m256i sv_lo_2 = _mm256_sign_epi8(qv_lo_2, qu_lo_2);
                        //
                        // Perform multiplicatiton and create 16-bit values
                        //
                        const __m256i dot_hi_1 = _mm256_maddubs_epi16 (au_hi_1, sv_hi_1);
                        const __m256i dot_lo_1 = _mm256_maddubs_epi16 (au_lo_1, sv_lo_1);
                        const __m256i dot_hi_2 = _mm256_maddubs_epi16 (au_hi_2, sv_hi_2);
                        const __m256i dot_lo_2 = _mm256_maddubs_epi16 (au_lo_2, sv_lo_2);

                        const __m256i dot_hi_shift_1 = _mm256_srai_epi16 (dot_hi_1, 8);
                        const __m256i dot_lo_shift_1 = _mm256_srai_epi16 (dot_lo_1, 8);
                        const __m256i dot_hi_shift_2 = _mm256_srai_epi16 (dot_hi_2, 8);
                        const __m256i dot_lo_shift_2 = _mm256_srai_epi16 (dot_lo_2, 8);

                        const __m256i dot_16_1 = _mm256_add_epi16(dot_hi_shift_1, dot_lo_shift_1);
                        const __m256i dot_16_2 = _mm256_add_epi16(dot_hi_shift_2, dot_lo_shift_2);

                        const __m256i dot_32_1 = _mm256_madd_epi16(dot_16_1, clover_mm256_1_epi16);
                        const __m256i dot_32_2 = _mm256_madd_epi16(dot_16_2, clover_mm256_1_epi16);

                        const __m256  dot_f_1  = _mm256_cvtepi32_ps(dot_32_1);
                        const __m256  dot_f_2  = _mm256_cvtepi32_ps(dot_32_2);

                        //
                        // Perform dot product on the block
                        //
                        dot_product_acc_1 = _mm256_fmadd_ps(scaled_rcp_1, dot_f_1, dot_product_acc_1);
                        dot_product_acc_2 = _mm256_fmadd_ps(scaled_rcp_2, dot_f_2, dot_product_acc_2);
                    }

                    //
                    // Perform horizontal addition
                    //
                    const __m256 vacc   = _mm256_add_ps(dot_product_acc_1, dot_product_acc_2);
                    const __m128 hadd_0 = _mm256_extractf128_ps(vacc, 1);
                    const __m128 hadd_1 = _mm256_castps256_ps128(vacc);
                    const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                    const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                    const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));
                    //
                    // Store the result at the right place
                    //
                    _mm_store_ss(block_values + m_ji, hadd_4);
                    //
                    // Now find the minimum
                    //
                    const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                    max_ss = _mm_max_ss(habs, max_ss);
                }

                int8_t * r  = resultVector.getData() + i / 2;
                float * sr  = resultVector.getScales() + (i >> 6);

                //
                // Reload the memory blocks
                //
                const __m256 u_1 = _mm256_loadu_ps(block_values +  0);
                const __m256 u_2 = _mm256_loadu_ps(block_values +  8);
                const __m256 u_3 = _mm256_loadu_ps(block_values + 16);
                const __m256 u_4 = _mm256_loadu_ps(block_values + 24);
                const __m256 u_5 = _mm256_loadu_ps(block_values + 32);
                const __m256 u_6 = _mm256_loadu_ps(block_values + 40);
                const __m256 u_7 = _mm256_loadu_ps(block_values + 48);
                const __m256 u_8 = _mm256_loadu_ps(block_values + 56);
                //
                // Get the absolute values of each
                //
                const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);

                const __m256 hmax_6 = (__m256) _mm256_broadcastd_epi32( (__m128i) max_ss);

                //
                // Avoid zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_6, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_7 = _mm256_add_ps(cndOne, hmax_6);

                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_7);

                //
                // Store the scale to the right place
                //
                _mm256_maskstore_ps(sr, clover_mm256_mask_1st_epi32, hmax_7);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

                const __m256  rnd_1 = _mm256_setzero_ps();
                const __m256  rnd_2 = _mm256_setzero_ps();
                const __m256  rnd_3 = _mm256_setzero_ps();
                const __m256  rnd_4 = _mm256_setzero_ps();
                const __m256  rnd_5 = _mm256_setzero_ps();
                const __m256  rnd_6 = _mm256_setzero_ps();
                const __m256  rnd_7 = _mm256_setzero_ps();
                const __m256  rnd_8 = _mm256_setzero_ps();

#else
                //
                // Get the first set of 32 random numbers
                //
                const __m256i rnd_xor1 = avx_xorshift128plus(my_key1, my_key2);
               // const __m256i rnd_xor1 = avx_xorshift128plus(random_key1, random_key2);

                const __m256i rnd_i8_1 = _mm256_and_si256(rnd_xor1, clover_mm256_1st_bit_off_epi8);
                const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1,  8);
                const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
                const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

                const __m256  rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
                const __m256  rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
                const __m256  rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
                const __m256  rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

                const __m256  rnd_1 = _mm256_mul_ps (rnd_f8_1, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_2 = _mm256_mul_ps (rnd_f8_2, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_3 = _mm256_mul_ps (rnd_f8_3, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_4 = _mm256_mul_ps (rnd_f8_4, clover_mm256_rcp_2pow31_ps);

                //
                // Get the second set of 32 random numbers
                //
                const __m256i rnd_xor2 = avx_xorshift128plus(my_key1, my_key2);
                //const __m256i rnd_xor2 = avx_xorshift128plus(random_key1, random_key2);

                const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
                const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
                const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
                const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

                const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
                const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
                const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
                const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

                const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);
#endif
                //
                // Calculate the projected values
                //
                const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
                const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
                const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
                const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);

                const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
                const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
                const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
                const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);

                //
                // Truncate
                //
                const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
                const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
                const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
                const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);

                const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
                const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
                const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
                const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);

                //
                // Reassemble the signs
                //
                __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
                __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
                __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
                __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
                __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
                __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
                __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
                __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);

                q_1 = _mm256_slli_epi32(q_1, 28);
                q_2 = _mm256_slli_epi32(q_2, 28);
                q_3 = _mm256_slli_epi32(q_3, 28);
                q_4 = _mm256_slli_epi32(q_4, 28);
                q_5 = _mm256_slli_epi32(q_5, 28);
                q_6 = _mm256_slli_epi32(q_6, 28);
                q_7 = _mm256_slli_epi32(q_7, 28);
                q_8 = _mm256_slli_epi32(q_8, 28);

                q_1 = _mm256_srli_epi32(q_1, 6 * 4);
                q_2 = _mm256_srli_epi32(q_2, 7 * 4);
                q_3 = _mm256_srli_epi32(q_3, 4 * 4);
                q_4 = _mm256_srli_epi32(q_4, 5 * 4);
                q_5 = _mm256_srli_epi32(q_5, 2 * 4);
                q_6 = _mm256_srli_epi32(q_6, 3 * 4);
                q_7 = _mm256_srli_epi32(q_7, 0 * 4);
                q_8 = _mm256_srli_epi32(q_8, 1 * 4);

                const __m256i t1 = _mm256_or_si256(q_1, q_2);
                const __m256i t2 = _mm256_or_si256(q_3, q_4);
                const __m256i t3 = _mm256_or_si256(q_5, q_6);
                const __m256i t4 = _mm256_or_si256(q_7, q_8);
                const __m256i t5 = _mm256_or_si256(t1, t2);
                const __m256i t6 = _mm256_or_si256(t3, t4);
                const __m256i t7 = _mm256_or_si256(t5, t6);

                _mm256_storeu_si256((__m256i *)r, t7);
            }
            random_key1_perthread[tid] = my_key1;
            random_key2_perthread[tid] = my_key2;
        }
#else
        mvm(productVector, resultVector);
#endif
    }

    /**
     * Mixed precision 4-bit Matrix and 8-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides parallel version.
     *
     * @param productVector 8-bit vector that multiplies the current matrix
     * @param resultVector  8-bit vector that represents the result of the multiplication
     */
    inline void mvm_parallel(const CloverVector8 &productVector, CloverVector8 &resultVector)
    {
#if defined(_OPENMP)
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        _Pragma("omp parallel") {
            const int8_t * v        = productVector.getData();
            const float * sv        = productVector.getScales();
            const uint64_t h_blocks = cols >> 6;

            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            __m256i my_key1 = random_key1_perthread[tid];
            __m256i my_key2 = random_key2_perthread[tid];

            uint64_t n_rowblocks = (rows - 1) / 64 + 1;
            uint64_t rowblocks_per_thread = (n_rowblocks - 1) / nt + 1;
            uint64_t start = 64 * rowblocks_per_thread * tid;
            uint64_t end = std::min(rows, start + 64 * rowblocks_per_thread);

            for (uint64_t row = start; row < end; row += 64) {

                //
                // We process 64 rows in one go. Once we have the 64 values
                // we can quantize the same values.
                //
                float block_values[64];

                //
                // 128-bit SSE variable to keep the max element
                //
                __m128 max_ss = _mm_setzero_ps();

                //
                // Lets's calculate 64 dot products:
                //
                for (uint64_t i = 0; i < 64; i += 1) {

                    const uint64_t rowIdx = row + i;
                    const int8_t * u = values + ((rowIdx * cols) >> 1);
                    const float * su = scales + (row >> 6) * h_blocks;

                    __m256 dot_product_acc = _mm256_setzero_ps();

                    for (uint64_t b = 0; b < h_blocks; b += 1)
                    {
                        const uint64_t offset_1 = b * 32;
                        const uint64_t offset_2 = b * 64;
                        const uint64_t offset_3 = b * 64 + 32;

                        //
                        // Load 64 4-bit elements. Permutation defined as:
                        // [ 1  0  3  2  5  4  7  6  9  8  11  10 .... 63 62]
                        //
                        // Each element is in the range [-7, 7]
                        //
                        const __m256i qu_64   = _mm256_loadu_si256( (__m256i *) (u + offset_1) );
                        //
                        // Load twice 32 8-bit elements. Permutation defined as:
                        // [ 0  1  2  3  4  5  6  7  8 ... 31]
                        // [32 33 34 35 36 37 38 39 40 ... 63]
                        //
                        // Elements are in the range [-127, 127]
                        //
                        const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset_2) );
                        const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset_3) );
                        //
                        // Setup the scales
                        //
                        const __m256 scale_u   = _mm256_broadcast_ss(su + b);
                        const __m256 scale_v   = _mm256_broadcast_ss(sv + b);
                        const __m256 su_scaled = _mm256_mul_ps(scale_u, clover_mm256_rcp_7_ps);
                        const __m256 sv_scaled = _mm256_mul_ps(scale_v, clover_mm256_rcp_127_ps);
                        const __m256 scale     = _mm256_mul_ps(su_scaled, sv_scaled);
                        //
                        // Keep the pre-fetcher busy
                        //
                        _mm_prefetch((char *)(u + offset_1 + 32), _MM_HINT_T0);
                        _mm_prefetch((char *)(v + offset_3 + 64), _MM_HINT_T0);
                        _mm_prefetch((char *)(su + b + 64), _MM_HINT_T0);
                        _mm_prefetch((char *)(sv + b + 64), _MM_HINT_T0);

                        //
                        // Split the 64-bit elements into two registers of 32-bit elements of 8-bit chunks.
                        // Permutation defined as follows:
                        //
                        // qu_32_hi = [ 1  3  5  7  9 11 13 15 17 ... 63]
                        // qu_32_lo = [ 0  2  4  6  8 10 12 14 16 ... 62]
                        //
                        // Elements are shifted by 4 bits, thus are in the range [-7, 7] * 16 = [-112, 112]
                        //
                        const __m256i qu_lo_shift = _mm256_slli_epi16(qu_64, 4);
                        const __m256i qu_32_lo    = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_64);
                        const __m256i qu_32_hi    = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift);

                        //
                        // Now put them in the permutation defined by qv:
                        // qu_1 = [ 0  1  2  3  4  5  6  7  8 ... 31]
                        // qu_2 = [32 33 34 35 36 37 38 39 40 ... 63]
                        //
                        // Elements are shifted by 4 bits, thus are in the range [-7, 7] * 16 = [-112, 112]
                        //
                        const __m256i unpack_lo_u = _mm256_unpacklo_epi8(qu_32_lo, qu_32_hi);
                        const __m256i unpack_hi_u = _mm256_unpackhi_epi8(qu_32_lo, qu_32_hi);
                        const __m256i qu_1 = _mm256_permute2f128_si256(unpack_lo_u, unpack_hi_u, 0x20);
                        const __m256i qu_2 = _mm256_permute2f128_si256(unpack_lo_u, unpack_hi_u, 0x31);

                        //
                        // Get absolute values of u vectors
                        //
                        const __m256i av_1 = _mm256_sign_epi8(qv_1, qv_1);
                        const __m256i av_2 = _mm256_sign_epi8(qv_2, qv_2);
                        //
                        // Sign the values of the v vectors
                        //
                        const __m256i su_1 = _mm256_sign_epi8(qu_1, qv_1);
                        const __m256i su_2 = _mm256_sign_epi8(qu_2, qv_2);
                        //
                        // Perform multiplication and create 16-bit values
                        // each value is in the range [-112*127*2, +112*127*2]
                        //
                        const __m256i dot_16_1 = _mm256_maddubs_epi16 (av_1, su_1);
                        const __m256i dot_16_2 = _mm256_maddubs_epi16 (av_2, su_2);
                        //
                        // Now, convert to 32-bit values range: [-112*127*4, +112*127*4]
                        //
                        const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
                        const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);
                        const __m256i dot_32_s = _mm256_add_epi32(dot_32_1, dot_32_2);
                        //
                        // Shift back the 4-bits from the initial shifts
                        //
                        const __m256i dot_32   = _mm256_srai_epi32(dot_32_s, 4);
                        //
                        // Convert to float
                        //
                        const __m256  dot_f32  = _mm256_cvtepi32_ps(dot_32);
                        //
                        // Perform dot product on the block
                        //
                        dot_product_acc = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc);
                    }
                    //
                    // Perform horizontal addition
                    //
                    const __m256 vacc   = dot_product_acc;
                    const __m128 hadd_0 = _mm256_extractf128_ps(vacc, 1);
                    const __m128 hadd_1 = _mm256_castps256_ps128(vacc);
                    const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                    const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                    const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));
                    //
                    // Store the result at the right place
                    //
                    _mm_store_ss(block_values + i, hadd_4);
                    //
                    // Now find the maximum
                    //
                    const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                    max_ss = _mm_max_ss(habs, max_ss);
                }


                /* ========================================================================================== */
                // Re-quantization begins here
                /* ========================================================================================== */

                int8_t * r = resultVector.getData() + row;
                float * sr = resultVector.getScales() + (row >> 6);

                //
                // Reload the memory blocks
                //
                const __m256 u_1 = _mm256_loadu_ps(block_values +  0);
                const __m256 u_2 = _mm256_loadu_ps(block_values +  8);
                const __m256 u_3 = _mm256_loadu_ps(block_values + 16);
                const __m256 u_4 = _mm256_loadu_ps(block_values + 24);
                const __m256 u_5 = _mm256_loadu_ps(block_values + 32);
                const __m256 u_6 = _mm256_loadu_ps(block_values + 40);
                const __m256 u_7 = _mm256_loadu_ps(block_values + 48);
                const __m256 u_8 = _mm256_loadu_ps(block_values + 56);
                //
                // Get the absolute values of each
                //
                const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
                const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);

                const __m256 hmax_6 = (__m256) _mm256_broadcastd_epi32( (__m128i) max_ss);

                //
                // Avoid zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_6, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_7 = _mm256_add_ps(cndOne, hmax_6);


                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_7);

                //
                // Store the scale to the right place
                //
                _mm256_maskstore_ps(sr, clover_mm256_mask_1st_epi32, hmax_7);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

                const __m256 rnd_1 = _mm256_setzero_ps();
                const __m256 rnd_2 = _mm256_setzero_ps();
                const __m256 rnd_3 = _mm256_setzero_ps();
                const __m256 rnd_4 = _mm256_setzero_ps();
                const __m256 rnd_5 = _mm256_setzero_ps();
                const __m256 rnd_6 = _mm256_setzero_ps();
                const __m256 rnd_7 = _mm256_setzero_ps();
                const __m256 rnd_8 = _mm256_setzero_ps();

                //
                // Meanwhile, keep busy the pre-fetcher
                //
                // _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);
#else
                //
                // Get the first set of 32 random numbers
                //
                const __m256i rnd_xor1 = avx_xorshift128plus(my_key1, my_key2);

                const __m256i rnd_i8_1 = _mm256_and_si256(rnd_xor1, clover_mm256_1st_bit_off_epi8);
                const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1,  8);
                const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
                const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

                const __m256  rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
                const __m256  rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
                const __m256  rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
                const __m256  rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

                const __m256  rnd_1 = _mm256_mul_ps (rnd_f8_1, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_2 = _mm256_mul_ps (rnd_f8_2, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_3 = _mm256_mul_ps (rnd_f8_3, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_4 = _mm256_mul_ps (rnd_f8_4, clover_mm256_rcp_2pow31_ps);

                //
                // Meanwhile, keep busy the pre-fetcher
                //
    //            _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);

                //
                // Get the second set of 32 random numbers
                //
                const __m256i rnd_xor2 = avx_xorshift128plus(my_key1, my_key2);

                const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
                const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
                const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
                const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

                const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
                const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
                const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
                const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

                const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
                const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);

#endif
                //
                // Calculate the projected values
                //
                const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
                const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
                const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
                const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);
                const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
                const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
                const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
                const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);
                //
                // Truncate
                //
                const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
                const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
                const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
                const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);
                const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
                const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
                const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
                const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);
                //
                // Reassemble the signs
                //
                const __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
                const __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
                const __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
                const __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
                const __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
                const __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
                const __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
                const __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);
                //
                // Start Packing
                //
                const __m256i left1 = _mm256_slli_epi32(q_1, 24);
                const __m256i left2 = _mm256_slli_epi32(q_2, 24);
                const __m256i left3 = _mm256_slli_epi32(q_3, 24);
                const __m256i left4 = _mm256_slli_epi32(q_4, 24);
                const __m256i left5 = _mm256_slli_epi32(q_5, 24);
                const __m256i left6 = _mm256_slli_epi32(q_6, 24);
                const __m256i left7 = _mm256_slli_epi32(q_7, 24);
                const __m256i left8 = _mm256_slli_epi32(q_8, 24);

                const __m256i right1 = _mm256_srli_epi32(left1, 24);
                const __m256i right2 = _mm256_srli_epi32(left2, 16);
                const __m256i right3 = _mm256_srli_epi32(left3, 24);
                const __m256i right4 = _mm256_srli_epi32(left4, 16);
                const __m256i right5 = _mm256_srli_epi32(left5, 24);
                const __m256i right6 = _mm256_srli_epi32(left6, 16);
                const __m256i right7 = _mm256_srli_epi32(left7, 24);
                const __m256i right8 = _mm256_srli_epi32(left8, 16);
                //
                // Combine the 8-bit chunks into 16-bit chunks
                //
                const __m256i pack16_1 = _mm256_or_si256(right1, right2);
                const __m256i pack16_2 = _mm256_or_si256(right3, right4);
                const __m256i pack16_3 = _mm256_or_si256(right5, right6);
                const __m256i pack16_4 = _mm256_or_si256(right7, right8);
                //
                // Interleave them across the 128-bit barrier
                //
                const __m256i interleave_lo_1 = _mm256_permute2f128_si256(pack16_1, pack16_2, 0x20);
                const __m256i interleave_hi_1 = _mm256_permute2f128_si256(pack16_1, pack16_2, 0x31);
                const __m256i interleave_lo_2 = _mm256_permute2f128_si256(pack16_3, pack16_4, 0x20);
                const __m256i interleave_hi_2 = _mm256_permute2f128_si256(pack16_3, pack16_4, 0x31);
                //
                // Permute them into the 128-lanes
                //
                const __m256i permute_lo_1 = _mm256_shuffle_epi8(interleave_lo_1, clover_mm256_8bit_perm_lo);
                const __m256i permute_hi_1 = _mm256_shuffle_epi8(interleave_hi_1, clover_mm256_8bit_perm_hi);
                const __m256i permute_lo_2 = _mm256_shuffle_epi8(interleave_lo_2, clover_mm256_8bit_perm_lo);
                const __m256i permute_hi_2 = _mm256_shuffle_epi8(interleave_hi_2, clover_mm256_8bit_perm_hi);
                //
                // Assemble the final package
                //
                const __m256i pack8_lo = _mm256_or_si256(permute_lo_1, permute_hi_1);
                const __m256i pack8_hi = _mm256_or_si256(permute_lo_2, permute_hi_2);

                _mm256_storeu_si256((__m256i *)(r +  0), pack8_lo);
                _mm256_storeu_si256((__m256i *)(r + 32), pack8_hi);

            }

            random_key1_perthread[tid] = my_key1;
            random_key2_perthread[tid] = my_key2;
        }
#else
        mvm(productVector, resultVector);
#endif
    }

    /**
     * Mixed precision 4-bit Matrix and 32-bit Vector - Matrix Vector Multiplication
     * This method will multiply the current matrix with the product vector and store
     * the result into the result vector. It provides parallel version.
     *
     * @param productVector 32-bit vector that multiplies the current matrix
     * @param resultVector  32-bit vector that represents the result of the multiplication
     */
    inline void mvm_parallel(const CloverVector32 &productVector, CloverVector32 &resultVector)
    {
#if defined(_OPENMP)

        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }

        _Pragma("omp parallel") {
            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            const int8_t * u0       = values;
            const float * v         = productVector.getData();
            float * r               = resultVector.getData();
            const uint64_t h_blocks = cols >> 6;


            uint64_t rows_per_thread = (rows - 1) / nt + 1;
            uint64_t start = 64 * rows_per_thread * tid;
            uint64_t end = std::min(rows, start + 64 * rows_per_thread);

            for (uint64_t i = start; i < end; i += 1) {
                const int8_t *u = u0 + ((i * cols) >> 1);
                const float *su = scales + (i >> 6) * h_blocks;

                __m256 acc_1 = _mm256_setzero_ps();
                __m256 acc_2 = _mm256_setzero_ps();
                __m256 acc_3 = _mm256_setzero_ps();
                __m256 acc_4 = _mm256_setzero_ps();

                for (uint64_t b = 0; b < h_blocks; b += 1)
                {
                    const uint64_t offset0 = b * 64;
                    const uint64_t offset1 = b * 32;

                    const __m256i qu_64 = _mm256_loadu_si256((__m256i *) (u + offset1));

                    const __m256i qu_1 = _mm256_slli_epi32(qu_64, 4 * 6);
                    const __m256i qu_2 = _mm256_slli_epi32(qu_64, 4 * 7);
                    const __m256i qu_3 = _mm256_slli_epi32(qu_64, 4 * 4);
                    const __m256i qu_4 = _mm256_slli_epi32(qu_64, 4 * 5);
                    const __m256i qu_5 = _mm256_slli_epi32(qu_64, 4 * 2);
                    const __m256i qu_6 = _mm256_slli_epi32(qu_64, 4 * 3);
                    const __m256i qu_7 = _mm256_slli_epi32(qu_64, 4 * 0);
                    const __m256i qu_8 = _mm256_slli_epi32(qu_64, 4 * 1);

                    const float su_ss = su[b] / 7.0f;
                    const __m256 scale = _mm256_set1_ps(su_ss);

                    __m256i q_1 = _mm256_srai_epi32(qu_1, 28);
                    __m256i q_2 = _mm256_srai_epi32(qu_2, 28);
                    __m256i q_3 = _mm256_srai_epi32(qu_3, 28);
                    __m256i q_4 = _mm256_srai_epi32(qu_4, 28);
                    __m256i q_5 = _mm256_srai_epi32(qu_5, 28);
                    __m256i q_6 = _mm256_srai_epi32(qu_6, 28);
                    __m256i q_7 = _mm256_srai_epi32(qu_7, 28);
                    __m256i q_8 = _mm256_srai_epi32(qu_8, 28);

                    _mm256_transpose8_epi32(q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8);

                    const __m256 fu_1 = _mm256_cvtepi32_ps(q_1);
                    const __m256 fu_2 = _mm256_cvtepi32_ps(q_2);
                    const __m256 fu_3 = _mm256_cvtepi32_ps(q_3);
                    const __m256 fu_4 = _mm256_cvtepi32_ps(q_4);
                    const __m256 fu_5 = _mm256_cvtepi32_ps(q_5);
                    const __m256 fu_6 = _mm256_cvtepi32_ps(q_6);
                    const __m256 fu_7 = _mm256_cvtepi32_ps(q_7);
                    const __m256 fu_8 = _mm256_cvtepi32_ps(q_8);

                    const __m256  v_1 = _mm256_loadu_ps(v + offset0 +  0);
                    const __m256  v_2 = _mm256_loadu_ps(v + offset0 +  8);
                    const __m256  v_3 = _mm256_loadu_ps(v + offset0 + 16);
                    const __m256  v_4 = _mm256_loadu_ps(v + offset0 + 24);
                    const __m256  v_5 = _mm256_loadu_ps(v + offset0 + 32);
                    const __m256  v_6 = _mm256_loadu_ps(v + offset0 + 40);
                    const __m256  v_7 = _mm256_loadu_ps(v + offset0 + 48);
                    const __m256  v_8 = _mm256_loadu_ps(v + offset0 + 56);

                    const __m256 f_1 = _mm256_mul_ps(fu_1, scale);
                    const __m256 f_2 = _mm256_mul_ps(fu_2, scale);
                    const __m256 f_3 = _mm256_mul_ps(fu_3, scale);
                    const __m256 f_4 = _mm256_mul_ps(fu_4, scale);
                    const __m256 f_5 = _mm256_mul_ps(fu_5, scale);
                    const __m256 f_6 = _mm256_mul_ps(fu_6, scale);
                    const __m256 f_7 = _mm256_mul_ps(fu_7, scale);
                    const __m256 f_8 = _mm256_mul_ps(fu_8, scale);

                    acc_1 = _mm256_fmadd_ps(v_1, f_1, acc_1);
                    acc_2 = _mm256_fmadd_ps(v_2, f_2, acc_2);
                    acc_3 = _mm256_fmadd_ps(v_3, f_3, acc_3);
                    acc_4 = _mm256_fmadd_ps(v_4, f_4, acc_4);

                    acc_1 = _mm256_fmadd_ps(v_5, f_5, acc_1);
                    acc_2 = _mm256_fmadd_ps(v_6, f_6, acc_2);
                    acc_3 = _mm256_fmadd_ps(v_7, f_7, acc_3);
                    acc_4 = _mm256_fmadd_ps(v_8, f_8, acc_4);
                }

                const __m256 sum_1 = _mm256_add_ps(acc_1, acc_2);
                const __m256 sum_2 = _mm256_add_ps(acc_3, acc_4);
                const __m256 sum_3 = _mm256_add_ps(sum_1, sum_2);

                r[i] = _mm256_haddf32_ps(sum_3);
            }
        }
#endif
    }


    inline void transpose_parallel(CloverMatrix4 &other)
    {
#if defined(_OPENMP)

        int8_t * u = values;
        int8_t * v = other.values;

        const uint32_t h_stride = (uint32_t)(cols >> 1);
        const uint32_t v_stride = (uint32_t)(rows >> 1);

        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        register  const __m256i bit_masks = _mm256_setr_epi32 (
                0x000000F0, 0x0000000F, 0x0000F000, 0x00000F00,
                0x00F00000, 0x000F0000, 0xF0000000, 0x0F000000
        );

        register const __m256i l_masks_0 = _mm256_setr_epi32 ( 0,  4,  0,  0,  0,  0,  0,  0);
        register const __m256i l_masks_1 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  0,  0);
        register const __m256i l_masks_2 = _mm256_setr_epi32 ( 8, 12,  0,  4,  0,  0,  0,  0);
        register const __m256i l_masks_3 = _mm256_setr_epi32 ( 4,  8,  0,  0,  0,  0,  0,  0);
        register const __m256i l_masks_4 = _mm256_setr_epi32 (16, 20,  8, 12,  0,  4,  0,  0);
        register const __m256i l_masks_5 = _mm256_setr_epi32 (12, 16,  4,  8,  0,  0,  0,  0);
        register const __m256i l_masks_6 = _mm256_setr_epi32 (24, 28, 16, 20,  8, 12,  0,  4);
        register const __m256i l_masks_7 = _mm256_setr_epi32 (20, 24, 12, 16,  4,  8,  0,  0);

        register const __m256i r_masks_0 = _mm256_setr_epi32 ( 0,  0,  8,  4, 16, 12, 24, 20);
        register const __m256i r_masks_1 = _mm256_setr_epi32 ( 4,  0, 12,  8, 20, 16, 28, 24);
        register const __m256i r_masks_2 = _mm256_setr_epi32 ( 0,  0,  0,  0,  8,  4, 16, 12);
        register const __m256i r_masks_3 = _mm256_setr_epi32 ( 0,  0,  4,  0, 12,  8, 20, 16);
        register const __m256i r_masks_4 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  8,  4);
        register const __m256i r_masks_5 = _mm256_setr_epi32 ( 0,  0,  0,  0,  4,  0, 12,  8);
        register const __m256i r_masks_6 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  0,  0);
        register const __m256i r_masks_7 = _mm256_setr_epi32 ( 0,  0,  0,  0,  0,  0,  4,  0);

        _Pragma("omp parallel") {

            uint32_t scatter[8];

            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            uint64_t n_rowblocks = (rows - 1) / 8 + 1;
            uint64_t rowblocks_per_thread = (n_rowblocks - 1) / nt + 1;
            uint64_t start = 8 * rowblocks_per_thread * tid;
            uint64_t end = std::min(rows, start + 8 * rowblocks_per_thread);

            for(uint64_t i = start; i < end; i += 8)
            {
                for(uint64_t j = 0; j < cols; j += 8)
                {
                    int8_t * src = u + ((i * cols + j) >> 1);
                    int8_t * dst = v + ((j * rows + i) >> 1);

                    __m256i S0 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 0) );
                    __m256i S1 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 1) );
                    __m256i S2 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 2) );
                    __m256i S3 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 3) );
                    __m256i S4 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 4) );
                    __m256i S5 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 5) );
                    __m256i S6 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 6) );
                    __m256i S7 = (__m256i) _mm256_broadcast_ss ( (float *) (src + h_stride * 7) );

                    const __m256i M0 = _mm256_and_si256(S0, bit_masks);
                    const __m256i M1 = _mm256_and_si256(S1, bit_masks);
                    const __m256i M2 = _mm256_and_si256(S2, bit_masks);
                    const __m256i M3 = _mm256_and_si256(S3, bit_masks);
                    const __m256i M4 = _mm256_and_si256(S4, bit_masks);
                    const __m256i M5 = _mm256_and_si256(S5, bit_masks);
                    const __m256i M6 = _mm256_and_si256(S6, bit_masks);
                    const __m256i M7 = _mm256_and_si256(S7, bit_masks);

                    const __m256i L0 = _mm256_sllv_epi32(M0, l_masks_0);
                    const __m256i L1 = _mm256_sllv_epi32(M1, l_masks_1);
                    const __m256i L2 = _mm256_sllv_epi32(M2, l_masks_2);
                    const __m256i L3 = _mm256_sllv_epi32(M3, l_masks_3);
                    const __m256i L4 = _mm256_sllv_epi32(M4, l_masks_4);
                    const __m256i L5 = _mm256_sllv_epi32(M5, l_masks_5);
                    const __m256i L6 = _mm256_sllv_epi32(M6, l_masks_6);
                    const __m256i L7 = _mm256_sllv_epi32(M7, l_masks_7);

                    const __m256i R0 = _mm256_srlv_epi32(L0, r_masks_0);
                    const __m256i R1 = _mm256_srlv_epi32(L1, r_masks_1);
                    const __m256i R2 = _mm256_srlv_epi32(L2, r_masks_2);
                    const __m256i R3 = _mm256_srlv_epi32(L3, r_masks_3);
                    const __m256i R4 = _mm256_srlv_epi32(L4, r_masks_4);
                    const __m256i R5 = _mm256_srlv_epi32(L5, r_masks_5);
                    const __m256i R6 = _mm256_srlv_epi32(L6, r_masks_6);
                    const __m256i R7 = _mm256_srlv_epi32(L7, r_masks_7);

                    const __m256i T0 = _mm256_or_si256(R0, R1);
                    const __m256i T1 = _mm256_or_si256(R2, R3);
                    const __m256i T2 = _mm256_or_si256(R4, R5);
                    const __m256i T3 = _mm256_or_si256(R6, R7);
                    const __m256i T4 = _mm256_or_si256(T0, T1);
                    const __m256i T5 = _mm256_or_si256(T2, T3);
                    const __m256i T6 = _mm256_or_si256(T4, T5);

                    _mm256_storeu_si256((__m256i *) scatter, T6);
                    *(uint32_t *)(dst + v_stride * 0) = scatter[0];
                    *(uint32_t *)(dst + v_stride * 1) = scatter[1];
                    *(uint32_t *)(dst + v_stride * 2) = scatter[2];
                    *(uint32_t *)(dst + v_stride * 3) = scatter[3];
                    *(uint32_t *)(dst + v_stride * 4) = scatter[4];
                    *(uint32_t *)(dst + v_stride * 5) = scatter[5];
                    *(uint32_t *)(dst + v_stride * 6) = scatter[6];
                    *(uint32_t *)(dst + v_stride * 7) = scatter[7];

                }
            }
        }

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );

        //
        // Transpose the scales using MKL
        //
        // mkl_somatcopy ('R', 'T', v_blocks, h_blocks, 1, scales, h_blocks, other.scales, v_blocks);
#else
        transpose(other);
#endif
    }

    // ===============================================================================================================
    // = End of parallelized and vectorized methods
    // ===============================================================================================================

    // ===============================================================================================================
    // = Experimental
    // ===============================================================================================================

    inline void transpose_scalar_faster(CloverMatrix4 &other)
    {
        int8_t * u = values;
        int8_t * v = other.values;

        const uint64_t h_shift = cols >> 1;
        const uint64_t v_shift = rows >> 1;

        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        for(uint64_t i = 0; i < rows; i += 8)
        {
            for(uint64_t j = 0; j < cols; j += 8)
            {
                int8_t * src = u + ((i * cols + j) >> 1);
                int8_t * dst = v + ((j * rows + i) >> 1);

                const uint32_t R0 = *(uint32_t *)(src + h_shift * 0);
                const uint32_t R1 = *(uint32_t *)(src + h_shift * 1);
                const uint32_t R2 = *(uint32_t *)(src + h_shift * 2);
                const uint32_t R3 = *(uint32_t *)(src + h_shift * 3);
                const uint32_t R4 = *(uint32_t *)(src + h_shift * 4);
                const uint32_t R5 = *(uint32_t *)(src + h_shift * 5);
                const uint32_t R6 = *(uint32_t *)(src + h_shift * 6);
                const uint32_t R7 = *(uint32_t *)(src + h_shift * 7);
                //
                // Extract R0
                //
                uint32_t x00 = (R0 >>  0) & 0x000000F0;
                uint32_t x10 = (R0 <<  4) & 0x000000F0;
                uint32_t x20 = (R0 >>  8) & 0x000000F0;
                uint32_t x30 = (R0 >>  4) & 0x000000F0;
                uint32_t x40 = (R0 >> 16) & 0x000000F0;
                uint32_t x50 = (R0 >> 12) & 0x000000F0;
                uint32_t x60 = (R0 >> 24) & 0x000000F0;
                uint32_t x70 = (R0 >> 20) & 0x000000F0;
                //
                // Extract R1
                //
                uint32_t x01 = (R1 >>  4) & 0x0000000F;
                uint32_t x11 = (R1 >>  0) & 0x0000000F;
                uint32_t x21 = (R1 >> 12) & 0x0000000F;
                uint32_t x31 = (R1 >>  8) & 0x0000000F;
                uint32_t x41 = (R1 >> 20) & 0x0000000F;
                uint32_t x51 = (R1 >> 16) & 0x0000000F;
                uint32_t x61 = (R1 >> 28) & 0x0000000F;
                uint32_t x71 = (R1 >> 24) & 0x0000000F;
                //
                // Extract R2
                //
                uint32_t x02 = (R2 <<  8) & 0x0000F000;
                uint32_t x12 = (R2 << 12) & 0x0000F000;
                uint32_t x22 = (R2 >>  0) & 0x0000F000;
                uint32_t x32 = (R2 <<  4) & 0x0000F000;
                uint32_t x42 = (R2 >>  8) & 0x0000F000;
                uint32_t x52 = (R2 >>  4) & 0x0000F000;
                uint32_t x62 = (R2 >> 16) & 0x0000F000;
                uint32_t x72 = (R2 >> 12) & 0x0000F000;
                //
                // Extract R3
                //
                uint32_t x03 = (R3 <<  4) & 0x00000F00;
                uint32_t x13 = (R3 <<  8) & 0x00000F00;
                uint32_t x23 = (R3 >>  4) & 0x00000F00;
                uint32_t x33 = (R3 >>  0) & 0x00000F00;
                uint32_t x43 = (R3 >> 12) & 0x00000F00;
                uint32_t x53 = (R3 >>  8) & 0x00000F00;
                uint32_t x63 = (R3 >> 20) & 0x00000F00;
                uint32_t x73 = (R3 >> 16) & 0x00000F00;
                //
                // Extract R4
                //
                uint32_t x04 = (R4 << 16) & 0x00F00000;
                uint32_t x14 = (R4 << 20) & 0x00F00000;
                uint32_t x24 = (R4 <<  8) & 0x00F00000;
                uint32_t x34 = (R4 << 12) & 0x00F00000;
                uint32_t x44 = (R4 >>  0) & 0x00F00000;
                uint32_t x54 = (R4 <<  4) & 0x00F00000;
                uint32_t x64 = (R4 >>  8) & 0x00F00000;
                uint32_t x74 = (R4 >>  4) & 0x00F00000;
                //
                // Extract R5
                //
                uint32_t x05 = (R5 << 12) & 0x000F0000;
                uint32_t x15 = (R5 << 16) & 0x000F0000;
                uint32_t x25 = (R5 <<  4) & 0x000F0000;
                uint32_t x35 = (R5 <<  8) & 0x000F0000;
                uint32_t x45 = (R5 >>  4) & 0x000F0000;
                uint32_t x55 = (R5 >>  0) & 0x000F0000;
                uint32_t x65 = (R5 >> 12) & 0x000F0000;
                uint32_t x75 = (R5 >>  8) & 0x000F0000;
                //
                // Extract R6
                //
                uint32_t x06 = (R6 << 24) & 0xF0000000;
                uint32_t x16 = (R6 << 28) & 0xF0000000;
                uint32_t x26 = (R6 << 16) & 0xF0000000;
                uint32_t x36 = (R6 << 20) & 0xF0000000;
                uint32_t x46 = (R6 <<  8) & 0xF0000000;
                uint32_t x56 = (R6 << 12) & 0xF0000000;
                uint32_t x66 = (R6 >>  0) & 0xF0000000;
                uint32_t x76 = (R6 <<  4) & 0xF0000000;
                //
                // Extract R7
                //
                uint32_t x07 = (R7 << 20) & 0x0F000000;
                uint32_t x17 = (R7 << 24) & 0x0F000000;
                uint32_t x27 = (R7 << 12) & 0x0F000000;
                uint32_t x37 = (R7 << 16) & 0x0F000000;
                uint32_t x47 = (R7 <<  4) & 0x0F000000;
                uint32_t x57 = (R7 <<  8) & 0x0F000000;
                uint32_t x67 = (R7 >>  4) & 0x0F000000;
                uint32_t x77 = (R7 >>  0) & 0x0F000000;

                const uint32_t T0 = x00 | x01 | x02 | x03 | x04 | x05 | x06 | x07;
                const uint32_t T1 = x10 | x11 | x12 | x13 | x14 | x15 | x16 | x17;
                const uint32_t T2 = x20 | x21 | x22 | x23 | x24 | x25 | x26 | x27;
                const uint32_t T3 = x30 | x31 | x32 | x33 | x34 | x35 | x36 | x37;
                const uint32_t T4 = x40 | x41 | x42 | x43 | x44 | x45 | x46 | x47;
                const uint32_t T5 = x50 | x51 | x52 | x53 | x54 | x55 | x56 | x57;
                const uint32_t T6 = x60 | x61 | x62 | x63 | x64 | x65 | x66 | x67;
                const uint32_t T7 = x70 | x71 | x72 | x73 | x74 | x75 | x76 | x77;

                *(uint32_t *)(dst + v_shift * 0) = T0;
                *(uint32_t *)(dst + v_shift * 1) = T1;
                *(uint32_t *)(dst + v_shift * 2) = T2;
                *(uint32_t *)(dst + v_shift * 3) = T3;
                *(uint32_t *)(dst + v_shift * 4) = T4;
                *(uint32_t *)(dst + v_shift * 5) = T5;
                *(uint32_t *)(dst + v_shift * 6) = T6;
                *(uint32_t *)(dst + v_shift * 7) = T7;
            }
        }

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );

        //
        // Transpose the scales using MKL
        //
        // mkl_somatcopy ('R', 'T', v_blocks, h_blocks, 1, scales, h_blocks, other.scales, v_blocks);
    }
};

#endif /* CLOVER_MATRIX4_H */
