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

#ifndef CLOVER_MATRIX8_H
#define CLOVER_MATRIX8_H

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "CloverMatrix.h"
#include "CloverMatrix32.h"
#include "CloverVector8.h"


/**
 *
 *  CloverMatrix8 is a quantized matrix that contains M x N values.
 *  It is stored in a row-major order. One scalar value is being used
 *  for 4096 values, corresponding to one block of 64 x 64 elements,
 *  as illustrated bellow:
 *
 *              h_block
 *             64x 8-bit
 *            --------------------------  .......  --------------
 *            |           |            |           |            |
 *   v_block  | scales[0] | scales[1]  |           |            |
 *  64x 8-bit |           |            |           |            |
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
class CloverMatrix8 : public CloverMatrix {

protected:
    int8_t * values;
    float  * scales;

    void allocate()
    {
        uint64_t length   = rows * cols;
        uint64_t h_blocks = rows >> 6;
        uint64_t v_blocks = cols >> 6;

        uint64_t value_bytes = length % 2 == 0 ? length : length + 1;
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

    CloverMatrix8 (uint64_t h, uint64_t w) : CloverMatrix(h, w)
    {
        allocate();
    }

    uint64_t getBitsLength () const {
        return 8;
    }

    inline uint64_t getBytes () const
    {
        uint64_t length   = rows * cols;
        uint64_t v_blocks = rows >> 6;
        uint64_t h_blocks = cols >> 6;

        uint64_t value_bytes = length % 2 == 0 ? length : length + 1;
        uint64_t scale_bytes = h_blocks * v_blocks * sizeof(float);

        return value_bytes + scale_bytes;
    }

    inline float get(uint64_t i, uint64_t j) const
    {
        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;

        const uint64_t b_i = i >> 6;
        const uint64_t b_j = j >> 6;

        const float scale = scales[b_i * h_blocks + b_j] / 127.0f;
        const int8_t qval = values[i * cols + j];

        return scale * (float) qval;
    }


    void quantize_scalar(const CloverMatrix32 &m)
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
                const float scaled_rcp_max = 127.0f / max;
                //
                // Perform the quantization
                //
                for (uint64_t i = 0; i < 64; i += 1) {
                    for (uint64_t j = 0; j < 64; j += 1) {
                        const uint64_t idx = block_offset + i * cols + j;
#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                        const float rnd_noise1 = 0;
#else
                        const float rnd_noise1 = get_random_float();
#endif
                        const float    f_value = u[idx];
                        const int8_t   u_value_sgn = (int8_t) 1 + ((int8_t) (*(int32_t *) &f_value >> 31) << 1);
                        const uint32_t u_value_abs = clover_1st_bit_off_32 & *(uint32_t *) &f_value;
                        const float    f_value_abs = *(float *) &u_value_abs;
                        const int8_t   q_value_abs = (int8_t) floorf(
                                _mm_fmadd_ss(f_value_abs, scaled_rcp_max, rnd_noise1));
                        const int8_t   q_value = q_value_abs * u_value_sgn;

                        r[idx] = q_value;
                    }
                }
            }
        }
    }

    void quantize(const CloverMatrix32 &m)
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
                const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_7);

                //
                // Store the scale to the right place
                //
                _mm256_maskstore_ps(scales + block_index, clover_mm256_mask_1st_epi32, hmax_7);

                //
                // Get the starting position of the resulting memory space:
                //
                int8_t * r0 = values + block_offset;

                for (uint64_t i = 0; i < 64; i += 1) {

                    const float * u1 = u0 + i * cols;
                    int8_t * r = r0 + i * cols;

                    //
                    // Reload stuff again
                    //
                    const __m256 u_1 = _mm256_loadu_ps(u1 +  0);
                    const __m256 u_2 = _mm256_loadu_ps(u1 +  8);
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
                    _mm_prefetch((char *)(u0 + 64), _MM_HINT_T0);

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
                    _mm_prefetch((char *)(u0 + 64), _MM_HINT_T0);


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
        }
    }

    void mvm_scalar(const CloverVector8 &productVector, CloverVector8 &resultVector)
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
                CloverVector8 rowVector(cols, values + row_offset, scales + row_scales);
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

            sr[b_i] = max;
            const float scaled_rcp_max = 127.0f / max;

            for (uint64_t idx = 0; idx < 64; idx += 1) {

                const uint64_t i = (b_i << 6) + idx;
#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                const float rnd_noise1 = 0;
#else
                const float rnd_noise1 = get_random_float();
#endif
                const float    f_value = block_values[idx];
                const int8_t   u_value_sgn = (int8_t) 1 + ((int8_t) (*(int32_t *) &f_value >> 31) << 1);
                const uint32_t u_value_abs = clover_1st_bit_off_32 & *(uint32_t *) &f_value;
                const float    f_value_abs = *(float *) &u_value_abs;
                const int8_t   q_value_abs = (int8_t) floorf(_mm_fmadd_ss(f_value_abs, scaled_rcp_max, rnd_noise1));
                const int8_t   q_value = q_value_abs * u_value_sgn;

                r[i] = q_value;
            }
        }
    }

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

            const int8_t * u = u0 + (i * cols);
            const float * su = scales + (i >> 6) * h_blocks;

            __m256 acc_1 = _mm256_setzero_ps();
            __m256 acc_2 = _mm256_setzero_ps();
            __m256 acc_3 = _mm256_setzero_ps();
            __m256 acc_4 = _mm256_setzero_ps();

            for (uint64_t b = 0; b < h_blocks; b += 1) {

                const uint64_t offset0   = b * 64;
                const uint64_t offset1   = b * 64 + 32;

                const __m256  scale   = _mm256_set1_ps(su[b] / 127.0f);
                const __m256i qu_32_1 = _mm256_loadu_si256( (__m256i *) (u + offset0) );
                const __m256i qu_32_2 = _mm256_loadu_si256( (__m256i *) (u + offset1) );

                const __m256i qu_32_1_switch = _mm256_permute2f128_si256(qu_32_1, qu_32_1, 0x21);
                const __m256i qu_32_2_switch = _mm256_permute2f128_si256(qu_32_2, qu_32_2, 0x21);

                const __m256i qu_32_1_half_1 = _mm256_shuffle_epi8(qu_32_1       , clover_mm256_8bit_restore_perm_lo);
                const __m256i qu_32_1_half_2 = _mm256_shuffle_epi8(qu_32_1_switch, clover_mm256_8bit_restore_perm_hi);
                const __m256i qu_32_2_half_1 = _mm256_shuffle_epi8(qu_32_2       , clover_mm256_8bit_restore_perm_lo);
                const __m256i qu_32_2_half_2 = _mm256_shuffle_epi8(qu_32_2_switch, clover_mm256_8bit_restore_perm_hi);

                const __m256i q_32_1 = _mm256_or_si256(qu_32_1_half_1, qu_32_1_half_2);
                const __m256i q_32_2 = _mm256_or_si256(qu_32_2_half_1, qu_32_2_half_2);

                const __m256i qu_1 = _mm256_slli_epi32(q_32_1, 8 * 3);
                const __m256i qu_2 = _mm256_slli_epi32(q_32_1, 8 * 2);
                const __m256i qu_3 = _mm256_slli_epi32(q_32_1, 8 * 1);
                const __m256i qu_4 = _mm256_slli_epi32(q_32_1, 8 * 0);
                const __m256i qu_5 = _mm256_slli_epi32(q_32_2, 8 * 3);
                const __m256i qu_6 = _mm256_slli_epi32(q_32_2, 8 * 2);
                const __m256i qu_7 = _mm256_slli_epi32(q_32_2, 8 * 1);
                const __m256i qu_8 = _mm256_slli_epi32(q_32_2, 8 * 0);

                const __m256i q_1 = _mm256_srai_epi32(qu_1, 24);
                const __m256i q_2 = _mm256_srai_epi32(qu_2, 24);
                const __m256i q_3 = _mm256_srai_epi32(qu_3, 24);
                const __m256i q_4 = _mm256_srai_epi32(qu_4, 24);
                const __m256i q_5 = _mm256_srai_epi32(qu_5, 24);
                const __m256i q_6 = _mm256_srai_epi32(qu_6, 24);
                const __m256i q_7 = _mm256_srai_epi32(qu_7, 24);
                const __m256i q_8 = _mm256_srai_epi32(qu_8, 24);

                const __m256  v_1 = _mm256_loadu_ps(v + offset0 +  0);
                const __m256  v_2 = _mm256_loadu_ps(v + offset0 +  8);
                const __m256  v_3 = _mm256_loadu_ps(v + offset0 + 16);
                const __m256  v_4 = _mm256_loadu_ps(v + offset0 + 24);
                const __m256  v_5 = _mm256_loadu_ps(v + offset0 + 32);
                const __m256  v_6 = _mm256_loadu_ps(v + offset0 + 40);
                const __m256  v_7 = _mm256_loadu_ps(v + offset0 + 48);
                const __m256  v_8 = _mm256_loadu_ps(v + offset0 + 56);

                const __m256 f_1 = _mm256_cvtepi32_ps(q_1);
                const __m256 f_2 = _mm256_cvtepi32_ps(q_2);
                const __m256 f_3 = _mm256_cvtepi32_ps(q_3);
                const __m256 f_4 = _mm256_cvtepi32_ps(q_4);
                const __m256 f_5 = _mm256_cvtepi32_ps(q_5);
                const __m256 f_6 = _mm256_cvtepi32_ps(q_6);
                const __m256 f_7 = _mm256_cvtepi32_ps(q_7);
                const __m256 f_8 = _mm256_cvtepi32_ps(q_8);

                const __m256 t_1 = _mm256_mul_ps(v_1, scale);
                const __m256 t_2 = _mm256_mul_ps(v_2, scale);
                const __m256 t_3 = _mm256_mul_ps(v_3, scale);
                const __m256 t_4 = _mm256_mul_ps(v_4, scale);
                const __m256 t_5 = _mm256_mul_ps(v_5, scale);
                const __m256 t_6 = _mm256_mul_ps(v_6, scale);
                const __m256 t_7 = _mm256_mul_ps(v_7, scale);
                const __m256 t_8 = _mm256_mul_ps(v_8, scale);

                acc_1 = _mm256_fmadd_ps(t_1, f_1, acc_1);
                acc_2 = _mm256_fmadd_ps(t_2, f_2, acc_2);
                acc_3 = _mm256_fmadd_ps(t_3, f_3, acc_3);
                acc_4 = _mm256_fmadd_ps(t_4, f_4, acc_4);

                acc_1 = _mm256_fmadd_ps(t_5, f_5, acc_1);
                acc_2 = _mm256_fmadd_ps(t_6, f_6, acc_2);
                acc_3 = _mm256_fmadd_ps(t_7, f_7, acc_3);
                acc_4 = _mm256_fmadd_ps(t_8, f_8, acc_4);
            }

            const __m256 sum_1 = _mm256_add_ps(acc_1, acc_2);
            const __m256 sum_2 = _mm256_add_ps(acc_3, acc_4);
            const __m256 sum_3 = _mm256_add_ps(sum_1, sum_2);

            r[i] = _mm256_haddf32_ps(sum_3);
        }
    }


    void mvm_parallel(const CloverVector8 &productVector, CloverVector8 &resultVector)
    {
#if defined(_OPENMP)
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }
        _Pragma("omp parallel") {
            const int8_t * u0       = values;
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
                // We process 64 values at a time
                //
                float block_values[64];

                //
                // 128-bit SSE variable to keep the max element
                //
                __m128 max_ss = _mm_setzero_ps();

                for (uint64_t idx = 0; idx < 64; idx +=1) {

                    const int8_t * u = u0 + (row + idx) * cols;
                    const float * su = scales + (row >> 6) * h_blocks;

                    __m256 dot_product_acc_1 = _mm256_setzero_ps();

                    for (uint64_t b = 0; b < h_blocks; b += 1)
                    {
                        const uint64_t offset0   = b * 64;
                        const uint64_t offset1   = offset0 + 32;
                        const uint64_t offset2   = offset1 + 64;

                        const __m256i qu_1 = _mm256_loadu_si256( (__m256i *) (u + offset0) );
                        const __m256i qu_2 = _mm256_loadu_si256( (__m256i *) (u + offset1) );
                        const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset0) );
                        const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset1) );

                        const __m256 su_scale = _mm256_broadcast_ss(su + b);
                        const __m256 sv_scale = _mm256_broadcast_ss(sv + b);

                        const __m256 su_rcp = _mm256_mul_ps(su_scale, clover_mm256_rcp_127_ps);
                        const __m256 sv_rcp = _mm256_mul_ps(sv_scale, clover_mm256_rcp_127_ps);
                        const __m256 scale  = _mm256_mul_ps(su_rcp, sv_rcp);
                        //
                        // Meanwhile keep the pre-fetcher busy
                        //
                        _mm_prefetch((char *)(u + offset2), _MM_HINT_T0);
                        _mm_prefetch((char *)(v + offset2), _MM_HINT_T0);
                        _mm_prefetch((char *)(su + b + 1) , _MM_HINT_T0);
                        _mm_prefetch((char *)(sv + b + 1) , _MM_HINT_T0);
                        //
                        // Get absolute values of u vectors
                        //
                        const __m256i au_1 = _mm256_sign_epi8(qu_1, qu_1);
                        const __m256i au_2 = _mm256_sign_epi8(qu_2, qu_2);
                        //
                        // Sign the values of the v vectors
                        //
                        const __m256i sv_1 = _mm256_sign_epi8(qv_1, qu_1);
                        const __m256i sv_2 = _mm256_sign_epi8(qv_2, qu_2);
                        //
                        // Perform multiplication and create 16-bit values
                        // each value is in the range [-127^2*2, +127^2*2]
                        //
                        const __m256i dot_16_1 = _mm256_maddubs_epi16 (au_1, sv_1);
                        const __m256i dot_16_2 = _mm256_maddubs_epi16 (au_2, sv_2);
                        //
                        // Now, convert to 32-bit values range: [-127^2*4, +127^2*4]
                        //
                        const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
                        const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);
                        const __m256i dot_32   = _mm256_add_epi32(dot_32_1, dot_32_2);
                        //
                        // Convert to float
                        //
                        const __m256  dot_f32  = _mm256_cvtepi32_ps(dot_32);
                        //
                        // Perform dot product on the block
                        //
                        dot_product_acc_1 = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc_1);
                    }

                    const __m128 hadd_0  = _mm256_extractf128_ps(dot_product_acc_1, 1);
                    const __m128 hadd_1 = _mm256_castps256_ps128(dot_product_acc_1);
                    const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                    const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                    const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));

                    //
                    // Store the result at the right place
                    //
                    _mm_store_ss(block_values + idx, hadd_4);
                    //
                    // Now find the maximum
                    //
                    const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                    max_ss = _mm_max_ss(habs, max_ss);
                }

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
                _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);
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
                _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);

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

    void dense_matrix_transpose_times_sparse_vector_parallel(const CloverVector8 &productVector, CloverVector8 &resultVector, int32_t* sparsity, int k)
    {
        const uint64_t h_blocks = (cols) >> 6;

        int8_t * u            = resultVector.getData();
        float * su            = resultVector.getScales();

        for (int index = 0; index < k; index++)
        {
            int row =  sparsity[index];
            const float alpha = productVector.get(row);

            const uint64_t pos = row * cols;
           // const uint64_t idx = pos >> 1;
            int8_t * row_data = &values[pos];

            const uint64_t block = row >> 6;
            float * row_scales = &scales[block * h_blocks];
            
            resultVector.scaleAndAdd_parallel (u, row_data, alpha, su, row_scales, resultVector.size() / 64, u, su);
        }
    }

    void mvm(const CloverVector8 &productVector, CloverVector8 &resultVector)
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }
        const int8_t * u0       = values;
        const int8_t * v        = productVector.getData();
        const float * sv        = productVector.getScales();
        const uint64_t h_blocks = cols >> 6;

        for (uint64_t row = 0; row < rows; row += 64) {
            //
            // We process 64 values at a time
            //
            float block_values[64];

            //
            // 128-bit SSE variable to keep the max element
            //
            __m128 max_ss = _mm_setzero_ps();

            for (uint64_t idx = 0; idx < 64; idx +=1) {

                const int8_t * u = u0 + (row + idx) * cols;
                const float * su = scales + (row >> 6) * h_blocks;

                __m256 dot_product_acc_1 = _mm256_setzero_ps();

                for (uint64_t b = 0; b < h_blocks; b += 1)
                {
                    const uint64_t offset0   = b * 64;
                    const uint64_t offset1   = offset0 + 32;
                    const uint64_t offset2   = offset1 + 64;

                    const __m256i qu_1 = _mm256_loadu_si256( (__m256i *) (u + offset0) );
                    const __m256i qu_2 = _mm256_loadu_si256( (__m256i *) (u + offset1) );
                    const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset0) );
                    const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset1) );

                    const __m256 su_scale = _mm256_broadcast_ss(su + b);
                    const __m256 sv_scale = _mm256_broadcast_ss(sv + b);

                    const __m256 su_rcp = _mm256_mul_ps(su_scale, clover_mm256_rcp_127_ps);
                    const __m256 sv_rcp = _mm256_mul_ps(sv_scale, clover_mm256_rcp_127_ps);
                    const __m256 scale  = _mm256_mul_ps(su_rcp, sv_rcp);
                    //
                    // Meanwhile keep the pre-fetcher busy
                    //
                    _mm_prefetch((char *)(u + offset2), _MM_HINT_T0);
                    _mm_prefetch((char *)(v + offset2), _MM_HINT_T0);
                    _mm_prefetch((char *)(su + b + 1) , _MM_HINT_T0);
                    _mm_prefetch((char *)(sv + b + 1) , _MM_HINT_T0);
                    //
                    // Get absolute values of u vectors
                    //
                    const __m256i au_1 = _mm256_sign_epi8(qu_1, qu_1);
                    const __m256i au_2 = _mm256_sign_epi8(qu_2, qu_2);
                    //
                    // Sign the values of the v vectors
                    //
                    const __m256i sv_1 = _mm256_sign_epi8(qv_1, qu_1);
                    const __m256i sv_2 = _mm256_sign_epi8(qv_2, qu_2);
                    //
                    // Perform multiplication and create 16-bit values
                    // each value is in the range [-127^2*2, +127^2*2]
                    //
                    const __m256i dot_16_1 = _mm256_maddubs_epi16 (au_1, sv_1);
                    const __m256i dot_16_2 = _mm256_maddubs_epi16 (au_2, sv_2);
                    //
                    // Now, convert to 32-bit values range: [-127^2*4, +127^2*4]
                    //
                    const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
                    const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);
                    const __m256i dot_32   = _mm256_add_epi32(dot_32_1, dot_32_2);
                    //
                    // Convert to float
                    //
                    const __m256  dot_f32  = _mm256_cvtepi32_ps(dot_32);
                    //
                    // Perform dot product on the block
                    //
                    dot_product_acc_1 = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc_1);
                }

                const __m128 hadd_0  = _mm256_extractf128_ps(dot_product_acc_1, 1);
                const __m128 hadd_1 = _mm256_castps256_ps128(dot_product_acc_1);
                const __m128 hadd_2 = _mm_add_ps(hadd_0, hadd_1);
                const __m128 hadd_3 = _mm_add_ps(hadd_2, _mm_movehl_ps(hadd_2, hadd_2));
                const __m128 hadd_4 = _mm_add_ss(hadd_3, _mm_shuffle_ps(hadd_3, hadd_3, 0x55));

                //
                // Store the result at the right place
                //
                _mm_store_ss(block_values + idx, hadd_4);
                //
                // Now find the maximum
                //
                const __m128 habs = _mm_and_ps(clover_mm_1st_bit_off_ps, hadd_4);
                max_ss = _mm_max_ss(habs, max_ss);
            }

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
            _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);
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
            _mm_prefetch((char *)(u0 + (row + 64) * cols), _MM_HINT_T0);

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


    inline void restore_scalar(CloverMatrix32 &other) const
    {
        float * r = other.getData();
        for (uint64_t i = 0; i < rows; i += 1) {
            for (uint64_t j = 0; j < cols; i += 1) {
                uint64_t pos = i * cols + j;
                r[pos] = get(i, j);
            }
        }
    }


    inline void transpose_scalar(CloverMatrix8 &other)
    {

        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;
        const int8_t * u = values;
        int8_t * v = other.values;

        for (uint64_t i = 0; i < rows; i += 1) {
            for (uint64_t j = 0; j < cols; j += 1) {
                const uint64_t idx0 = i * cols + j;
                const uint64_t idx1 = j * rows + i;
                v[idx1] = u[idx0];
            }
        }
        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );
    }

    inline void transpose_parallel (CloverMatrix8 &other)
    {
        //
        // Transpose the values using Intel IPP
        //
        // ippSetNumThreads(get_OpenMP_threads());

        IppiSize valuesRoi = { (int) cols, (int) rows  };
        ippiTranspose_8u_C1R ( (Ipp8u*) values, (int) cols, (Ipp8u*) other.values, (int) rows, valuesRoi );

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;
        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );
    }


    inline void transpose(CloverMatrix8 &other)
    {
        //
        // Make sure we are running a single thread
        //
        ippSetNumThreads(1);

        //
        // Transpose the values using Intel IPP
        //
        IppiSize valuesRoi = { (int) cols, (int) rows  };
        ippiTranspose_8u_C1R ( (Ipp8u*) values, (int) cols, (Ipp8u*) other.values, (int) rows, valuesRoi );

        const uint64_t v_blocks = rows >> 6;
        const uint64_t h_blocks = cols >> 6;
        //
        // Transpose the scales using Intel IPP
        //
        IppiSize srcRoi = { (int) h_blocks, (int) v_blocks  };
        ippiTranspose_32f_C1R ( scales, (int) h_blocks * sizeof(float), other.scales, (int) v_blocks * sizeof(float), srcRoi );

        //
        // Get back to the official number of threads
        //
        ippSetNumThreads(get_OpenMP_threads());
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

    ~CloverMatrix8()
    {
        free(values);
    }

};



#endif /* CLOVER_MATRIX8_H */
