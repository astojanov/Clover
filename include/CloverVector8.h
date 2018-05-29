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

#ifndef CLOVER_VECTOR8_H
#define CLOVER_VECTOR8_H

#include <iostream>
#include <cmath>
#include <bitset>

#include "CloverVector.h"
#include "CloverVector32.h"

class CloverVector8 : public CloverVector {

private:
    bool memoryManagement;


protected:
    int8_t * values;
    float * scales;

    void allocate ()
    {
        uint64_t blocks     = length_pad / CLOVER_VECTOR_BLOCK;
        uint64_t blocks_pad = blocks % CLOVER_VECTOR_BLOCK ? blocks + CLOVER_VECTOR_BLOCK - (blocks % CLOVER_VECTOR_BLOCK) : blocks;

        uint64_t value_bytes = length_pad * sizeof(int8_t);
        uint64_t scale_bytes = blocks_pad * sizeof(float);

        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes + scale_bytes);
        if (ret == 0)
        {
            scales = (float *) (values + value_bytes);
            const uint64_t psize1 = length_pad / 64;
            //
            // Make sure that the value padding is zeroed-out
            //
            for (uint64_t i = length; i < length_pad; i += 1) {
                values[i] = 0;
            }
            //
            // Make sure that the scales are set to 1.0
            //
            for (uint64_t i = length / 64; i < psize1; i += 1) {
                scales[i] = 1;
            }
            //
            // Setup the memory management
            //
            memoryManagement = true;
        } else {
            std::cout << "Could not allocate memory for CloverVector8. Exiting ..." << std::endl;
            exit(1);
        }
    }


public:

    CloverVector8 (uint64_t s, int8_t * data_values, float * data_scales) : CloverVector(s)
    {
        values = data_values;
        scales = data_scales;
        memoryManagement = false;
    }

    CloverVector8 (uint64_t s) : CloverVector(s)
    {
        allocate();
    }

    CloverVector8 (const CloverVector32 &other) : CloverVector(other.size())
    {
        allocate();
        quantize(other);
    }

    CloverVector8 (const CloverVector8& other): CloverVector(other.length)
    {
        allocate();

        const uint64_t blocks      = length_pad / 64;
        const uint64_t value_bytes = length_pad * sizeof(int8_t);
        const uint64_t scale_bytes = blocks     * sizeof(float);

        memcpy(values, other.values, value_bytes);
        memcpy(scales, other.scales, scale_bytes);
    }

    uint64_t getBitsLength () const {
        return 8;
    }

    inline int8_t * getData () const
    {
        return values;
    }

    inline float * getScales () const
    {
        return scales;
    }

    uint64_t getBytes () const
    {
        const uint64_t blocks      = length_pad / 64;
        const uint64_t value_bytes = length_pad * sizeof(int8_t);
        const uint64_t scale_bytes = blocks     * sizeof(float);

        return value_bytes + scale_bytes;
    }

    inline float get(uint64_t i) const
    {
        return values[i] * scales[i >> 6] / 127.0f;
    }

    inline float getAbs(uint64_t i) const
    {
        Restorator result;
        result.f = values[i] * scales[i >> 6] / 127.0f;
        result.i = result.i & 0x7FFFFFFF;
        return result.f;
    }

    inline void set(uint64_t i, float v) const
    {
        float rcp_scale = 127.0f / scales[i >> 6];
        values[i] = (int8_t)roundf(rcp_scale * v);
    }

    inline int8_t getBits(uint64_t i)
    {
        return values[i];
    }

    inline void setBits(uint64_t i, int8_t bits)
    {
        values[i] = bits;
    }

    ~CloverVector8()
    {
        if (memoryManagement) {
            free(values);
        }
    }

    std::string inline toString () const
    {
        std::stringstream sout;

        const int8_t * u = values;
        const uint64_t n = length_pad;

        for (uint64_t i = 0; i < n; i += 1)
        {
            const int8_t q1  = u[i];
            const float s    = scales[i >> 6];
            const float val1 = q1 * s / 127.0f;

            sout << std::setw(10) << i;
            sout << " | ";;
            sout << std::setw(20) << std::fixed << std::setprecision(7) << val1;
            sout << " | ";
            sout << float2hex(val1);
            sout << " | ";
            sout << std::setw(20) << std::fixed << std::setprecision(7) << s;
            sout << " | ";
            sout << std::setw(5) << (int) q1;
            sout << " | ";
            sout << std::bitset<8>(*(uint8_t*)&q1);
            sout << std::endl;
        }
        return sout.str();
    }

    /* ============================================================================================================== */
    /* = Scalar Operations                                                                                            */
    /* ============================================================================================================== */

    inline void quantize_scalar(const CloverVector32 &other)
    {
        const uint64_t n      = length_pad;
        const uint64_t blocks = n >> 6;
        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;


        for (uint64_t b = 0; b < blocks; b += 1) {

            //
            // Define the start and end point of the chunk
            //
            const uint64_t offset = b << 6;

            //
            // Get the absolute max of the chunk
            //
            float max = 0;
            for (uint64_t idx = 0; idx < 64; idx += 1) {
                const uint64_t i = idx + offset;
                const float fmax = fabsf(u[i]);
                if (fmax > max) {
                    max = fmax;
                }
            }
            s[b] = max;
            const float scaled_rcp_max = 127.0f / max;

            for (uint64_t idx = 0; idx < 64; idx += 1) {

                const uint64_t i = offset + idx;
#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                const float rnd_noise1 = 0;
#else
                const float rnd_noise1 = get_random_float();
#endif
                const float    f_value = u[i];
                const int8_t   u_value_sgn = (int8_t) 1 + ((int8_t) (*(int32_t *) &f_value >> 31) << 1);
                const uint32_t u_value_abs = clover_1st_bit_off_32 & *(uint32_t *) &f_value;
                const float    f_value_abs = *(float *) &u_value_abs;
                const int8_t   q_value_abs = (int8_t) floorf(_mm_fmadd_ss(f_value_abs, scaled_rcp_max, rnd_noise1));
                const int8_t   q_value = q_value_abs * u_value_sgn;

                r[i] = q_value;
            }
        }
    }

    inline void restore_scalar(CloverVector32 &other) const
    {
        const int8_t *u   = values;
        const float * su  = scales;
        float  *r         = other.getData();
        uint64_t n        = length_pad;

        for (uint64_t i = 0; i < n; i += 1) {
            const float scale = su[i >> 6] / 127.0f;
            r[i] = scale * u[i];
        }
    }

    inline float dot_scalar(const CloverVector8 &other) const
    {
        const uint64_t n      = length_pad;
        const uint64_t blocks = n >> 6;
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;

        float result = 0;

        for (uint64_t b = 0; b < blocks; b += 1) {
            //
            // Calculate the offset
            //
            const uint64_t offset = b << 6;
            //
            // Setup the scale
            //
            const float su_scale = su[b] / 127.0f;
            const float sv_scale = sv[b] / 127.0f;
            const float scale = su_scale * sv_scale;
            //
            // The dot product of a whole block can fit nicely into 32-bit register: we
            // have 64 numbers all in the range [-127^2, +127^2] i.e. [-127^2 * 64, +127^2 * 64]
            // which is [-1032256, 1032256] and can easily fit into 32-bit.
            //
            int32_t block_dot_product = 0;

            for (uint64_t idx = 0; idx < 64; idx += 1)
            {
                const uint64_t i = offset + idx;
                const int32_t qu = (int32_t) u[i];
                const int32_t qv = (int32_t) v[i];
                block_dot_product += qu * qv;
            }

            result += block_dot_product * scale;
        }

        return result;
    }

    void inline scaleAndAdd_scalar (const CloverVector8 &other, float a)
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t blocks = length_pad / 64;

        scaleAndAdd_scalar (u, v, a, su, sv, blocks, u, su);
    }

    void inline scaleAndAdd_scalar (const CloverVector8 &other, float a, CloverVector8 &result)
    {
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        int8_t * r      = result.values;
        float  * sr     = result.scales;

        scaleAndAdd_scalar (u, v, a, su, sv, blocks, r, sr);
    }


    void inline scaleAndAdd_scalar (const int8_t * u, const int8_t * v, const float a, const float * su, const float * sv, const uint64_t blocks, int8_t * r, float * sr)
    {
        for (uint64_t b = 0; b < blocks; b += 1) {
            //
            // Calculate the offset
            //
            const uint64_t offset = b << 6;
            //
            // Setup the scale
            //
            const float su_scale = su[b] / 127.0f;
            const float sv_scale = sv[b] * a / 127.0f;

            float max = 0;
            float block_values[64];

            for (uint64_t idx = 0; idx < 64; idx += 1) {
                const uint64_t i = offset + idx;
                const float fu = u[i] * su_scale;
                const float fv = v[i];
                const float value = _mm_fmadd_ss(fv, sv_scale, fu);

                block_values[idx] = value;
                const float fmax = fabsf(value);
                if (fmax > max) {
                    max = fmax;
                }
            }

            sr[b] = max;
            const float scaled_rcp_max = 127.0f / max;

            for (uint64_t idx = 0; idx < 64; idx += 1) {
                const uint64_t i = offset + idx;
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


    /* ============================================================================================================== */
    /* = Vector Operations                                                                                            */
    /* ============================================================================================================== */

    inline void quantize(const CloverVector32 &other)
    {

        const uint64_t n0     = other.size_pad();
        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;

        const uint64_t blocks = n0 / 64;

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            const uint64_t offset = b << 6;
            const float * u1 = u + offset;
            const float * u2 = u1 + 64;

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
            //
            // Find the maximum
            //
            const __m256 m1 = _mm256_max_ps(u_abs_1, u_abs_2);
            const __m256 m2 = _mm256_max_ps(u_abs_3, u_abs_4);
            const __m256 m3 = _mm256_max_ps(u_abs_5, u_abs_6);
            const __m256 m4 = _mm256_max_ps(u_abs_7, u_abs_8);
            const __m256 m5 = _mm256_max_ps(m1, m2);
            const __m256 m6 = _mm256_max_ps(m3, m4);
            const __m256 m7 = _mm256_max_ps(m5, m6);

            //
            // Perform horizontal reduction, and make sure that the max is broadcasted in
            // all slots of the 256 bit lane
            //
            const __m256 hmax_5 = _mm256_hmax_ps(m7);

            //
            // Normalize if max is zero
            //
            const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_5, _mm256_setzero_si256());
            const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
            const __m256  hmax_6 = _mm256_add_ps(cndOne, hmax_5);

            //
            // Finally we have the scale
            //
            const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_6);
            //
            // Store the scale to the right place
            //
            _mm_store_ss(scales + b, _mm256_castps256_ps128(hmax_6));

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

            const __m256 rnd_1 = _mm256_setzero_ps();
            const __m256 rnd_2 = _mm256_setzero_ps();
            const __m256 rnd_3 = _mm256_setzero_ps();
            const __m256 rnd_4 = _mm256_setzero_ps();
            const __m256 rnd_5 = _mm256_setzero_ps();
            const __m256 rnd_6 = _mm256_setzero_ps();
            const __m256 rnd_7 = _mm256_setzero_ps();
            const __m256 rnd_8 = _mm256_setzero_ps();
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
            _mm_prefetch((char *)(u2 + 16), _MM_HINT_T0);
            _mm_prefetch((char *)(u2 + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(u2 + 48), _MM_HINT_T0);
            _mm_prefetch((char *)(u2 + 64), _MM_HINT_T0);


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

            _mm256_storeu_si256((__m256i *)(r + offset +  0), pack8_lo);
            _mm256_storeu_si256((__m256i *)(r + offset + 32), pack8_hi);
        }
    }

    inline void quantize_parallel(const CloverVector32 &other)
    {
#if defined(_OPENMP)
        const uint64_t n0     = other.size_pad();
        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;
        const uint64_t blocks = n0 / 64;

        _Pragma("omp parallel") {
            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            __m256i my_key1 = random_key1_perthread[tid];
            __m256i my_key2 = random_key2_perthread[tid];

            uint64_t blocks_per_thread = (blocks - 1) / nt + 1;
            uint64_t start = blocks_per_thread * tid;
            uint64_t end = std::min(blocks, start + blocks_per_thread);

            for (uint64_t b = start; b < end; b += 1)
            {
                const uint64_t offset = b << 6;
                const float * u1 = u + offset;
                const float * u2 = u1 + 64;

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
                //
                // Find the maximum
                //
                const __m256 m1 = _mm256_max_ps(u_abs_1, u_abs_2);
                const __m256 m2 = _mm256_max_ps(u_abs_3, u_abs_4);
                const __m256 m3 = _mm256_max_ps(u_abs_5, u_abs_6);
                const __m256 m4 = _mm256_max_ps(u_abs_7, u_abs_8);
                const __m256 m5 = _mm256_max_ps(m1, m2);
                const __m256 m6 = _mm256_max_ps(m3, m4);
                const __m256 m7 = _mm256_max_ps(m5, m6);

                //
                // Perform horizontal reduction, and make sure that the max is broadcasted in
                // all slots of the 256 bit lane
                //
                const __m256 hmax_5 = _mm256_hmax_ps(m7);

                //
                // Normalize if max is zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_5, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_6 = _mm256_add_ps(cndOne, hmax_5);

                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_6);
                //
                // Store the scale to the right place
                //
                _mm_store_ss(scales + b, _mm256_castps256_ps128(hmax_6));

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

                const __m256 rnd_1 = _mm256_setzero_ps();
                const __m256 rnd_2 = _mm256_setzero_ps();
                const __m256 rnd_3 = _mm256_setzero_ps();
                const __m256 rnd_4 = _mm256_setzero_ps();
                const __m256 rnd_5 = _mm256_setzero_ps();
                const __m256 rnd_6 = _mm256_setzero_ps();
                const __m256 rnd_7 = _mm256_setzero_ps();
                const __m256 rnd_8 = _mm256_setzero_ps();
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
                _mm_prefetch((char *)(u2 + 16), _MM_HINT_T0);
                _mm_prefetch((char *)(u2 + 32), _MM_HINT_T0);
                _mm_prefetch((char *)(u2 + 48), _MM_HINT_T0);
                _mm_prefetch((char *)(u2 + 64), _MM_HINT_T0);


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

                _mm256_storeu_si256((__m256i *)(r + offset +  0), pack8_lo);
                _mm256_storeu_si256((__m256i *)(r + offset + 32), pack8_hi);
            }
        }
#else
        quantize(other);
#endif
    }

    inline void restore(CloverVector32 &other) const
    {
        const int8_t * u      = values;
        const float * su      = scales;
        float  * r            = other.getData();
        const uint64_t n      = length_pad;
        const uint64_t blocks = length_pad / 64;

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            const uint64_t offset0   = b * 64;
            const uint64_t offset1   = b * 64 + 32;

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

            const __m256 scale = _mm256_set1_ps(su[b] / 127.0f);

            const __m256i q_1 = _mm256_srai_epi32(qu_1, 24);
            const __m256i q_2 = _mm256_srai_epi32(qu_2, 24);
            const __m256i q_3 = _mm256_srai_epi32(qu_3, 24);
            const __m256i q_4 = _mm256_srai_epi32(qu_4, 24);
            const __m256i q_5 = _mm256_srai_epi32(qu_5, 24);
            const __m256i q_6 = _mm256_srai_epi32(qu_6, 24);
            const __m256i q_7 = _mm256_srai_epi32(qu_7, 24);
            const __m256i q_8 = _mm256_srai_epi32(qu_8, 24);

            const __m256 fu_1 = _mm256_cvtepi32_ps(q_1);
            const __m256 fu_2 = _mm256_cvtepi32_ps(q_2);
            const __m256 fu_3 = _mm256_cvtepi32_ps(q_3);
            const __m256 fu_4 = _mm256_cvtepi32_ps(q_4);
            const __m256 fu_5 = _mm256_cvtepi32_ps(q_5);
            const __m256 fu_6 = _mm256_cvtepi32_ps(q_6);
            const __m256 fu_7 = _mm256_cvtepi32_ps(q_7);
            const __m256 fu_8 = _mm256_cvtepi32_ps(q_8);

            const __m256 f_1 = _mm256_mul_ps(fu_1, scale);
            const __m256 f_2 = _mm256_mul_ps(fu_2, scale);
            const __m256 f_3 = _mm256_mul_ps(fu_3, scale);
            const __m256 f_4 = _mm256_mul_ps(fu_4, scale);
            const __m256 f_5 = _mm256_mul_ps(fu_5, scale);
            const __m256 f_6 = _mm256_mul_ps(fu_6, scale);
            const __m256 f_7 = _mm256_mul_ps(fu_7, scale);
            const __m256 f_8 = _mm256_mul_ps(fu_8, scale);

            _mm256_store_ps(r + offset0 +  0, f_1);
            _mm256_store_ps(r + offset0 +  8, f_2);
            _mm256_store_ps(r + offset0 + 16, f_3);
            _mm256_store_ps(r + offset0 + 24, f_4);
            _mm256_store_ps(r + offset0 + 32, f_5);
            _mm256_store_ps(r + offset0 + 40, f_6);
            _mm256_store_ps(r + offset0 + 48, f_7);
            _mm256_store_ps(r + offset0 + 56, f_8);
        }
    }

    inline float dot (CloverVector8 const &other) const
    {
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        __m256 dot_product_acc = _mm256_setzero_ps();

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            const uint64_t offset0   = b * 64;
            const uint64_t offset1   = b * 64 + 32;
            const uint64_t offset2   = b * 64 + 64;

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
            dot_product_acc = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc);
        }
        return _mm256_haddf32_ps(dot_product_acc);
    }

    inline float dot_parallel (CloverVector8 const &other) const
    {
#if defined(_OPENMP)
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        float sum = 0.0;
        _Pragma("omp parallel reduction (+:sum)") {
            const uint64_t nt = omp_get_num_threads();
            const uint64_t tid = omp_get_thread_num();

            const uint64_t blocks_per_thread = (blocks - 1) / nt + 1;
            const uint64_t start = blocks_per_thread * tid;
            const uint64_t end = std::min(start + blocks_per_thread, blocks);

            __m256 dot_product_acc = _mm256_setzero_ps();

            for (uint64_t b = start; b < end; b += 1)
            {
                const uint64_t offset0   = b * 64;
                const uint64_t offset1   = b * 64 + 32;
                const uint64_t offset2   = b * 64 + 64;

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
                dot_product_acc = _mm256_fmadd_ps(scale, dot_f32, dot_product_acc);
            }
            sum = _mm256_haddf32_ps(dot_product_acc);
        }
        return sum;
#else
        return dot(other);
#endif
    }


    void inline scaleAndAdd (const CloverVector8 &other, float a)
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t blocks = length_pad / 64;

        scaleAndAdd (u, v, a, su, sv, blocks, u, su);
    }

    void inline scaleAndAdd (const CloverVector8 &other, float a, CloverVector8 &result)
    {
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        int8_t * r      = result.values;
        float  * sr     = result.scales;

        scaleAndAdd (u, v, a, su, sv, blocks, r, sr);
    }


    void inline scaleAndAdd (const int8_t * u, const int8_t * v, const float a, const float * su, const float * sv, const uint64_t blocks, int8_t * r, float * sr)
    {

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            const float su_ss = su[b];
            const float sv_ss = sv[b] * a;

            const uint64_t offset0 = b * 64;
            const uint64_t offset1 = b * 64 + 32;

            const __m256i qu_64_lo = _mm256_loadu_si256( (__m256i *) (u + offset0) );
            const __m256i qu_64_hi = _mm256_loadu_si256( (__m256i *) (u + offset1) );
            const __m256i qv_64_lo = _mm256_loadu_si256( (__m256i *) (v + offset0) );
            const __m256i qv_64_hi = _mm256_loadu_si256( (__m256i *) (v + offset1) );

            __m256i qu_1 = _mm256_slli_epi32(qu_64_lo, 8 * 3);
            __m256i qu_2 = _mm256_slli_epi32(qu_64_lo, 8 * 2);
            __m256i qu_3 = _mm256_slli_epi32(qu_64_lo, 8 * 1);
            __m256i qu_4 = _mm256_slli_epi32(qu_64_lo, 8 * 0);
            __m256i qu_5 = _mm256_slli_epi32(qu_64_hi, 8 * 3);
            __m256i qu_6 = _mm256_slli_epi32(qu_64_hi, 8 * 2);
            __m256i qu_7 = _mm256_slli_epi32(qu_64_hi, 8 * 1);
            __m256i qu_8 = _mm256_slli_epi32(qu_64_hi, 8 * 0);

            qu_1 = _mm256_srai_epi32(qu_1, 24);
            qu_2 = _mm256_srai_epi32(qu_2, 24);
            qu_3 = _mm256_srai_epi32(qu_3, 24);
            qu_4 = _mm256_srai_epi32(qu_4, 24);
            qu_5 = _mm256_srai_epi32(qu_5, 24);
            qu_6 = _mm256_srai_epi32(qu_6, 24);
            qu_7 = _mm256_srai_epi32(qu_7, 24);
            qu_8 = _mm256_srai_epi32(qu_8, 24);

            __m256i qv_1 = _mm256_slli_epi32(qv_64_lo, 8 * 3);
            __m256i qv_2 = _mm256_slli_epi32(qv_64_lo, 8 * 2);
            __m256i qv_3 = _mm256_slli_epi32(qv_64_lo, 8 * 1);
            __m256i qv_4 = _mm256_slli_epi32(qv_64_lo, 8 * 0);
            __m256i qv_5 = _mm256_slli_epi32(qv_64_hi, 8 * 3);
            __m256i qv_6 = _mm256_slli_epi32(qv_64_hi, 8 * 2);
            __m256i qv_7 = _mm256_slli_epi32(qv_64_hi, 8 * 1);
            __m256i qv_8 = _mm256_slli_epi32(qv_64_hi, 8 * 0);

            qv_1 = _mm256_srai_epi32(qv_1, 24);
            qv_2 = _mm256_srai_epi32(qv_2, 24);
            qv_3 = _mm256_srai_epi32(qv_3, 24);
            qv_4 = _mm256_srai_epi32(qv_4, 24);
            qv_5 = _mm256_srai_epi32(qv_5, 24);
            qv_6 = _mm256_srai_epi32(qv_6, 24);
            qv_7 = _mm256_srai_epi32(qv_7, 24);
            qv_8 = _mm256_srai_epi32(qv_8, 24);

            //
            // Time to start prefetching
            //
            _mm_prefetch((char *)(u + offset0 + 64), _MM_HINT_T0);
            _mm_prefetch((char *)(v + offset0 + 64), _MM_HINT_T0);

            const __m256 su_ps = _mm256_set1_ps(su_ss / 127.0f);
            const __m256 sv_ps = _mm256_set1_ps(sv_ss / 127.0f);

            //
            // Convert to 32-bit floating point
            //
            const __m256 fu_1 = _mm256_cvtepi32_ps(qu_1);
            const __m256 fu_2 = _mm256_cvtepi32_ps(qu_2);
            const __m256 fu_3 = _mm256_cvtepi32_ps(qu_3);
            const __m256 fu_4 = _mm256_cvtepi32_ps(qu_4);
            const __m256 fu_5 = _mm256_cvtepi32_ps(qu_5);
            const __m256 fu_6 = _mm256_cvtepi32_ps(qu_6);
            const __m256 fu_7 = _mm256_cvtepi32_ps(qu_7);
            const __m256 fu_8 = _mm256_cvtepi32_ps(qu_8);

            const __m256 fv_1 = _mm256_cvtepi32_ps(qv_1);
            const __m256 fv_2 = _mm256_cvtepi32_ps(qv_2);
            const __m256 fv_3 = _mm256_cvtepi32_ps(qv_3);
            const __m256 fv_4 = _mm256_cvtepi32_ps(qv_4);
            const __m256 fv_5 = _mm256_cvtepi32_ps(qv_5);
            const __m256 fv_6 = _mm256_cvtepi32_ps(qv_6);
            const __m256 fv_7 = _mm256_cvtepi32_ps(qv_7);
            const __m256 fv_8 = _mm256_cvtepi32_ps(qv_8);
            //
            // Multiply u-values with the sv scale
            //
            const __m256 du_1 = _mm256_mul_ps(fu_1, su_ps);
            const __m256 du_2 = _mm256_mul_ps(fu_2, su_ps);
            const __m256 du_3 = _mm256_mul_ps(fu_3, su_ps);
            const __m256 du_4 = _mm256_mul_ps(fu_4, su_ps);
            const __m256 du_5 = _mm256_mul_ps(fu_5, su_ps);
            const __m256 du_6 = _mm256_mul_ps(fu_6, su_ps);
            const __m256 du_7 = _mm256_mul_ps(fu_7, su_ps);
            const __m256 du_8 = _mm256_mul_ps(fu_8, su_ps);
            //
            // Multiply v-values with the sv scale and add the scaled u-values
            // At this point in time, we have calculated the scale-and-add
            //
            const __m256 u_1  = _mm256_fmadd_ps(fv_1, sv_ps, du_1);
            const __m256 u_2  = _mm256_fmadd_ps(fv_2, sv_ps, du_2);
            const __m256 u_3  = _mm256_fmadd_ps(fv_3, sv_ps, du_3);
            const __m256 u_4  = _mm256_fmadd_ps(fv_4, sv_ps, du_4);
            const __m256 u_5  = _mm256_fmadd_ps(fv_5, sv_ps, du_5);
            const __m256 u_6  = _mm256_fmadd_ps(fv_6, sv_ps, du_6);
            const __m256 u_7  = _mm256_fmadd_ps(fv_7, sv_ps, du_7);
            const __m256 u_8  = _mm256_fmadd_ps(fv_8, sv_ps, du_8);
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
            const __m256 max_1 = _mm256_max_ps(u_abs_1, u_abs_2);
            const __m256 max_2 = _mm256_max_ps(u_abs_3, u_abs_4);
            const __m256 max_3 = _mm256_max_ps(u_abs_5, u_abs_6);
            const __m256 max_4 = _mm256_max_ps(u_abs_7, u_abs_8);

            const __m256 max_5 = _mm256_max_ps(max_1, max_2);
            const __m256 max_6 = _mm256_max_ps(max_3, max_4);
            const __m256 max_7 = _mm256_max_ps(max_5, max_6);

            //
            // Perform horizontal reduction, and make sure that the max is broadcasted in
            // all slots of the 256 bit lane
            //
            const __m256 hmax_0 = _mm256_hmax_ps(max_7);

            //
            // Avoid zero
            //
            const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_0, _mm256_setzero_si256());
            const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
            const __m256  hmax_1 = _mm256_add_ps(cndOne, hmax_0);

            //
            // Finally we have the scale
            //
            const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_1);
            _mm256_maskstore_ps(sr + b, clover_mm256_mask_1st_epi32, hmax_1);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED

            const __m256 rnd_1 = _mm256_setzero_ps();
            const __m256 rnd_2 = _mm256_setzero_ps();
            const __m256 rnd_3 = _mm256_setzero_ps();
            const __m256 rnd_4 = _mm256_setzero_ps();
            const __m256 rnd_5 = _mm256_setzero_ps();
            const __m256 rnd_6 = _mm256_setzero_ps();
            const __m256 rnd_7 = _mm256_setzero_ps();
            const __m256 rnd_8 = _mm256_setzero_ps();
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

            //
            // Done, now start packing
            //
            q_1 = _mm256_slli_epi32(q_1, 24);
            q_2 = _mm256_slli_epi32(q_2, 24);
            q_3 = _mm256_slli_epi32(q_3, 24);
            q_4 = _mm256_slli_epi32(q_4, 24);
            q_5 = _mm256_slli_epi32(q_5, 24);
            q_6 = _mm256_slli_epi32(q_6, 24);
            q_7 = _mm256_slli_epi32(q_7, 24);
            q_8 = _mm256_slli_epi32(q_8, 24);

            q_1 = _mm256_srli_epi32(q_1, 3 * 8);
            q_2 = _mm256_srli_epi32(q_2, 2 * 8);
            q_3 = _mm256_srli_epi32(q_3, 1 * 8);
            q_4 = _mm256_srli_epi32(q_4, 0 * 8);
            q_5 = _mm256_srli_epi32(q_5, 3 * 8);
            q_6 = _mm256_srli_epi32(q_6, 2 * 8);
            q_7 = _mm256_srli_epi32(q_7, 1 * 8);
            q_8 = _mm256_srli_epi32(q_8, 0 * 8);

            //
            // Mixing all together
            //
            const __m256i q_12 = _mm256_or_si256(q_1, q_2);
            const __m256i q_34 = _mm256_or_si256(q_3, q_4);
            const __m256i q_lo = _mm256_or_si256(q_12, q_34);

            const __m256i q_56 = _mm256_or_si256(q_5, q_6);
            const __m256i q_78 = _mm256_or_si256(q_7, q_8);
            const __m256i q_hi = _mm256_or_si256(q_56, q_78);

            _mm256_storeu_si256((__m256i *)(r + offset0), q_lo);
            _mm256_storeu_si256((__m256i *)(r + offset1), q_hi);
        }
    }

    void inline scaleAndAdd_parallel (const CloverVector8 &other, float a)
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t blocks = length_pad / 64;

        scaleAndAdd_parallel (u, v, a, su, sv, blocks, u, su);
    }

    void inline scaleAndAdd_parallel (const CloverVector8 &other, float a, CloverVector8 &result)
    {
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        int8_t * r      = result.values;
        float  * sr     = result.scales;

        scaleAndAdd_parallel (u, v, a, su, sv, blocks, r, sr);
    }


    void inline scaleAndAdd_parallel (const int8_t * u, const int8_t * v, const float a, const float * su, const float * sv, const uint64_t blocks, int8_t * r, float * sr)
    {
#if defined(_OPENMP)
        _Pragma("omp parallel")
        {
            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            __m256i my_key1 = random_key1_perthread[tid];
            __m256i my_key2 = random_key2_perthread[tid];

            uint64_t blocks_per_thread = (blocks - 1) / nt + 1;
            uint64_t start = blocks_per_thread * tid;
            uint64_t end = std::min(blocks, start + blocks_per_thread);

            for (uint64_t b = start; b < end; b += 1)
            {
                const float su_ss = su[b];
                const float sv_ss = sv[b] * a;

                const uint64_t offset0 = b * 64;
                const uint64_t offset1 = b * 64 + 32;

                const __m256i qu_64_lo = _mm256_loadu_si256( (__m256i *) (u + offset0) );
                const __m256i qu_64_hi = _mm256_loadu_si256( (__m256i *) (u + offset1) );
                const __m256i qv_64_lo = _mm256_loadu_si256( (__m256i *) (v + offset0) );
                const __m256i qv_64_hi = _mm256_loadu_si256( (__m256i *) (v + offset1) );

                __m256i qu_1 = _mm256_slli_epi32(qu_64_lo, 8 * 3);
                __m256i qu_2 = _mm256_slli_epi32(qu_64_lo, 8 * 2);
                __m256i qu_3 = _mm256_slli_epi32(qu_64_lo, 8 * 1);
                __m256i qu_4 = _mm256_slli_epi32(qu_64_lo, 8 * 0);
                __m256i qu_5 = _mm256_slli_epi32(qu_64_hi, 8 * 3);
                __m256i qu_6 = _mm256_slli_epi32(qu_64_hi, 8 * 2);
                __m256i qu_7 = _mm256_slli_epi32(qu_64_hi, 8 * 1);
                __m256i qu_8 = _mm256_slli_epi32(qu_64_hi, 8 * 0);

                qu_1 = _mm256_srai_epi32(qu_1, 24);
                qu_2 = _mm256_srai_epi32(qu_2, 24);
                qu_3 = _mm256_srai_epi32(qu_3, 24);
                qu_4 = _mm256_srai_epi32(qu_4, 24);
                qu_5 = _mm256_srai_epi32(qu_5, 24);
                qu_6 = _mm256_srai_epi32(qu_6, 24);
                qu_7 = _mm256_srai_epi32(qu_7, 24);
                qu_8 = _mm256_srai_epi32(qu_8, 24);

                __m256i qv_1 = _mm256_slli_epi32(qv_64_lo, 8 * 3);
                __m256i qv_2 = _mm256_slli_epi32(qv_64_lo, 8 * 2);
                __m256i qv_3 = _mm256_slli_epi32(qv_64_lo, 8 * 1);
                __m256i qv_4 = _mm256_slli_epi32(qv_64_lo, 8 * 0);
                __m256i qv_5 = _mm256_slli_epi32(qv_64_hi, 8 * 3);
                __m256i qv_6 = _mm256_slli_epi32(qv_64_hi, 8 * 2);
                __m256i qv_7 = _mm256_slli_epi32(qv_64_hi, 8 * 1);
                __m256i qv_8 = _mm256_slli_epi32(qv_64_hi, 8 * 0);

                qv_1 = _mm256_srai_epi32(qv_1, 24);
                qv_2 = _mm256_srai_epi32(qv_2, 24);
                qv_3 = _mm256_srai_epi32(qv_3, 24);
                qv_4 = _mm256_srai_epi32(qv_4, 24);
                qv_5 = _mm256_srai_epi32(qv_5, 24);
                qv_6 = _mm256_srai_epi32(qv_6, 24);
                qv_7 = _mm256_srai_epi32(qv_7, 24);
                qv_8 = _mm256_srai_epi32(qv_8, 24);

                //
                // Time to start prefetching
                //
                _mm_prefetch((char *)(u + offset0 + 64), _MM_HINT_T0);
                _mm_prefetch((char *)(v + offset0 + 64), _MM_HINT_T0);

                const __m256 su_ps = _mm256_set1_ps(su_ss / 127.0f);
                const __m256 sv_ps = _mm256_set1_ps(sv_ss / 127.0f);

                //
                // Convert to 32-bit floating point
                //
                const __m256 fu_1 = _mm256_cvtepi32_ps(qu_1);
                const __m256 fu_2 = _mm256_cvtepi32_ps(qu_2);
                const __m256 fu_3 = _mm256_cvtepi32_ps(qu_3);
                const __m256 fu_4 = _mm256_cvtepi32_ps(qu_4);
                const __m256 fu_5 = _mm256_cvtepi32_ps(qu_5);
                const __m256 fu_6 = _mm256_cvtepi32_ps(qu_6);
                const __m256 fu_7 = _mm256_cvtepi32_ps(qu_7);
                const __m256 fu_8 = _mm256_cvtepi32_ps(qu_8);

                const __m256 fv_1 = _mm256_cvtepi32_ps(qv_1);
                const __m256 fv_2 = _mm256_cvtepi32_ps(qv_2);
                const __m256 fv_3 = _mm256_cvtepi32_ps(qv_3);
                const __m256 fv_4 = _mm256_cvtepi32_ps(qv_4);
                const __m256 fv_5 = _mm256_cvtepi32_ps(qv_5);
                const __m256 fv_6 = _mm256_cvtepi32_ps(qv_6);
                const __m256 fv_7 = _mm256_cvtepi32_ps(qv_7);
                const __m256 fv_8 = _mm256_cvtepi32_ps(qv_8);
                //
                // Multiply u-values with the sv scale
                //
                const __m256 du_1 = _mm256_mul_ps(fu_1, su_ps);
                const __m256 du_2 = _mm256_mul_ps(fu_2, su_ps);
                const __m256 du_3 = _mm256_mul_ps(fu_3, su_ps);
                const __m256 du_4 = _mm256_mul_ps(fu_4, su_ps);
                const __m256 du_5 = _mm256_mul_ps(fu_5, su_ps);
                const __m256 du_6 = _mm256_mul_ps(fu_6, su_ps);
                const __m256 du_7 = _mm256_mul_ps(fu_7, su_ps);
                const __m256 du_8 = _mm256_mul_ps(fu_8, su_ps);
                //
                // Multiply v-values with the sv scale and add the scaled u-values
                // At this point in time, we have calculated the scale-and-add
                //
                const __m256 u_1  = _mm256_fmadd_ps(fv_1, sv_ps, du_1);
                const __m256 u_2  = _mm256_fmadd_ps(fv_2, sv_ps, du_2);
                const __m256 u_3  = _mm256_fmadd_ps(fv_3, sv_ps, du_3);
                const __m256 u_4  = _mm256_fmadd_ps(fv_4, sv_ps, du_4);
                const __m256 u_5  = _mm256_fmadd_ps(fv_5, sv_ps, du_5);
                const __m256 u_6  = _mm256_fmadd_ps(fv_6, sv_ps, du_6);
                const __m256 u_7  = _mm256_fmadd_ps(fv_7, sv_ps, du_7);
                const __m256 u_8  = _mm256_fmadd_ps(fv_8, sv_ps, du_8);
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
                const __m256 max_1 = _mm256_max_ps(u_abs_1, u_abs_2);
                const __m256 max_2 = _mm256_max_ps(u_abs_3, u_abs_4);
                const __m256 max_3 = _mm256_max_ps(u_abs_5, u_abs_6);
                const __m256 max_4 = _mm256_max_ps(u_abs_7, u_abs_8);

                const __m256 max_5 = _mm256_max_ps(max_1, max_2);
                const __m256 max_6 = _mm256_max_ps(max_3, max_4);
                const __m256 max_7 = _mm256_max_ps(max_5, max_6);

                //
                // Perform horizontal reduction, and make sure that the max is broadcasted in
                // all slots of the 256 bit lane
                //
                const __m256 hmax_0 = _mm256_hmax_ps(max_7);

                //
                // Avoid zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_0, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_1 = _mm256_add_ps(cndOne, hmax_0);

                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_127_ps, hmax_1);
                _mm256_maskstore_ps(sr + b, clover_mm256_mask_1st_epi32, hmax_1);

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
                const __m256 rnd_1 = _mm256_setzero_ps();
                const __m256 rnd_2 = _mm256_setzero_ps();
                const __m256 rnd_3 = _mm256_setzero_ps();
                const __m256 rnd_4 = _mm256_setzero_ps();
                const __m256 rnd_5 = _mm256_setzero_ps();
                const __m256 rnd_6 = _mm256_setzero_ps();
                const __m256 rnd_7 = _mm256_setzero_ps();
                const __m256 rnd_8 = _mm256_setzero_ps();
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
                __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
                __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
                __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
                __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
                __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
                __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
                __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
                __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);

                //
                // Done, now start packing
                //
                q_1 = _mm256_slli_epi32(q_1, 24);
                q_2 = _mm256_slli_epi32(q_2, 24);
                q_3 = _mm256_slli_epi32(q_3, 24);
                q_4 = _mm256_slli_epi32(q_4, 24);
                q_5 = _mm256_slli_epi32(q_5, 24);
                q_6 = _mm256_slli_epi32(q_6, 24);
                q_7 = _mm256_slli_epi32(q_7, 24);
                q_8 = _mm256_slli_epi32(q_8, 24);

                q_1 = _mm256_srli_epi32(q_1, 3 * 8);
                q_2 = _mm256_srli_epi32(q_2, 2 * 8);
                q_3 = _mm256_srli_epi32(q_3, 1 * 8);
                q_4 = _mm256_srli_epi32(q_4, 0 * 8);
                q_5 = _mm256_srli_epi32(q_5, 3 * 8);
                q_6 = _mm256_srli_epi32(q_6, 2 * 8);
                q_7 = _mm256_srli_epi32(q_7, 1 * 8);
                q_8 = _mm256_srli_epi32(q_8, 0 * 8);

                //
                // Mixing all together
                //
                const __m256i q_12 = _mm256_or_si256(q_1, q_2);
                const __m256i q_34 = _mm256_or_si256(q_3, q_4);
                const __m256i q_lo = _mm256_or_si256(q_12, q_34);

                const __m256i q_56 = _mm256_or_si256(q_5, q_6);
                const __m256i q_78 = _mm256_or_si256(q_7, q_8);
                const __m256i q_hi = _mm256_or_si256(q_56, q_78);

                _mm256_storeu_si256((__m256i *)(r + offset0), q_lo);
                _mm256_storeu_si256((__m256i *)(r + offset1), q_hi);
            }

            random_key1_perthread[tid] = my_key1;
            random_key2_perthread[tid] = my_key2;
        }
#else
        scaleAndAdd (u, v, a, su, sv, blocks, r, sr);
#endif
    }


    // ==============================================================================================================
    // = Threshold-ing
    // ==============================================================================================================

    inline void threshold (uint64_t k)
    {
        idx_t * min_heaps = get_min_heaps_mem (k);
        threshold_min_heap(min_heaps, k);
    }

    inline void threshold_parallel (uint64_t k)
    {
        uint64_t nThreads = (uint64_t) get_OpenMP_threads();
        idx_t * min_heaps = get_min_heaps_mem (k * nThreads);
        threshold_min_heap_parallel(min_heaps, k);
    }

    //
    // Perform hard threshold-ing such that only the k highest values will remain
    //
    inline void threshold_min_heap (idx_t * min_heap, uint64_t k)
    {
        const uint64_t n0 = length;
        //
        // Copy the first K-elements
        //
        for (uint64_t i = 0; i < k; i += 1) {
            const float value = getAbs(i);
            const int8_t bits = getBits(i);
            min_heap[i].value = value;
            min_heap[i].bits.i = bits;
            min_heap[i].idx = i;
            setBits(i, 0);
        }
        //
        // Create min-heap O(k) complexity
        //
        std::make_heap(min_heap, min_heap + k, gt_idx_t);
        //
        // Now, swap the min element (root) with the
        // value of the array, only if it is larger
        // then the minimum and call heapify.
        //
        // Complexity: O[(n-k)*log(k)]
        //
        for (uint64_t i = k; i < n0; i += 1)
        {
            const float value = getAbs(i);
            if (value > min_heap[0].value) {
                const int8_t bits = getBits(i);
                min_heap[0].value = value;
                min_heap[0].idx = i;
                min_heap[0].bits.i = bits;
                min_heapify(min_heap, 0, k);
            }
            setBits(i, 0);
        }
        //
        // Only copy the max K elements O(k)
        //
        for (int i = 0; i < k; i += 1) {
            const uint64_t idx = min_heap[i].idx;
            setBits(idx, (int8_t) min_heap[i].bits.i);
        }
    }

    inline void threshold_min_heap_parallel (idx_t * min_heaps, uint64_t k)
    {
#if defined(_OPENMP)
        const uint64_t n0 = length;
        uint64_t nt = (uint64_t) get_OpenMP_threads();
        //
        // Each thread gets their own chunk of the vector and finds the k highest elements
        //
        _Pragma("omp parallel") {
            const uint64_t tid = omp_get_thread_num();

            const uint64_t elems_per_thread = (n0 - 1) / nt + 1;
            const uint64_t start = elems_per_thread * tid;
            const uint64_t end = std::min(start + elems_per_thread, n0);

            idx_t* min_heap = &min_heaps[k * tid];
            //
            // Copy the first K-elements
            //
            for (uint64_t i = start; i < std::min(start + k, end); i += 1) {
                const float value = getAbs(i);
                const int8_t bits = getBits(i);
                min_heap[i - start].value = value;
                min_heap[i - start].bits.i = bits;
                min_heap[i - start].idx = i;
                setBits(i, 0);
            }
            //
            // If there's fewer than K elements, fill the rest of the heap
            //
            for (int64_t i = end; i < start+k; i += 1)
            {
                min_heap[i-start].value = -1.0;
            }

            //
            // Create min-heap O(k) complexity
            //
            std::make_heap(min_heap, min_heap + k, gt_idx_t);
            //
            // Now, swap the min element (root) with the
            // value of the array, only if it is larger
            // then the minimum and call heapify.
            //
            // Complexity: O[(n-k)*log(k)]
            //
            for (uint64_t i = start + k; i < end; i += 1)
            {
                const float value = getAbs(i);
                if (value > min_heap[0].value) {
                    const int8_t bits = getBits(i);
                    min_heap[0].value = value;
                    min_heap[0].idx = i;
                    min_heap[0].bits.i = bits;
                    min_heapify(min_heap, 0, k);
                }
                setBits(i, 0);
            }
            std::sort_heap(min_heap, min_heap + k, gt_idx_t);
        }

        //
        // Only copy the max K elements O(k)
        //
        int indices[nt];
        for(int j = 0; j < nt; j++) { indices[j] = 0;}
        for (int i = 0; i < k; i += 1) {
            //select id of thread
            int best_j = 0;
            float best_j_val = min_heaps[0*k + indices[0]].value;
            for(int j = 1; j < nt; j++) {
                if( min_heaps[j*k + indices[j]].value > best_j_val || best_j_val == -1.0 ){
                    best_j = j;
                    best_j_val = min_heaps[j*k + indices[j]].value;
                }
            }

            const uint64_t idx = min_heaps[best_j*k + indices[best_j]].idx;
            setBits(idx, (int8_t) min_heaps[best_j*k + indices[best_j]].bits.i);

            indices[best_j]++;
        }
#else
        threshold_min_heap(min_heaps, k);
#endif
    }


    inline void clear()
    {
        int8_t * u = values;
        const uint64_t blocks = length_pad / 64;
        //
        // Clear the values and the scales
        //
        for (uint64_t b = 0; b < blocks; b += 1) {
            const uint64_t offset0 = b * 64;
            const uint64_t offset1 = b * 64 + 32;
            _mm256_storeu_si256((__m256i *)(u + offset0), _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(u + offset1), _mm256_setzero_si256());
            scales[b] = 1.0f;
        }
    }

};



#endif /* CLOVER_VECTOR8_H */
