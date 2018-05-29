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

#ifndef CLOVER_VECTOR4_H
#define CLOVER_VECTOR4_H

#include <cmath>
#include <bitset>
#include <limits>
#include <vector>

#include "CloverVector.h"
#include "CloverVector32.h"
#include "CloverBase.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../lib/simd_debug.h" // TODO: (Alen Stojanov) This should be removed soon
#include "CloverVector8.h"

/**
 * CloverVector4: 4-bit Vector - Blocked
 *
 * The Blocked version is organized into blocks of CLOVER_VECTOR_BLOCK elements.
 * Each CLOVER_VECTOR_BLOCK elements contain a single float that represents their
 * scale. Elements are stored as 4-bit values in two's complements format.
 *
 * For an element at position pos, the following are given:
 *
 *  - scales[i / CLOVER_VECTOR_BLOCK] defines the scale of the i-th element.
 *  - values[i >> 1] * scales[i / CLOVER_VECTOR_BLOCK] / (2^3 - 1) represents
 *    the value of the vector at a give
 *    n position i.
 *
 */
class CloverVector4 : public CloverVector {

private:
    bool memoryManagement;

protected:
    int8_t * values;
    float  * scales;

    void allocate()
    {
        uint64_t blocks     = length_pad / CLOVER_VECTOR_BLOCK;
        uint64_t blocks_pad = blocks % CLOVER_VECTOR_BLOCK ? blocks + CLOVER_VECTOR_BLOCK - (blocks % CLOVER_VECTOR_BLOCK) : blocks;

        uint64_t value_bytes = length_pad * sizeof(int8_t) / 2;
        uint64_t scale_bytes = blocks_pad * sizeof(float);

        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes + scale_bytes);
        if (ret == 0)
        {
            scales = (float *) (values + value_bytes);

            const uint64_t psize0 = length_pad >> 1;
            const uint64_t psize1 = length_pad / 64;
            //
            // Make sure that the value padding is zeroed-out
            //
            for (uint64_t i = length >> 1; i < psize0; i += 1) {
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
            std::cout << "Could not allocate memory for CloverVector4. Exiting ..." << std::endl;
            exit(1);
        }
    }

public:

    // ===============================================================================================================
    // = Support methods and functions: Constructor / Destructor / Getters / Setters / toString etc.
    // ===============================================================================================================

    //
    // Explicit creation of the 4-bit vector
    //
    CloverVector4(uint64_t s, int8_t * data, float * data_scales) : CloverVector(s)
    {
        values = data;
        scales = data_scales;
        memoryManagement = false;
    }
    //
    // Default constructor: an empty 4-bit array
    //
    CloverVector4(uint64_t s) : CloverVector(s)
    {
        allocate();
    }
    //
    // Construct a 4-bit quantized array given 32-bit floating point vector
    //
    CloverVector4(const CloverVector32 &other) : CloverVector(other.size())
    {
        allocate();
        quantize(other);
    }
    //
    // Copy constructor
    //
    CloverVector4 (const CloverVector4& other): CloverVector(other.length)
    {
        allocate();

        const uint64_t blocks      = length_pad / 64;
        const uint64_t value_bytes = length_pad * sizeof(int8_t) / 2;
        const uint64_t scale_bytes = blocks     * sizeof(float);

        memcpy(values, other.values, value_bytes);
        memcpy(scales, other.scales, scale_bytes);
    }

    uint64_t getBitsLength () const {
        return 4;
    }

    inline int8_t getBits(uint64_t pos)
    {
        const uint64_t idx = pos >> 1;
        int8_t qu_p  = values[idx];

        return  _mm_srai_epi8_ss(qu_p << (pos % 2) * 4, 4);
    }

    inline void setBits(uint64_t pos, int8_t bits)
    {
        const uint64_t idx = pos >> 1;
        const int8_t qu    = (bits & 0x0F) << ((1 - pos % 2) * 4);

        //
        // Clearing out the bad bits before we or them
        // -Tyler
        //
        if( pos % 2 == 0) {
            values[idx] &= 0x0F;
        } else {
            values[idx] &= 0xF0;
        }
        values[idx] |= qu;
    }

    inline float get(uint64_t pos) const
    {
        const uint64_t idx   = pos >> 1;
        const uint64_t block = pos >> 6;

        const float scale = scales[block] / 7.0f;
        const int8_t qu_p = values[idx];

        return scale * (float) _mm_srai_epi8_ss(qu_p << (pos % 2) * 4, 4);
    }

    inline float getAbs(uint64_t pos) const
    {
        Restorator result;

        const uint64_t idx   = pos >> 1;
        const uint64_t block = pos >> 6;

        const float scale = scales[block] / 7.0f;
        const int8_t qu_p = values[idx];

        result.f = scale * (float) _mm_srai_epi8_ss(qu_p << (pos % 2) * 4, 4);
        result.i = result.i & 0x7FFFFFFF;

        return result.f;
    }

    inline void set(uint64_t pos, float value)
    {
        const uint64_t idx   = pos >> 1;
        const uint64_t block = pos >> 6;

        const float rcp_scale = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(scales + block)));
        const float scale = 7.0f * rcp_scale;

        const int8_t qu_p = (int8_t) roundf(value * scale);
        const int8_t qu   = qu_p << ((1 - pos % 2) * 4);

        //
        // Clearing out the bad bits before we or them
        // -Tyler
        //
        if( pos % 2 == 0) {
            values[idx] &= 0x0F;
        } else {
            values[idx] &= 0xF0;
        }
        values[idx] |= qu;
    }

    inline int8_t * getData() const
    {
        return values;
    }

    inline float * getScales() const
    {
        return scales;
    }

    inline uint64_t getBytes() const
    {
        const uint64_t blocks      = length_pad / 64;
        const uint64_t value_bytes = length_pad * sizeof(int8_t) / 2;
        const uint64_t scale_bytes = blocks     * sizeof(float);
        return value_bytes + scale_bytes;
    }

    std::string toString () const
    {
        std::stringstream sout;

        const int8_t * u      = values;
        const float * su      = scales;
        const uint64_t blocks = length_pad / 64;

        for (uint64_t b = 0; b < blocks; b += 1) {
            //
            // Define the start and end point of the chunk
            //
            const uint64_t offset = b * 64;
            //
            // Set the scale
            //
            const float scale = su[b];

            for (uint64_t idx = 0; idx < 64; idx += 2)
            {
                const uint64_t i = (idx + offset) >> 1;

                const int8_t q1 = _mm_srai_epi8_ss(u[i], 4);
                const int8_t q2 = _mm_srai_epi8_ss(u[i] << 4, 4);

                const float val1 = q1 * scale / 7.0f;
                sout << std::setw(10) << 2 * i;
                sout << " | ";;
                sout << std::setw(20) << std::fixed << std::setprecision(7) << val1;
                sout << " | ";
                sout << float2hex(val1);
                sout << " | ";
                sout << std::setw(20) << std::fixed << std::setprecision(7) << scale;
                sout << " | ";
                sout << std::setw(5) << (int) q1;
                sout << " | ";
                sout << std::bitset<8>(u[i]);
                sout << std::endl;


                const float val2 = q2 * scale / 7.0f;
                sout << std::setw(10) << 2 * i + 1;
                sout << " | ";;
                sout << std::setw(20) << std::fixed << std::setprecision(7) << val2;
                sout << " | ";
                sout << float2hex(val2);
                sout << " | ";
                sout << std::setw(20) << std::fixed << std::setprecision(7) << scale;
                sout << " | ";
                sout << std::setw(5) << (int) q2;
                sout << " | ";
                sout << std::bitset<8>(u[i]);
                sout << std::endl;
            }

        }
        return sout.str();
    }

    inline void clear ()
    {
        int8_t * u = values;
        const uint64_t blocks = length_pad / 64;

        //
        // Clear the values and the scales
        //
        for (uint64_t b = 0; b < blocks; b += 1) {
            const uint64_t offset = b * 32;
            _mm256_storeu_si256((__m256i *)(u + offset), _mm256_setzero_si256());
            scales[b] = 1.0f;
        }
    }

    ~CloverVector4()
    {
        if (memoryManagement) {
            free(values);
        }
    }

    // ===============================================================================================================
    // = End of support methods and functions
    // ===============================================================================================================

    // ===============================================================================================================
    // = Scalar methods
    // ===============================================================================================================

    void inline scaleAndAdd_scalar (const CloverVector4 &other, float a)
    {
        int8_t * u            = values;
        float * su            = scales;
        const int8_t * v      = other.values;
        const float * sv      = other.scales;

        scaleAndAdd_scalar(u, su, v, sv, a, u, su);
    }

    void inline scaleAndAdd_scalar (CloverVector4 &other, float a, CloverVector4 &result)
    {
        const int8_t * u  = values;
        const float * su  = scales;
        const int8_t * v  = other.values;
        const float * sv  = other.scales;
        int8_t * r        = result.values;
        float * sr        = result.scales;

        scaleAndAdd_scalar(u, su, v, sv, a, r, sr);
    }


    void inline scaleAndAdd_scalar (const int8_t * u, const float * su, const int8_t * v, const float * sv, float a, int8_t * r,  float * sr)
    {
        const uint64_t n0     = length_pad;
        const uint64_t blocks = n0 / 64;
        float block_values[64];

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            //
            // Define the start and end point of the chunk
            //
            const uint64_t offset = b * 64;
            //
            // Get the scales
            //
            const float su_rcp = su[b] / 7.0f;
            const float sv_rcp = sv[b] * a / 7.0f;
            //
            // Perform de-quantization of the block
            //
            for (uint64_t idx = 0; idx < 64; idx += 2)
            {
                const uint64_t i0 = offset + idx;
                const uint64_t i1 = i0 >> 1;

                const int8_t q_u = u[i1];
                const int8_t q_v = v[i1];

                const int8_t q_u_hi = _mm_srai_epi8_ss(q_u, 4);
                const int8_t q_u_lo = _mm_srai_epi8_ss(q_u << 4, 4);
                const int8_t q_v_hi = _mm_srai_epi8_ss(q_v, 4);
                const int8_t q_v_lo = _mm_srai_epi8_ss(q_v << 4, 4);

                const float f_u_hi = su_rcp * (float) q_u_hi;
                const float f_u_lo = su_rcp * (float) q_u_lo;

                block_values[idx + 0] = _mm_fmadd_ss(q_v_hi, sv_rcp, f_u_hi);
                block_values[idx + 1] = _mm_fmadd_ss(q_v_lo, sv_rcp, f_u_lo);

            }

            //
            // Get the absolute max of the chunk
            //
            float max = 0;
            for (uint64_t idx = 0; idx < 64; idx += 1) {
                const float fmax = fabsf(block_values[idx]);
                if (fmax > max) {
                    max = fmax;
                }
            }
            sr[b] = max;
            const float scaled_rcp_max = 7.0f / max;

            //
            // Perform quantization on the chunk
            //
            for (uint64_t idx = 0; idx < 64; idx += 2)
            {
                const uint64_t i = idx + offset;

                const float u1 = block_values[idx + 0];
                const float u2 = block_values[idx + 1];

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

                r[i >> 1] = q_1 | q_2;
            }
        }
    }


    inline void quantize_scalar(const CloverVector32 &other)
    {
        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;
        const uint64_t n0     = length_pad;
        const uint64_t blocks = n0 / 64;

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            //
            // Define the start and end point of the chunk
            //
            const uint64_t offset = b * 64;

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
            const float scaled_rcp_max = 7.0f / max;

            //
            // Perform quantization on the chunk
            //
            for (uint64_t idx = 0; idx < 64; idx += 2)
            {
                const uint64_t i = idx + offset;

                const float u1 = u[i + 0];
                const float u2 = u[i + 1];

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

                r[i >> 1] = q_1 | q_2;
            }
        }
    }

    inline void restore_scalar(CloverVector32 &other) const
    {
        const int8_t * u      = values;
        float * s             = scales;
        const uint64_t n0     = length_pad;
        const uint64_t blocks = n0 / 64;
        float * r             = other.getData();

        for (uint64_t b = 0; b < blocks; b += 1)
        {
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
            for (uint64_t idx = 0; idx < 64; idx += 2)
            {
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

    inline float dot_scalar(const CloverVector4 &other) const
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t n0     = length_pad;
        const uint64_t blocks = n0 / 64;
        float result = 0;

        float rcp_49 = 1.0f / 49.0f;

        for (uint64_t b = 0; b < blocks; b += 1)
        {
            //
            // Define the start and end point of the chunk
            //
            const uint64_t offset = b * 32;
            int16_t acc = 0;

            for (uint64_t idx = 0; idx < 32; idx += 1)
            {
                const uint64_t i = idx + offset;

                const int8_t qu_p = u[i];
                const int8_t qv_p = v[i];

                const int8_t qu_1 = _mm_srai_epi8_ss(qu_p, 4);
                const int8_t qu_2 = _mm_srai_epi8_ss(qu_p << 4, 4);
                const int8_t qv_1 = _mm_srai_epi8_ss(qv_p, 4);
                const int8_t qv_2 = _mm_srai_epi8_ss(qv_p << 4, 4);

                acc += (int16_t)(qu_1 * qv_1) + (int16_t)(qu_2 * qv_2);
            }

            const float scaled_rcp_ss = (su[b] / 7.0f) * (sv[b] / 7.0f);
            result += scaled_rcp_ss * (float) acc;
        }

        return result;
    }

    // ===============================================================================================================
    // = End of scalar methods
    // ===============================================================================================================

    // ===============================================================================================================
    // = SIMD / Vectorized operations
    // ===============================================================================================================

    inline void quantize (const CloverVector32 &other)
    {
        //assert(getRows() == other.getRows() && getCols() == other.getCols());

        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;
        const uint64_t n0     = other.size_pad();
        const uint64_t blocks = n0 / 64;


        for (uint64_t b = 0; b < blocks; b += 1)
        {
            const uint64_t offset = b * 64;
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
            const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_6);

            //
            // Store the scale to the right place
            //
            _mm256_maskstore_ps(scales + b, clover_mm256_mask_1st_epi32, hmax_6);


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

            _mm256_storeu_si256((__m256i *)(r + (offset >> 1)), t7);
        }
    }

    inline void quantize_parallel (const CloverVector32 &other)
    {
#if defined(_OPENMP)
        const float * u       = other.getData();
        int8_t * r            = values;
        float * s             = scales;
        const uint64_t n0     = other.size_pad();
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
                const uint64_t offset = b * 64;
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
                const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_6);

                //
                // Store the scale to the right place
                //
                _mm256_maskstore_ps(scales + b, clover_mm256_mask_1st_epi32, hmax_6);


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

                _mm256_storeu_si256((__m256i *)(r + (offset >> 1)), t7);
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
        float * r             = other.getData();
        const uint64_t blocks = length_pad / 64;

        for (uint64_t b = 0; b < blocks; b += 1) {

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

            const __m256 f_1 = _mm256_mul_ps(fu_1, scale);
            const __m256 f_2 = _mm256_mul_ps(fu_2, scale);
            const __m256 f_3 = _mm256_mul_ps(fu_3, scale);
            const __m256 f_4 = _mm256_mul_ps(fu_4, scale);
            const __m256 f_5 = _mm256_mul_ps(fu_5, scale);
            const __m256 f_6 = _mm256_mul_ps(fu_6, scale);
            const __m256 f_7 = _mm256_mul_ps(fu_7, scale);
            const __m256 f_8 = _mm256_mul_ps(fu_8, scale);

            float * u1 = r + offset0;

            _mm256_storeu_ps(u1 +  0, f_1);
            _mm256_storeu_ps(u1 +  8, f_2);
            _mm256_storeu_ps(u1 + 16, f_3);
            _mm256_storeu_ps(u1 + 24, f_4);
            _mm256_storeu_ps(u1 + 32, f_5);
            _mm256_storeu_ps(u1 + 40, f_6);
            _mm256_storeu_ps(u1 + 48, f_7);
            _mm256_storeu_ps(u1 + 56, f_8);
        }
    }

    inline float dot(const CloverVector4 &other) const
    {
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        __m256 dot_product_acc_1 = _mm256_setzero_ps();
        __m256 dot_product_acc_2 = _mm256_setzero_ps();

        for (uint64_t b = 0; b < blocks; b += 2)
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
            const __m256i qu_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_2);
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
            // Perform multiplication and create 16-bit values
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

            const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
            const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);

            const __m256  dot_f_1  = _mm256_cvtepi32_ps(dot_32_1);
            const __m256  dot_f_2  = _mm256_cvtepi32_ps(dot_32_2);

            //
            // Perform dot product on the block
            //
            dot_product_acc_1 = _mm256_fmadd_ps(scaled_rcp_1, dot_f_1, dot_product_acc_1);
            dot_product_acc_2 = _mm256_fmadd_ps(scaled_rcp_2, dot_f_2, dot_product_acc_2);
        }

        const __m256 vacc = _mm256_add_ps(dot_product_acc_1, dot_product_acc_2);
        return _mm256_haddf32_ps(vacc);
    }



    void inline scaleAndAdd (const CloverVector4 &other, float a)
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t blocks = length_pad / 64;

        scaleAndAdd (u, v, a, su, sv, blocks, u, su);
    }

    void inline scaleAndAdd (const CloverVector4 &other, float a, CloverVector4 &result)
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
            const uint64_t offset1 = b * 32;

            const __m256i qu_64 = _mm256_loadu_si256( (__m256i *) (u + offset1) );
            const __m256i qv_64 = _mm256_loadu_si256( (__m256i *) (v + offset1) );

            __m256i qu_1 = _mm256_slli_epi32(qu_64, 4 * 7);
            __m256i qu_2 = _mm256_slli_epi32(qu_64, 4 * 6);
            __m256i qu_3 = _mm256_slli_epi32(qu_64, 4 * 5);
            __m256i qu_4 = _mm256_slli_epi32(qu_64, 4 * 4);
            __m256i qu_5 = _mm256_slli_epi32(qu_64, 4 * 3);
            __m256i qu_6 = _mm256_slli_epi32(qu_64, 4 * 2);
            __m256i qu_7 = _mm256_slli_epi32(qu_64, 4 * 1);
            __m256i qu_8 = _mm256_slli_epi32(qu_64, 4 * 0);

            qu_1 = _mm256_srai_epi32(qu_1, 28);
            qu_2 = _mm256_srai_epi32(qu_2, 28);
            qu_3 = _mm256_srai_epi32(qu_3, 28);
            qu_4 = _mm256_srai_epi32(qu_4, 28);
            qu_5 = _mm256_srai_epi32(qu_5, 28);
            qu_6 = _mm256_srai_epi32(qu_6, 28);
            qu_7 = _mm256_srai_epi32(qu_7, 28);
            qu_8 = _mm256_srai_epi32(qu_8, 28);

            __m256i qv_1 = _mm256_slli_epi32(qv_64, 4 * 7);
            __m256i qv_2 = _mm256_slli_epi32(qv_64, 4 * 6);
            __m256i qv_3 = _mm256_slli_epi32(qv_64, 4 * 5);
            __m256i qv_4 = _mm256_slli_epi32(qv_64, 4 * 4);
            __m256i qv_5 = _mm256_slli_epi32(qv_64, 4 * 3);
            __m256i qv_6 = _mm256_slli_epi32(qv_64, 4 * 2);
            __m256i qv_7 = _mm256_slli_epi32(qv_64, 4 * 1);
            __m256i qv_8 = _mm256_slli_epi32(qv_64, 4 * 0);

            qv_1 = _mm256_srai_epi32(qv_1, 28);
            qv_2 = _mm256_srai_epi32(qv_2, 28);
            qv_3 = _mm256_srai_epi32(qv_3, 28);
            qv_4 = _mm256_srai_epi32(qv_4, 28);
            qv_5 = _mm256_srai_epi32(qv_5, 28);
            qv_6 = _mm256_srai_epi32(qv_6, 28);
            qv_7 = _mm256_srai_epi32(qv_7, 28);
            qv_8 = _mm256_srai_epi32(qv_8, 28);

            //
            // Time to start prefetching
            //
            _mm_prefetch((char *)(u + offset0), _MM_HINT_T0);
            _mm_prefetch((char *)(v + offset0), _MM_HINT_T0);
            _mm_prefetch((char *)(u + offset0 + 64), _MM_HINT_T0);
            _mm_prefetch((char *)(v + offset0 + 64), _MM_HINT_T0);

            const __m256 su_ps = _mm256_set1_ps(su_ss / 7.0f);
            const __m256 sv_ps = _mm256_set1_ps(sv_ss / 7.0f);

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

            const __m256 du_1 = _mm256_mul_ps(fu_1, su_ps);
            const __m256 du_2 = _mm256_mul_ps(fu_2, su_ps);
            const __m256 du_3 = _mm256_mul_ps(fu_3, su_ps);
            const __m256 du_4 = _mm256_mul_ps(fu_4, su_ps);
            const __m256 du_5 = _mm256_mul_ps(fu_5, su_ps);
            const __m256 du_6 = _mm256_mul_ps(fu_6, su_ps);
            const __m256 du_7 = _mm256_mul_ps(fu_7, su_ps);
            const __m256 du_8 = _mm256_mul_ps(fu_8, su_ps);

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
            // Avoid zero
            //
            const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_5, _mm256_setzero_si256());
            const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
            const __m256  hmax_6 = _mm256_add_ps(cndOne, hmax_5);

            //
            // Finally we have the scale
            //
            const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_6);
            _mm256_maskstore_ps(sr + b, clover_mm256_mask_1st_epi32, hmax_6);

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

            q_1 = _mm256_slli_epi32(q_1, 28);
            q_2 = _mm256_slli_epi32(q_2, 28);
            q_3 = _mm256_slli_epi32(q_3, 28);
            q_4 = _mm256_slli_epi32(q_4, 28);
            q_5 = _mm256_slli_epi32(q_5, 28);
            q_6 = _mm256_slli_epi32(q_6, 28);
            q_7 = _mm256_slli_epi32(q_7, 28);
            q_8 = _mm256_slli_epi32(q_8, 28);

            q_1 = _mm256_srli_epi32(q_1, 7 * 4);
            q_2 = _mm256_srli_epi32(q_2, 6 * 4);
            q_3 = _mm256_srli_epi32(q_3, 5 * 4);
            q_4 = _mm256_srli_epi32(q_4, 4 * 4);
            q_5 = _mm256_srli_epi32(q_5, 3 * 4);
            q_6 = _mm256_srli_epi32(q_6, 2 * 4);
            q_7 = _mm256_srli_epi32(q_7, 1 * 4);
            q_8 = _mm256_srli_epi32(q_8, 0 * 4);

            const __m256i t1 = _mm256_or_si256(q_1, q_2);
            const __m256i t2 = _mm256_or_si256(q_3, q_4);
            const __m256i t3 = _mm256_or_si256(q_5, q_6);
            const __m256i t4 = _mm256_or_si256(q_7, q_8);
            const __m256i t5 = _mm256_or_si256(t1, t2);
            const __m256i t6 = _mm256_or_si256(t3, t4);
            const __m256i t7 = _mm256_or_si256(t5, t6);

            _mm256_storeu_si256((__m256i *)(r + offset1), t7);
        }
    }

    // ===============================================================================================================
    // = End of SIMD / vectorized methods
    // ===============================================================================================================


    // ===============================================================================================================
    // = Parallel and vectorized methods
    // ===============================================================================================================

    void inline scaleAndAdd_parallel (const CloverVector4 &other, float a)
    {
        int8_t * u            = values;
        int8_t * v            = other.values;
        float * su            = scales;
        float * sv            = other.scales;
        const uint64_t blocks = length_pad / 64;

        scaleAndAdd_parallel (u, v, a, su, sv, blocks, u, su);
    }

    void inline scaleAndAdd_parallel (const CloverVector4 &other, float a, CloverVector4 &result)
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


                const float su_ss = su[b];
                const float sv_ss = sv[b] * a;

                const uint64_t offset0 = b * 64;
                const uint64_t offset1 = b * 32;

                const __m256i qu_64 = _mm256_loadu_si256( (__m256i *) (u + offset1) );
                const __m256i qv_64 = _mm256_loadu_si256( (__m256i *) (v + offset1) );

                __m256i qu_1 = _mm256_slli_epi32(qu_64, 4 * 7);
                __m256i qu_2 = _mm256_slli_epi32(qu_64, 4 * 6);
                __m256i qu_3 = _mm256_slli_epi32(qu_64, 4 * 5);
                __m256i qu_4 = _mm256_slli_epi32(qu_64, 4 * 4);
                __m256i qu_5 = _mm256_slli_epi32(qu_64, 4 * 3);
                __m256i qu_6 = _mm256_slli_epi32(qu_64, 4 * 2);
                __m256i qu_7 = _mm256_slli_epi32(qu_64, 4 * 1);
                __m256i qu_8 = _mm256_slli_epi32(qu_64, 4 * 0);

                qu_1 = _mm256_srai_epi32(qu_1, 28);
                qu_2 = _mm256_srai_epi32(qu_2, 28);
                qu_3 = _mm256_srai_epi32(qu_3, 28);
                qu_4 = _mm256_srai_epi32(qu_4, 28);
                qu_5 = _mm256_srai_epi32(qu_5, 28);
                qu_6 = _mm256_srai_epi32(qu_6, 28);
                qu_7 = _mm256_srai_epi32(qu_7, 28);
                qu_8 = _mm256_srai_epi32(qu_8, 28);

                __m256i qv_1 = _mm256_slli_epi32(qv_64, 4 * 7);
                __m256i qv_2 = _mm256_slli_epi32(qv_64, 4 * 6);
                __m256i qv_3 = _mm256_slli_epi32(qv_64, 4 * 5);
                __m256i qv_4 = _mm256_slli_epi32(qv_64, 4 * 4);
                __m256i qv_5 = _mm256_slli_epi32(qv_64, 4 * 3);
                __m256i qv_6 = _mm256_slli_epi32(qv_64, 4 * 2);
                __m256i qv_7 = _mm256_slli_epi32(qv_64, 4 * 1);
                __m256i qv_8 = _mm256_slli_epi32(qv_64, 4 * 0);

                qv_1 = _mm256_srai_epi32(qv_1, 28);
                qv_2 = _mm256_srai_epi32(qv_2, 28);
                qv_3 = _mm256_srai_epi32(qv_3, 28);
                qv_4 = _mm256_srai_epi32(qv_4, 28);
                qv_5 = _mm256_srai_epi32(qv_5, 28);
                qv_6 = _mm256_srai_epi32(qv_6, 28);
                qv_7 = _mm256_srai_epi32(qv_7, 28);
                qv_8 = _mm256_srai_epi32(qv_8, 28);

                //
                // Time to start prefetching
                //
                _mm_prefetch((char *)(u + offset0), _MM_HINT_T0);
                _mm_prefetch((char *)(v + offset0), _MM_HINT_T0);
                _mm_prefetch((char *)(u + offset0 + 64), _MM_HINT_T0);
                _mm_prefetch((char *)(v + offset0 + 64), _MM_HINT_T0);

                const __m256 su_ps = _mm256_set1_ps(su_ss / 7.0f);
                const __m256 sv_ps = _mm256_set1_ps(sv_ss / 7.0f);

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

                const __m256 du_1 = _mm256_mul_ps(fu_1, su_ps);
                const __m256 du_2 = _mm256_mul_ps(fu_2, su_ps);
                const __m256 du_3 = _mm256_mul_ps(fu_3, su_ps);
                const __m256 du_4 = _mm256_mul_ps(fu_4, su_ps);
                const __m256 du_5 = _mm256_mul_ps(fu_5, su_ps);
                const __m256 du_6 = _mm256_mul_ps(fu_6, su_ps);
                const __m256 du_7 = _mm256_mul_ps(fu_7, su_ps);
                const __m256 du_8 = _mm256_mul_ps(fu_8, su_ps);

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
                // Avoid zero
                //
                const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_5, _mm256_setzero_si256());
                const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
                const __m256  hmax_6 = _mm256_add_ps(cndOne, hmax_5);

                //
                // Finally we have the scale
                //
                const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_6);
                _mm256_maskstore_ps(sr + b, clover_mm256_mask_1st_epi32, hmax_6);

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

                q_1 = _mm256_slli_epi32(q_1, 28);
                q_2 = _mm256_slli_epi32(q_2, 28);
                q_3 = _mm256_slli_epi32(q_3, 28);
                q_4 = _mm256_slli_epi32(q_4, 28);
                q_5 = _mm256_slli_epi32(q_5, 28);
                q_6 = _mm256_slli_epi32(q_6, 28);
                q_7 = _mm256_slli_epi32(q_7, 28);
                q_8 = _mm256_slli_epi32(q_8, 28);

                q_1 = _mm256_srli_epi32(q_1, 7 * 4);
                q_2 = _mm256_srli_epi32(q_2, 6 * 4);
                q_3 = _mm256_srli_epi32(q_3, 5 * 4);
                q_4 = _mm256_srli_epi32(q_4, 4 * 4);
                q_5 = _mm256_srli_epi32(q_5, 3 * 4);
                q_6 = _mm256_srli_epi32(q_6, 2 * 4);
                q_7 = _mm256_srli_epi32(q_7, 1 * 4);
                q_8 = _mm256_srli_epi32(q_8, 0 * 4);

                const __m256i t1 = _mm256_or_si256(q_1, q_2);
                const __m256i t2 = _mm256_or_si256(q_3, q_4);
                const __m256i t3 = _mm256_or_si256(q_5, q_6);
                const __m256i t4 = _mm256_or_si256(q_7, q_8);
                const __m256i t5 = _mm256_or_si256(t1, t2);
                const __m256i t6 = _mm256_or_si256(t3, t4);
                const __m256i t7 = _mm256_or_si256(t5, t6);

                _mm256_storeu_si256((__m256i *)(r + offset1), t7);
            }

            random_key1_perthread[tid] = my_key1;
            random_key2_perthread[tid] = my_key2;
        }
#else
        scaleAndAdd (u, v, a, su, sv, blocks, r, sr);
#endif
    }

    inline float dot_parallel(const CloverVector4 &other) const
    {
#if defined(_OPENMP)
        const int8_t * u      = values;
        const int8_t * v      = other.values;
        const float * su      = scales;
        const float * sv      = other.scales;
        const uint64_t blocks = length_pad / 64;

        float sum = 0.0;
        _Pragma("omp parallel reduction(+:sum)") {
            const uint64_t nt = omp_get_num_threads();
            const uint64_t tid = omp_get_thread_num();

            const uint64_t n_iterations = (blocks - 1) / 2 + 1;
            const uint64_t iter_per_thread = (n_iterations - 1) / nt + 1;
            const uint64_t start = iter_per_thread * tid;
            const uint64_t end = std::min(start + iter_per_thread, n_iterations);

            __m256 dot_product_acc_1 = _mm256_setzero_ps();
            __m256 dot_product_acc_2 = _mm256_setzero_ps();

            for (uint64_t i = start; i < end; i += 1)
            {
                uint64_t b = 2*i;
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
                const __m256i qu_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_2);
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
                // Perform multiplication and create 16-bit values
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

                const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
                const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);

                const __m256  dot_f_1  = _mm256_cvtepi32_ps(dot_32_1);
                const __m256  dot_f_2  = _mm256_cvtepi32_ps(dot_32_2);

                //
                // Perform dot product on the block
                //
                dot_product_acc_1 = _mm256_fmadd_ps(scaled_rcp_1, dot_f_1, dot_product_acc_1);
                dot_product_acc_2 = _mm256_fmadd_ps(scaled_rcp_2, dot_f_2, dot_product_acc_2);
            }

            const __m256 vacc = _mm256_add_ps(dot_product_acc_1, dot_product_acc_2);
            sum = _mm256_haddf32_ps(vacc);
        }
        return sum;
#else
        return dot(other);
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
};


#endif /* CLOVER_VECTOR4_H */
