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

#ifndef CLOVER_VECTOR16_H
#define CLOVER_VECTOR16_H

#include <iostream>
#include "CloverVector.h"
#include "CloverVector32.h"

#if defined(_OPENMP)
#include <omp.h>
#endif


class CloverVector16 : public CloverVector {

private:
    bool memoryManagement;

protected:
    uint16_t * values;

    inline void allocate ()
    {
        uint64_t bytes = length_pad * sizeof(uint16_t);

        const int ret = posix_memalign((void **) &values, get_system_pagesize(), bytes);
        if (ret == 0)
        {
            //
            // Make sure that the value padding is zeroed-out
            //
            for (uint64_t i = length; i < length_pad; i += 1) {
                values[i] = 0;
            }
            memoryManagement = true;
        } else {
            std::cout << "Could not allocate memory for CloverVector16. Exiting ..." << std::endl;
            exit(1);
        }
    }

public:

    CloverVector16 (uint64_t s, uint16_t * data) : CloverVector(s)
    {
        values = data;
        memoryManagement = false;
    }

    CloverVector16 (uint64_t s) : CloverVector(s)
    {
        allocate();
    }

    CloverVector16 (const CloverVector32 &other) : CloverVector(other.size())
    {
        allocate();
        quantize(other);
    }

    CloverVector16 (const CloverVector16& other): CloverVector(other.length)
    {
        allocate();
        const uint64_t value_bytes = length_pad * sizeof(uint16_t);
        memcpy(values, other.values, value_bytes);
    }

    inline float get(uint64_t idx) const
    {
        return _mm_cvtsh_ss(values[idx]);
    }

    inline float getAbs(uint64_t idx) const
    {
        Restorator result;
        result.f = _mm_cvtsh_ss(values[idx]);
        result.i = result.i & 0x7FFFFFFF;
        return result.f;
    }

    inline void set(uint64_t idx, float value)
    {
        values[idx] = _mm_cvtss_sh(value);
    }

    inline uint16_t getBits(uint64_t idx)
    {
        return values[idx];
    }

    inline void setBits(uint64_t idx, uint16_t value)
    {
        values[idx] = value;
    }

    inline uint16_t * getData () const
    {
        return values;
    }

    uint64_t getBytes () const
    {
        return length_pad * sizeof(uint16_t);
    }

    ~CloverVector16()
    {
        if (memoryManagement) {
            free(values);
        }
    }

    std::string toString () const
    {
        CloverVector32 tmp(size());
        restore(tmp);
        return tmp.toString();
    }

    inline void quantize_scalar(const CloverVector32 &other)
    {
        const float * u   = other.getData();
        uint16_t * r      = values;

        for (uint64_t i = 0; i < length_pad; i += 1) {
            r[i] = _mm_cvtss_sh(u[i]);
        }
    }

    inline void restore_scalar(const CloverVector32 &other)
    {
        const uint16_t * u = values;
        float * r          = other.getData();

        for (uint64_t i = 0; i < length_pad; i += 1) {
            r[i] = _mm_cvtsh_ss(u[i]);
        }
    }

    inline void scaleAndAdd_scalar(CloverVector16 &other, float s)
    {

        uint16_t * u = values;
        const uint16_t * v = other.values;

        for (uint64_t i = 0; i < length_pad; i += 1) {
            const float f_u = _mm_cvtsh_ss(u[i]);
            const float f_v = _mm_cvtsh_ss(v[i]);
            const float f_r = _mm_fmadd_ss(f_v, s, f_u);
            u[i] = _mm_cvtss_sh(f_r);
        }
    }

    inline void scaleAndAdd_scalar(const CloverVector16 &other, float s, CloverVector16 &result)
    {

        const uint16_t * u = values;
        const uint16_t * v = other.values;
        uint16_t * r = result.values;

        for (uint64_t i = 0; i < length_pad; i += 1) {
            const float f_u = _mm_cvtsh_ss(u[i]);
            const float f_v = _mm_cvtsh_ss(v[i]);
            const float f_r = _mm_fmadd_ss(f_v, s, f_u);
            r[i] = _mm_cvtss_sh(f_r);
        }
    }


    inline float dot_scalar(const CloverVector16 &other) const
    {
        float dot_product  = 0;
        const uint16_t * u = values;
        const uint16_t * v = other.values;

        for (uint64_t i = 0; i < length_pad; i += 1) {
            const float f_u = _mm_cvtsh_ss(u[i]);
            const float f_v = _mm_cvtsh_ss(v[i]);
            dot_product += f_u * f_v;
        }

        return dot_product;
    }

    /* ============================================================================================================== */
    /* = AVX Operations                                                                                               */
    /* ============================================================================================================== */

    inline void quantize(const CloverVector32 &other)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const float * u   = other.getData();
        uint16_t * r      = values;
        const uint64_t n0 = length_pad;

        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m256 f0 = _mm256_loadu_ps(u + i + 0);
            const __m256 f1 = _mm256_loadu_ps(u + i + 8);
            const __m256 f2 = _mm256_loadu_ps(u + i + 16);
            const __m256 f3 = _mm256_loadu_ps(u + i + 24);

            _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(u + i + 64), _MM_HINT_T0);

            const __m128i q0 = _mm256_cvtps_ph(f0, 0);
            const __m128i q1 = _mm256_cvtps_ph(f1, 0);
            const __m128i q2 = _mm256_cvtps_ph(f2, 0);
            const __m128i q3 = _mm256_cvtps_ph(f3, 0);

            _mm_storeu_si128 ((__m128i *) (r + i +  0), q0);
            _mm_storeu_si128 ((__m128i *) (r + i +  8), q1);
            _mm_storeu_si128 ((__m128i *) (r + i + 16), q2);
            _mm_storeu_si128 ((__m128i *) (r + i + 24), q3);
        }
    }

    inline void quantize_parallel(const CloverVector32 &other)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const float * u   = other.getData();
        uint16_t * r      = values;
        const uint64_t n0 = length_pad;

        _Pragma("omp parallel for schedule(static)")
        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m256 f0 = _mm256_loadu_ps(u + i + 0);
            const __m256 f1 = _mm256_loadu_ps(u + i + 8);
            const __m256 f2 = _mm256_loadu_ps(u + i + 16);
            const __m256 f3 = _mm256_loadu_ps(u + i + 24);

            _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(u + i + 64), _MM_HINT_T0);

            const __m128i q0 = _mm256_cvtps_ph(f0, 0);
            const __m128i q1 = _mm256_cvtps_ph(f1, 0);
            const __m128i q2 = _mm256_cvtps_ph(f2, 0);
            const __m128i q3 = _mm256_cvtps_ph(f3, 0);

            _mm_storeu_si128 ((__m128i *) (r + i +  0), q0);
            _mm_storeu_si128 ((__m128i *) (r + i +  8), q1);
            _mm_storeu_si128 ((__m128i *) (r + i + 16), q2);
            _mm_storeu_si128 ((__m128i *) (r + i + 24), q3);
        }
    }

    inline void restore(CloverVector32 &other) const
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const uint16_t * u = values;
        float * r          = other.getData();
        const uint64_t n0  = length_pad;

        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m128i q0 = _mm_loadu_si128 ((__m128i *) (u + i +  0));
            const __m128i q1 = _mm_loadu_si128 ((__m128i *) (u + i +  8));
            const __m128i q2 = _mm_loadu_si128 ((__m128i *) (u + i + 16));
            const __m128i q3 = _mm_loadu_si128 ((__m128i *) (u + i + 24));

            const __m256 f0 = _mm256_cvtph_ps(q0);
            const __m256 f1 = _mm256_cvtph_ps(q1);
            const __m256 f2 = _mm256_cvtph_ps(q2);
            const __m256 f3 = _mm256_cvtph_ps(q3);

            _mm256_storeu_ps (r + i +  0, f0);
            _mm256_storeu_ps (r + i +  8, f1);
            _mm256_storeu_ps (r + i + 16, f2);
            _mm256_storeu_ps (r + i + 24, f3);
        }
    }

    inline void scaleAndAdd(const CloverVector16 &other, float s)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        uint16_t * u       = values;
        const uint16_t * v = other.values;

        scaleAndAdd(u, v, u, s);
    }

    inline void scaleAndAdd(const CloverVector16 &other, float s, CloverVector16 &result)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const uint16_t * u = values;
        const uint16_t * v = other.values;
        uint16_t * r       = result.values;
        scaleAndAdd(u, v, r, s);
    }

    inline void scaleAndAdd(const uint16_t * u, const uint16_t * v,  uint16_t * r, float s)
    {
        const uint64_t n0  = length_pad;

        const __m256 scale = _mm256_set1_ps(s);

        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + i +  0));
            const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + i + 16));
            const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(v + i +  0));
            const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(v + i + 16));

            _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(v + i + 64), _MM_HINT_T0);

            const __m128i qu_0_lo = _mm256_castsi256_si128(qu_0);
            const __m128i qu_0_hi = _mm256_extractf128_si256(qu_0, 1);
            const __m128i qu_1_lo = _mm256_castsi256_si128(qu_1);
            const __m128i qu_1_hi = _mm256_extractf128_si256(qu_1, 1);

            const __m128i qv_0_lo = _mm256_castsi256_si128(qv_0);
            const __m128i qv_0_hi = _mm256_extractf128_si256(qv_0, 1);
            const __m128i qv_1_lo = _mm256_castsi256_si128(qv_1);
            const __m128i qv_1_hi = _mm256_extractf128_si256(qv_1, 1);

            const __m256 u_0_lo = _mm256_cvtph_ps(qu_0_lo);
            const __m256 u_0_hi = _mm256_cvtph_ps(qu_0_hi);
            const __m256 u_1_lo = _mm256_cvtph_ps(qu_1_lo);
            const __m256 u_1_hi = _mm256_cvtph_ps(qu_1_hi);

            const __m256 v_0_lo = _mm256_cvtph_ps(qv_0_lo);
            const __m256 v_0_hi = _mm256_cvtph_ps(qv_0_hi);
            const __m256 v_1_lo = _mm256_cvtph_ps(qv_1_lo);
            const __m256 v_1_hi = _mm256_cvtph_ps(qv_1_hi);

            const __m256 r_0_lo = _mm256_fmadd_ps(v_0_lo, scale, u_0_lo);
            const __m256 r_0_hi = _mm256_fmadd_ps(v_0_hi, scale, u_0_hi);
            const __m256 r_1_lo = _mm256_fmadd_ps(v_1_lo, scale, u_1_lo);
            const __m256 r_1_hi = _mm256_fmadd_ps(v_1_hi, scale, u_1_hi);

            const __m128i q0 = _mm256_cvtps_ph(r_0_lo, 0);
            const __m128i q1 = _mm256_cvtps_ph(r_0_hi, 0);
            const __m128i q2 = _mm256_cvtps_ph(r_1_lo, 0);
            const __m128i q3 = _mm256_cvtps_ph(r_1_hi, 0);

            _mm_storeu_si128 ((__m128i *) (r + i +  0), q0);
            _mm_storeu_si128 ((__m128i *) (r + i +  8), q1);
            _mm_storeu_si128 ((__m128i *) (r + i + 16), q2);
            _mm_storeu_si128 ((__m128i *) (r + i + 24), q3);
        }
    }

    inline void scaleAndAdd_parallel(const CloverVector16 &other, float s)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        uint16_t * u       = values;
        const uint16_t * v = other.values;

        scaleAndAdd_parallel(u, v, u, s);
    }

    inline void scaleAndAdd_parallel(const CloverVector16 &other, float s, CloverVector16 &result)
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const uint16_t * u = values;
        const uint16_t * v = other.values;
        uint16_t * r       = result.values;
        scaleAndAdd_parallel(u, v, r, s);
    }

    inline void scaleAndAdd_parallel(const uint16_t * u, const uint16_t * v,  uint16_t * r, float s)
    {
#if defined(_OPENMP)
        const uint64_t n0  = length_pad;

        const __m256 scale = _mm256_set1_ps(s);

        _Pragma("omp parallel for schedule(static)")
        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + i +  0));
            const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + i + 16));
            const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(v + i +  0));
            const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(v + i + 16));

            _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(v + i + 64), _MM_HINT_T0);

            const __m128i qu_0_lo = _mm256_castsi256_si128(qu_0);
            const __m128i qu_0_hi = _mm256_extractf128_si256(qu_0, 1);
            const __m128i qu_1_lo = _mm256_castsi256_si128(qu_1);
            const __m128i qu_1_hi = _mm256_extractf128_si256(qu_1, 1);

            const __m128i qv_0_lo = _mm256_castsi256_si128(qv_0);
            const __m128i qv_0_hi = _mm256_extractf128_si256(qv_0, 1);
            const __m128i qv_1_lo = _mm256_castsi256_si128(qv_1);
            const __m128i qv_1_hi = _mm256_extractf128_si256(qv_1, 1);

            const __m256 u_0_lo = _mm256_cvtph_ps(qu_0_lo);
            const __m256 u_0_hi = _mm256_cvtph_ps(qu_0_hi);
            const __m256 u_1_lo = _mm256_cvtph_ps(qu_1_lo);
            const __m256 u_1_hi = _mm256_cvtph_ps(qu_1_hi);

            const __m256 v_0_lo = _mm256_cvtph_ps(qv_0_lo);
            const __m256 v_0_hi = _mm256_cvtph_ps(qv_0_hi);
            const __m256 v_1_lo = _mm256_cvtph_ps(qv_1_lo);
            const __m256 v_1_hi = _mm256_cvtph_ps(qv_1_hi);

            const __m256 r_0_lo = _mm256_fmadd_ps(v_0_lo, scale, u_0_lo);
            const __m256 r_0_hi = _mm256_fmadd_ps(v_0_hi, scale, u_0_hi);
            const __m256 r_1_lo = _mm256_fmadd_ps(v_1_lo, scale, u_1_lo);
            const __m256 r_1_hi = _mm256_fmadd_ps(v_1_hi, scale, u_1_hi);

            const __m128i q0 = _mm256_cvtps_ph(r_0_lo, 0);
            const __m128i q1 = _mm256_cvtps_ph(r_0_hi, 0);
            const __m128i q2 = _mm256_cvtps_ph(r_1_lo, 0);
            const __m128i q3 = _mm256_cvtps_ph(r_1_hi, 0);

            _mm_storeu_si128 ((__m128i *) (r + i +  0), q0);
            _mm_storeu_si128 ((__m128i *) (r + i +  8), q1);
            _mm_storeu_si128 ((__m128i *) (r + i + 16), q2);
            _mm_storeu_si128 ((__m128i *) (r + i + 24), q3);
        }
#else
        scaleAndAdd(u, v, r, s);
#endif
    }


    inline float dot(const CloverVector16 &other) const
    {
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const uint16_t * u = values;
        const uint16_t * v = other.values;
        const uint64_t n0  = length_pad;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (uint64_t i = 0; i < n0; i += 32)
        {
            const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + i +  0));
            const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + i + 16));
            const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(v + i +  0));
            const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(v + i + 16));

            _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
            _mm_prefetch((char *)(v + i + 64), _MM_HINT_T0);

            const __m128i qu_0_lo = _mm256_castsi256_si128(qu_0);
            const __m128i qu_0_hi = _mm256_extractf128_si256(qu_0, 1);
            const __m128i qu_1_lo = _mm256_castsi256_si128(qu_1);
            const __m128i qu_1_hi = _mm256_extractf128_si256(qu_1, 1);

            const __m128i qv_0_lo = _mm256_castsi256_si128(qv_0);
            const __m128i qv_0_hi = _mm256_extractf128_si256(qv_0, 1);
            const __m128i qv_1_lo = _mm256_castsi256_si128(qv_1);
            const __m128i qv_1_hi = _mm256_extractf128_si256(qv_1, 1);

            const __m256 u_0_lo = _mm256_cvtph_ps(qu_0_lo);
            const __m256 u_0_hi = _mm256_cvtph_ps(qu_0_hi);
            const __m256 u_1_lo = _mm256_cvtph_ps(qu_1_lo);
            const __m256 u_1_hi = _mm256_cvtph_ps(qu_1_hi);

            const __m256 v_0_lo = _mm256_cvtph_ps(qv_0_lo);
            const __m256 v_0_hi = _mm256_cvtph_ps(qv_0_hi);
            const __m256 v_1_lo = _mm256_cvtph_ps(qv_1_lo);
            const __m256 v_1_hi = _mm256_cvtph_ps(qv_1_hi);

            acc0 = _mm256_fmadd_ps(v_0_lo, u_0_lo, acc0);
            acc1 = _mm256_fmadd_ps(v_0_hi, u_0_hi, acc1);
            acc2 = _mm256_fmadd_ps(v_1_lo, u_1_lo, acc2);
            acc3 = _mm256_fmadd_ps(v_1_hi, u_1_hi, acc3);
        }

        const __m256 sum0 = _mm256_add_ps(acc0, acc1);
        const __m256 sum1 = _mm256_add_ps(acc2, acc3);
        const __m256 sum2 = _mm256_add_ps(sum0, sum1);

        return _mm256_haddf32_ps(sum2);
    }

    inline float dot_parallel(const CloverVector16 &other) const
    {
#if defined(_OPENMP)
        //
        // Verify matching sizes
        //
        assert(size() == other.size());

        const uint16_t * u = values;
        const uint16_t * v = other.values;
        const uint64_t n0  = length_pad;
        
        float sum = 0.0;
        _Pragma("omp parallel reduction(+:sum)") {
            const uint64_t nt = omp_get_num_threads();
            const uint64_t tid = omp_get_thread_num();

            const uint64_t n_blocks = (n0 - 1) / 32 + 1;
            const uint64_t blocks_per_thread = (n_blocks - 1) / nt + 1;
            const uint64_t start = 32 * blocks_per_thread * tid;
            const uint64_t end = std::min(start + 32*blocks_per_thread, n0);

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (uint64_t i = start; i < end; i += 32)
            {
                const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + i +  0));
                const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + i + 16));
                const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(v + i +  0));
                const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(v + i + 16));

                _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
                _mm_prefetch((char *)(v + i + 64), _MM_HINT_T0);

                const __m128i qu_0_lo = _mm256_castsi256_si128(qu_0);
                const __m128i qu_0_hi = _mm256_extractf128_si256(qu_0, 1);
                const __m128i qu_1_lo = _mm256_castsi256_si128(qu_1);
                const __m128i qu_1_hi = _mm256_extractf128_si256(qu_1, 1);

                const __m128i qv_0_lo = _mm256_castsi256_si128(qv_0);
                const __m128i qv_0_hi = _mm256_extractf128_si256(qv_0, 1);
                const __m128i qv_1_lo = _mm256_castsi256_si128(qv_1);
                const __m128i qv_1_hi = _mm256_extractf128_si256(qv_1, 1);

                const __m256 u_0_lo = _mm256_cvtph_ps(qu_0_lo);
                const __m256 u_0_hi = _mm256_cvtph_ps(qu_0_hi);
                const __m256 u_1_lo = _mm256_cvtph_ps(qu_1_lo);
                const __m256 u_1_hi = _mm256_cvtph_ps(qu_1_hi);

                const __m256 v_0_lo = _mm256_cvtph_ps(qv_0_lo);
                const __m256 v_0_hi = _mm256_cvtph_ps(qv_0_hi);
                const __m256 v_1_lo = _mm256_cvtph_ps(qv_1_lo);
                const __m256 v_1_hi = _mm256_cvtph_ps(qv_1_hi);

                acc0 = _mm256_fmadd_ps(v_0_lo, u_0_lo, acc0);
                acc1 = _mm256_fmadd_ps(v_0_hi, u_0_hi, acc1);
                acc2 = _mm256_fmadd_ps(v_1_lo, u_1_lo, acc2);
                acc3 = _mm256_fmadd_ps(v_1_hi, u_1_hi, acc3);
            }

            const __m256 sum0 = _mm256_add_ps(acc0, acc1);
            const __m256 sum1 = _mm256_add_ps(acc2, acc3);
            const __m256 sum2 = _mm256_add_ps(sum0, sum1);
            
            sum = _mm256_haddf32_ps(sum2);
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
            const uint16_t bits = getBits(i);
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
                const uint16_t bits = getBits(i);
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
            setBits(idx, (uint16_t) min_heap[i].bits.i);
        }
    }


    inline void threshold_min_heap_parallel (idx_t * min_heaps, uint64_t k)
    {
#if defined(_OPENMP)
        const uint64_t n0 = length;
        uint64_t nt = (uint64_t) get_OpenMP_threads();;
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
                const int16_t bits = getBits(i);
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
                    const int16_t bits = getBits(i);
                    min_heap[0].value = value;
                    min_heap[0].idx = i;
                    min_heap[0].bits.i = bits;
                    min_heapify(min_heap, 0, k);
                }
                setBits(i, 0);
            }
            std::sort_heap(min_heap, min_heap + k, gt_idx_t);
        }

//        std::sort(min_heaps, min_heaps + k*nt, gt_idx_t);

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
            setBits(idx, (int16_t) min_heaps[best_j*k + indices[best_j]].bits.i);

            indices[best_j]++;
        }
#else
        threshold(k);
#endif
    }


    uint64_t getBitsLength () const {
        return 16;
    }

    inline void clear ()
    {
        uint16_t * u = values;
        const uint64_t blocks = length_pad / 16;
        //
        // Clear the values
        //
        for (uint64_t b = 0; b < blocks; b += 1) {
            const uint64_t offset = b * 16;
            _mm256_storeu_si256((__m256i *)(u + offset), _mm256_setzero_si256());
        }
    }
};


#endif /* CLOVER_VECTOR16_H */
