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

#ifndef CLOVER_VECTOR32_H
#define CLOVER_VECTOR32_H

#include <limits>
#include <tuple>
#include <algorithm>
#include <string.h>
#include <vector>

#include "CloverVector.h"
#include "simdxorshift128plus.h"

class CloverVector32 : public CloverVector {

private:
    bool memoryManagement;

protected:
    float * values;

public:

    CloverVector32(uint64_t s, float * data): CloverVector(s)
    {
        values = data;
        memoryManagement = false;
    }

    CloverVector32(uint64_t s): CloverVector(s)
    {
        uint64_t values_bytes = length_pad * sizeof(float);
        const int ret = posix_memalign((void **) &values, get_system_pagesize(), values_bytes);

        if (ret == 0) {
            //
            // Make sure that the value padding is zeroed-out
            //
            for (uint64_t i = length; i < length_pad; i += 1) {
                values[i] = 0;
            }
            memoryManagement = true;
        } else {
            std::cout << "Could not allocate memory for CloverVector32. Exiting ..." << std::endl;
            exit(1);
        }
    }

    CloverVector32 (const CloverVector32& other): CloverVector(other.length)
    {
        const uint64_t value_bytes = length_pad * sizeof(float);
        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes);

        if (ret == 0) {
            //
            // Make sure that the value padding is zeroed-out
            //
            for (uint64_t i = length; i < length_pad; i += 1) {
                values[i] = 0;
            }
            memoryManagement = true;
        } else {
            std::cout << "Could not allocate memory for CloverVector32. Exiting ..." << std::endl;
            exit(1);
        }
        memcpy(values, other.values, value_bytes);
    }


    inline float get(uint64_t i) const
    {
        return values[i];
    }

    inline float getAbs(uint64_t i) const
    {
        Restorator result;
        result.f = values[i];
        result.i = result.i & 0x7FFFFFFF;
        return result.f;
    }

    inline void set(uint64_t i, float v)
    {
        values[i] = v;
    }

    inline void setData(float * data)
    {
        values = data;
    }


    uint64_t getBitsLength () const {
        return 32;
    }

    std::string toString () const
    {
        std::stringstream sout;
        const uint64_t n0 = length;
        for (uint64_t i = 0; i < n0; i += 1)
        {
            const float val = values[i];
            sout << std::setw(10) << i;
            sout << " | ";;
            sout << std::setw(20) << std::fixed << std::setprecision(7) << val;
            sout << " | ";
            sout << float2hex(val);
            sout << " | ";
            sout << std::endl;
        }
        return sout.str();
    }

    inline float * getData () const
    {
        return values;
    }

    inline uint64_t getBytes () const
    {
        return length_pad * sizeof(float);
    }

    ~CloverVector32()
    {
        if (memoryManagement) {
            free(values);
        }
    }

    /* ============================================================================================================== */
    /* = Scalar Operations                                                                                            */
    /* ============================================================================================================== */

    inline void scaleAndAdd_scalar(const CloverVector32 &other, const float s)
    {
        //
        // ACCUMULATION, ACCUMULATION, ACCUMULATION, ACCUMULATION, ACCUMULATION, ACCUMULATION, ACCUMULATION,
        //
        float * u          = values;
        const float * v    = other.values;
        const uint64_t n0  = length_pad;
        scaleAndAdd_scalar(u, v, s, u, n0);
    }


    inline void scaleAndAdd_scalar(const CloverVector32 &other, const float s, const CloverVector32 &result)
    {
        const float * u    = values;
        const float * v    = other.values;
        float * r          = result.values;
        const uint64_t n0  = length_pad;
        scaleAndAdd_scalar(u, v, s, r, n0);
    }

    inline void scaleAndAdd_scalar (const float * u, const float * v, const float s, float * r, const uint64_t n0)
    {
        //
        // Actual implementation
        //
        for (uint64_t i = 0; i < n0; i += 1) {
            r[i] = u[i] + v[i] * s;
        }
    }

    inline float dot_scalar(const CloverVector32 &other) const
    {
        assert(size() == other.size());

        const float * u    = values;
        const float * v    = other.values;
        const uint64_t n0  = length;
        float result       = 0;

        for (uint64_t i = 0; i < n0; i += 1) {
            result += u[i] * v[i];
        }

        return result;
    }

    inline void quantize_scalar(const CloverVector32 &other)
    {
        quantize(other);
    }

    /* ============================================================================================================== */
    /* = Vector Operations                                                                                            */
    /* ============================================================================================================== */


    void inline quantize(const CloverVector32 &other)
    {
        const uint64_t n0 = length_pad;
        float * u         = values;
        const float * v   = other.values;

        #if defined(__AVX__)
            for (uint64_t i = 0; i < n0; i += 32)
            {
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                const __m256 v4 = _mm256_loadu_ps(v + i + 24);

                _mm256_storeu_ps(u + i +  0, v1);
                _mm256_storeu_ps(u + i +  8, v2);
                _mm256_storeu_ps(u + i + 16, v3);
                _mm256_storeu_ps(u + i + 24, v4);
            }
        #else
            memcpy(u, v, sizof(float) * n0);
        #endif
    }

    void inline quantize_parallel(const CloverVector32 &other)
    {
        const uint64_t n0 = length_pad;
        float * u         = values;
        const float * v   = other.values;

        #if defined(__AVX__)
            _Pragma("omp parallel for schedule(static)")
            for (uint64_t i = 0; i < n0; i += 32)
            {
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                const __m256 v4 = _mm256_loadu_ps(v + i + 24);

                _mm256_storeu_ps(u + i +  0, v1);
                _mm256_storeu_ps(u + i +  8, v2);
                _mm256_storeu_ps(u + i + 16, v3);
                _mm256_storeu_ps(u + i + 24, v4);
            }
        #else
            memcpy(u, v, sizof(float) * n0);
        #endif
    }

    void inline restore(CloverVector32 &other)
    {
        const uint64_t n0 = length_pad;
        float * u         = other.values;
        const float * v   = values;

        #if defined(__AVX__)
            for (uint64_t i = 0; i < n0; i += 32)
            {
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                const __m256 v4 = _mm256_loadu_ps(v + i + 24);

                _mm256_storeu_ps(u + i +  0, v1);
                _mm256_storeu_ps(u + i +  8, v2);
                _mm256_storeu_ps(u + i + 16, v3);
                _mm256_storeu_ps(u + i + 24, v4);
            }
        #else
            memcpy(u, v, sizof(float) * n0);
        #endif
    }


    inline void scaleAndAdd(const float * u, const float * v, const float s, float * r, const uint64_t n0)
    {
        #if defined(__AVX__)
            //
            // Define the scale
            //
            const __m256 scale = _mm256_set1_ps(s);

            for (size_t i = 0; i < n0; i += 16)
            {
                const __m256 u1 = _mm256_loadu_ps(u + i + 0);
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 u2 = _mm256_loadu_ps(u + i + 8);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);

                #if defined(__FMA__)
                    const __m256 sa_u1 = _mm256_fmadd_ps(v1, scale, u1);
                    const __m256 sa_u2 = _mm256_fmadd_ps(v2, scale, u2);
                #else
                    const __m256 mul_1 = _mm256_mul_ps(v1, scale);
                    const __m256 mul_2 = _mm256_mul_ps(v2, scale);
                    const __m256 sa_u1 = _mm256_add_ps(mul_1, u1);
                    const __m256 sa_u2 = _mm256_add_ps(mul_2, u2);
                #endif

                _mm256_storeu_ps(r + (i + 0), sa_u1);
                _mm256_storeu_ps(r + (i + 8), sa_u2);
            }
        #else
            std::cout << "Currently, a non-AVX version of getAbsMax is not defined. Exiting ..." << std::endl;
            exit(1);
        #endif
    }

    inline void scaleAndAdd(const CloverVector32 &other, const float s)
    {
        assert(size() == other.size());
        float * u          = values;
        const float * v    = other.values;
        const uint64_t n0  = length_pad;
        scaleAndAdd(u, v, s, u, n0);
    }

    inline void scaleAndAdd(const CloverVector32 &other, const float s, const CloverVector32 &result)
    {
        assert(size() == other.size());
        assert(size() == result.size());

        const float * u    = values;
        const float * v    = other.values;
        float * r          = result.values;
        const uint64_t n0  = length_pad;

        scaleAndAdd(u, v, s, result.values, n0);
    }

    inline void scaleAndAdd_parallel(const float * u, const float * v, const float s, float * r, const uint64_t n0)
    {
        #if defined(__AVX__)
            //
            // Define the scale
            //
            const __m256 scale = _mm256_set1_ps(s);
            
            _Pragma("omp parallel for schedule(static)")
            for (size_t i = 0; i < n0; i += 16)
            {
                const __m256 u1 = _mm256_loadu_ps(u + i + 0);
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 u2 = _mm256_loadu_ps(u + i + 8);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);

                #if defined(__FMA__)
                    const __m256 sa_u1 = _mm256_fmadd_ps(v1, scale, u1);
                    const __m256 sa_u2 = _mm256_fmadd_ps(v2, scale, u2);
                #else
                    const __m256 mul_1 = _mm256_mul_ps(v1, scale);
                    const __m256 mul_2 = _mm256_mul_ps(v2, scale);
                    const __m256 sa_u1 = _mm256_add_ps(mul_1, u1);
                    const __m256 sa_u2 = _mm256_add_ps(mul_2, u2);
                #endif

                _mm256_storeu_ps(r + (i + 0), sa_u1);
                _mm256_storeu_ps(r + (i + 8), sa_u2);
            }
        #else
            std::cout << "Currently, a non-AVX version of getAbsMax is not defined. Exiting ..." << std::endl;
            exit(1);
        #endif
    }

    inline void scaleAndAdd_parallel(const CloverVector32 &other, const float s)
    {
        assert(size() == other.size());
        float * u          = values;
        const float * v    = other.values;
        const uint64_t n0  = length_pad;
        scaleAndAdd_parallel(u, v, s, u, n0);
    }

    inline void scaleAndAdd_parallel(const CloverVector32 &other, const float s, const CloverVector32 &result)
    {
        assert(size() == other.size());
        assert(size() == result.size());

        const float * u    = values;
        const float * v    = other.values;
        float * r          = result.values;
        const uint64_t n0  = length_pad;

        scaleAndAdd_parallel(u, v, s, result.values, n0);
    }



    float inline dot(const CloverVector32 &other) const
    {
        const uint64_t n0 = length_pad;
        const float * u   = values;
        const float * v   = other.values;

        #if defined(__AVX__)

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (uint64_t i = 0; i < n0; i += 32)
            {
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                const __m256 v4 = _mm256_loadu_ps(v + i + 24);

                const __m256 u1 = _mm256_loadu_ps(u + i + 0);
                const __m256 u2 = _mm256_loadu_ps(u + i + 8);
                const __m256 u3 = _mm256_loadu_ps(u + i + 16);
                const __m256 u4 = _mm256_loadu_ps(u + i + 24);

                #if defined(__FMA__)
                    acc1 = _mm256_fmadd_ps(v1, u1, acc1);
                    acc2 = _mm256_fmadd_ps(v2, u2, acc2);
                    acc3 = _mm256_fmadd_ps(v3, u3, acc3);
                    acc4 = _mm256_fmadd_ps(v4, u4, acc4);
                #else
                    mul1 = _mm256_mul_ps(v1, u1);
                    mul2 = _mm256_mul_ps(v2, u2);
                    mul3 = _mm256_mul_ps(v3, u3);
                    mul4 = _mm256_mul_ps(v4, u4);
                    acc1 = _mm256_add_ps(mul1, acc1);
                    acc2 = _mm256_add_ps(mul2, acc2);
                    acc3 = _mm256_add_ps(mul3, acc3);
                    acc4 = _mm256_add_ps(mul4, acc4);
                #endif
            }

            // add the accumulators
            const __m256 tmp1 = _mm256_add_ps(acc1, acc2);
            const __m256 tmp2 = _mm256_add_ps(acc3, acc4);
            const __m256 tmp3 = _mm256_add_ps(tmp1, tmp2);

            return _mm256_haddf32_ps(tmp3);

        #endif
    }

    float inline dot_parallel(const CloverVector32 &other) const
    {
#if defined(_OPENMP)
        const uint64_t n0 = length_pad;
        const float * u   = values;
        const float * v   = other.values;

        float sum = 0.0;
        _Pragma("omp parallel reduction(+:sum)") {
            const uint64_t nt = omp_get_num_threads();
            const uint64_t tid = omp_get_thread_num();

            const uint64_t n_blocks = (n0 - 1) / 32 + 1;
            const uint64_t blocks_per_thread = (n_blocks - 1) / nt + 1;
            const uint64_t start = 32 * blocks_per_thread * tid;
            const uint64_t end = std::min(start + 32 * blocks_per_thread, n0);

            #if defined(__AVX__)

                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();

                for (uint64_t i = start; i < end; i += 32)
                {
                    const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                    const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                    const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                    const __m256 v4 = _mm256_loadu_ps(v + i + 24);

                    const __m256 u1 = _mm256_loadu_ps(u + i + 0);
                    const __m256 u2 = _mm256_loadu_ps(u + i + 8);
                    const __m256 u3 = _mm256_loadu_ps(u + i + 16);
                    const __m256 u4 = _mm256_loadu_ps(u + i + 24);

                    #if defined(__FMA__)
                        acc1 = _mm256_fmadd_ps(v1, u1, acc1);
                        acc2 = _mm256_fmadd_ps(v2, u2, acc2);
                        acc3 = _mm256_fmadd_ps(v3, u3, acc3);
                        acc4 = _mm256_fmadd_ps(v4, u4, acc4);
                    #else
                        mul1 = _mm256_mul_ps(v1, u1);
                        mul2 = _mm256_mul_ps(v2, u2);
                        mul3 = _mm256_mul_ps(v3, u3);
                        mul4 = _mm256_mul_ps(v4, u4);
                        acc1 = _mm256_add_ps(mul1, acc1);
                        acc2 = _mm256_add_ps(mul2, acc2);
                        acc3 = _mm256_add_ps(mul3, acc3);
                        acc4 = _mm256_add_ps(mul4, acc4);
                    #endif
                }

                // add the accumulators
                const __m256 tmp1 = _mm256_add_ps(acc1, acc2);
                const __m256 tmp2 = _mm256_add_ps(acc3, acc4);
                const __m256 tmp3 = _mm256_add_ps(tmp1, tmp2);

                sum = _mm256_haddf32_ps(tmp3);

            #endif
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
            min_heap[i].value = value;
            min_heap[i].bits.f = values[i];
            min_heap[i].idx = i;
            values[i] = 0;
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
                min_heap[0].value = value;
                min_heap[0].bits.f = values[i];
                min_heap[0].idx = i;
                min_heapify(min_heap, 0, k);
            }
            values[i] = 0;
        }
        //
        // Only copy the max K elements O(k)
        //
        for (int i = 0; i < k; i += 1) {
            const uint64_t idx = min_heap[i].idx;
            values[idx] = min_heap[i].bits.f;
        }
    }

void threshold_min_heap_parallel (idx_t * min_heaps, uint64_t k)
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
                min_heap[i - start].value = getAbs(i);
                min_heap[i - start].bits.f = values[i];
                min_heap[i - start].idx = i;
                values[i] = 0.0;
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
                    min_heap[0].value = value;
                    min_heap[0].bits.f = values[i];
                    min_heap[0].idx = i;
                    min_heapify(min_heap, 0, k);
                }
                values[i] = 0.0;
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
            values[idx] = min_heaps[best_j*k + indices[best_j]].bits.f;

            indices[best_j]++;
        }
#else
        threshold(k);
#endif
    }



    //
    //  Zero-out memory
    //
    void clear()
    {
        const uint64_t n0  = length_pad;
        const __m256 zero = _mm256_setzero_ps();
        for (uint64_t i = 0; i < n0; i += 8) {
            _mm256_storeu_ps(values + i, zero);
        }
    }

    // ==============================================================================================================
    // = Initialization
    // ==============================================================================================================

    inline void setRandomInteger(float max_value_ss)
    {
        setRandomInteger(-max_value_ss, max_value_ss, random_key1, random_key2);
    }

    inline void setRandomInteger(float max_value_ss, __m256i &key1, __m256i &key2)
    {
        setRandomInteger(-max_value_ss, max_value_ss, key1, key2);
    }

    inline void setRandomInteger(float min_value_ss, float max_value_ss)
    {
        setRandomInteger(min_value_ss, max_value_ss, random_key1, random_key2);
    }

    inline void setRandomInteger(float min_value_ss, float max_value_ss, __m256i &key1, __m256i &key2)
    {
        //
        // Setup the constants
        //
        float * f32_mem = values;
        const uint64_t vsize0 = (length >> 3) << 3;
        const __m256 rcp_2pow31 = _mm256_set1_ps((max_value_ss - min_value_ss) / 2147483648.0f);
        const __m256 min_value  = _mm256_set1_ps(min_value_ss);
        //
        // Populate with random data
        //
        for (uint64_t i = 0; i < vsize0; i += 8)
        {
            const __m256i irandom    = _mm256_abs_epi32(avx_xorshift128plus(key1, key2));
            const __m256  frandom    = _mm256_cvtepi32_ps (irandom);
            const __m256  range_rnd0 = _mm256_fmadd_ps(frandom, rcp_2pow31, min_value);
            const __m256  range_rnd  = _mm256_round_ps (range_rnd0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(f32_mem + i, range_rnd);
        }
        //
        // Handle the left-overs
        //
        const __m256i mask_first_32bits = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFFU);
        for (uint64_t i = vsize0; i < length; i += 1) {

            const __m256i irandom    = _mm256_abs_epi32(avx_xorshift128plus(key1, key2));
            const __m256  frandom    = _mm256_cvtepi32_ps (irandom);
            const __m256  range_rnd0 = _mm256_fmadd_ps(frandom, rcp_2pow31, min_value);
            const __m256  range_rnd  = _mm256_round_ps (range_rnd0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_maskstore_ps(f32_mem + i, mask_first_32bits, range_rnd);
        }
    }

    inline void setRandomFloats(float min_value_ss, float max_value_ss)
    {
        setRandomFloats(min_value_ss, max_value_ss, random_key1, random_key2);
    }

    inline void setRandomFloats(float min_value_ss, float max_value_ss, __m256i &key1, __m256i &key2)
    {
        //
        // Setup the constants
        //
        float * f32_mem = values;
        const uint64_t vsize0 = (length >> 3) << 3;
        const __m256 rcp_2pow31 = _mm256_set1_ps((max_value_ss - min_value_ss) / 2147483648.0f);
        const __m256 min_value  = _mm256_set1_ps(min_value_ss);
        //
        // Populate with random data
        //
        for (uint64_t i = 0; i < vsize0; i += 8)
        {
            const __m256i irandom    = _mm256_abs_epi32(avx_xorshift128plus(key1, key2));
            const __m256  frandom    = _mm256_cvtepi32_ps (irandom);
            const __m256  range_rnd0 = _mm256_fmadd_ps(frandom, rcp_2pow31, min_value);
            _mm256_storeu_ps(f32_mem + i, range_rnd0);
        }
        //
        // Handle the left-overs
        //
        const __m256i mask_first_32bits = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFFU);
        for (uint64_t i = vsize0; i < length; i += 1) {

            const __m256i irandom    = _mm256_abs_epi32(avx_xorshift128plus(key1, key2));
            const __m256  frandom    = _mm256_cvtepi32_ps (irandom);
            const __m256  range_rnd0 = _mm256_fmadd_ps(frandom, rcp_2pow31, min_value);
            _mm256_maskstore_ps(f32_mem + i, mask_first_32bits, range_rnd0);
        }
    }
};


#endif /* CLOVER_VECTOR32_H */
