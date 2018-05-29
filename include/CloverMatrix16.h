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

#ifndef CLOVER_MATRIX16_H
#define CLOVER_MATRIX16_H

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "CloverMatrix.h"
#include "CloverVector32.h"
#include "CloverMatrix32.h"
#include "CloverVector16.h"
#include "mkl.h"

class CloverMatrix16 : public CloverMatrix {

protected:
    uint16_t * values;

    void allocate()
    {
        uint64_t value_bytes = cols * rows * sizeof(uint16_t);
        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes);
        if (ret != 0) {
            std::cout << "Could not allocate memory for CloverMatrix16. Exiting ..." << std::endl;
            exit(1);
        }
    }

public:
    CloverMatrix16 (uint64_t h, uint64_t w) : CloverMatrix(h, w)
    {
        allocate();
    }

    uint16_t * getData () const
    {
        return values;
    }

    uint64_t getBitsLength () const {
        return 16;
    }

    inline uint64_t getBytes () const
    {
        const uint64_t s = cols * rows;
        return s * sizeof(uint16_t);
    }

    inline std::string toString () const
    {
        std::stringstream sout;

        for (int i = 0; i < rows; i += 1) {
            for (int j = 0; j < cols; j += 1) {
                sout << values[i * cols + j] << " ";
            }
            sout << ";" << std::endl;
        }

        return sout.str();
    }

    inline float get(uint64_t i, uint64_t j)
    {
        return _mm_cvtsh_ss(values[i * cols + j]);
    }

    inline void set(uint64_t i, uint64_t j, float value)
    {
        values[i * cols + j] = _mm_cvtss_sh(value);
    }

    inline void mvm_scalar(const CloverVector16 &productVector, CloverVector16 &result) const
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }
        //
        // Setup the constants
        //
        const uint16_t * A = values;
        const uint16_t * x = productVector.getData();
        uint16_t * y       = result.getData();

        const uint64_t m = rows;
        const uint64_t n = cols;

        for (uint64_t i = 0; i < m; i += 1) {
            float y_i = 0;
            for (uint64_t j = 0; j < n; j += 1) {
                //
                // Get the index
                //
                const uint64_t idx = i * n + j;

                const float A_i = _mm_cvtsh_ss(values[idx]);
                const float x_j = _mm_cvtsh_ss(x[j]);

                y_i += A_i * x_j;
            }
            y[i] = _mm_cvtss_sh(y_i);
        }
    }



    inline void mvm_parallel(const CloverVector16 &productVector, CloverVector16 &result) const
    {
#if defined(_OPENMP)
        _Pragma("omp parallel") {
            if (productVector.size() != getCols()) {
                std::cout << "MVM can not be performed. Exiting ..." << std::endl;
                exit(1);
            }
            //
            // Setup the constants
            //
            const uint16_t * A = values;
            const uint16_t * x = productVector.getData();
            uint16_t * y = result.getData();
            const uint64_t m = getRows();
            const uint64_t n = getCols();

            //
            // Stuff for parallelization
            //
            uint64_t nt = omp_get_num_threads();
            uint64_t tid = omp_get_thread_num();

            uint64_t n_rowblocks = (m - 1) / 8 + 1;
            uint64_t rowblocks_per_thread = (n_rowblocks - 1) / nt + 1;
            uint64_t start = 8 * rowblocks_per_thread * tid;
            uint64_t end = std::min(m, start + 8 * rowblocks_per_thread);

            for (uint64_t b = start; b < end; b += 8)
            {
                float block[8];

                for (uint64_t i = 0; i < 8; i += 1) {

                    const uint64_t offset = b + i;
                    const uint16_t * u = A + (offset * cols);

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (uint64_t j = 0; j < n; j += 32) {

                        const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + j +  0));
                        const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + j + 16));
                        const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(x + j +  0));
                        const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(x + j + 16));

                        _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
                        _mm_prefetch((char *)(x + i + 64), _MM_HINT_T0);

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

                    block[i] = _mm256_haddf32_ps(sum2);
                }

                const __m256 f0 = _mm256_loadu_ps(block);
                const __m128i q0 = _mm256_cvtps_ph(f0, 0);

                _mm_storeu_si128 ((__m128i *) (y + b), q0);
            }
        }
#else
        mvm(productVector, result);
#endif
    }

    inline void mvm(const CloverVector16 &productVector, CloverVector16 &result) const
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }
        //
        // Setup the constants
        //
        const uint16_t * A = values;
        const uint16_t * x = productVector.getData();
        uint16_t * y = result.getData();
        const uint64_t m = getRows();
        const uint64_t n = getCols();

        for (uint64_t b = 0; b < m; b += 8)
        {
            float block[8];

            for (uint64_t i = 0; i < 8; i += 1) {

                const uint64_t offset = b + i;
                const uint16_t * u = A + (offset * cols);

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (uint64_t j = 0; j < n; j += 32) {

                    const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + j +  0));
                    const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + j + 16));
                    const __m256i qv_0 = _mm256_loadu_si256 ((__m256i *)(x + j +  0));
                    const __m256i qv_1 = _mm256_loadu_si256 ((__m256i *)(x + j + 16));

                    _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
                    _mm_prefetch((char *)(x + i + 64), _MM_HINT_T0);

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

                block[i] = _mm256_haddf32_ps(sum2);
            }

            const __m256 f0 = _mm256_loadu_ps(block);
            const __m128i q0 = _mm256_cvtps_ph(f0, 0);

            _mm_storeu_si128 ((__m128i *) (y + b), q0);
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
        //
        // Setup the constants
        //
        const uint16_t * A = values;
        const float * x = productVector.getData();
        float * y = resultVector.getData();
        const uint64_t m = getRows();
        const uint64_t n = getCols();

        for (uint64_t i = 0; i < m; i += 1)
        {
            const uint16_t * u = A + (i * cols);

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (uint64_t j = 0; j < n; j += 32) {

                const __m256i qu_0 = _mm256_loadu_si256 ((__m256i *)(u + j +  0));
                const __m256i qu_1 = _mm256_loadu_si256 ((__m256i *)(u + j + 16));

                _mm_prefetch((char *)(u + i + 32), _MM_HINT_T0);
                _mm_prefetch((char *)(x + i + 64), _MM_HINT_T0);

                const __m128i qu_0_lo = _mm256_castsi256_si128(qu_0);
                const __m128i qu_0_hi = _mm256_extractf128_si256(qu_0, 1);
                const __m128i qu_1_lo = _mm256_castsi256_si128(qu_1);
                const __m128i qu_1_hi = _mm256_extractf128_si256(qu_1, 1);

                const __m256 v_0_lo = _mm256_loadu_ps(x + j +  0);
                const __m256 v_0_hi = _mm256_loadu_ps(x + j +  8);
                const __m256 v_1_lo = _mm256_loadu_ps(x + j + 16);
                const __m256 v_1_hi = _mm256_loadu_ps(x + j + 24);

                const __m256 u_0_lo = _mm256_cvtph_ps(qu_0_lo);
                const __m256 u_0_hi = _mm256_cvtph_ps(qu_0_hi);
                const __m256 u_1_lo = _mm256_cvtph_ps(qu_1_lo);
                const __m256 u_1_hi = _mm256_cvtph_ps(qu_1_hi);

                acc0 = _mm256_fmadd_ps(v_0_lo, u_0_lo, acc0);
                acc1 = _mm256_fmadd_ps(v_0_hi, u_0_hi, acc1);
                acc2 = _mm256_fmadd_ps(v_1_lo, u_1_lo, acc2);
                acc3 = _mm256_fmadd_ps(v_1_hi, u_1_hi, acc3);
            }

            const __m256 sum0 = _mm256_add_ps(acc0, acc1);
            const __m256 sum1 = _mm256_add_ps(acc2, acc3);
            const __m256 sum2 = _mm256_add_ps(sum0, sum1);

            y[i] = _mm256_haddf32_ps(sum2);
        }
    }


    void inline quantize(const CloverMatrix32 &other)
    {
        const uint64_t n0 = size();
        uint16_t * r      = values;
        const float * u   = other.getData();

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


    inline void quantize_scalar(const CloverMatrix32 &other)
    {
        const uint64_t n0 = size();
        const float * u   = other.getData();
        uint16_t * r      = values;

        for (uint64_t i = 0; i < n0; i += 1) {
            r[i] = _mm_cvtss_sh(u[i]);
        }
    }


    inline void transpose_scalar (CloverMatrix16 &other)
    {
        if (other.rows != cols || other.cols != rows) {
            std::cout << "Matrix can not be transposed. Exiting ..." << std::endl;
            exit(1);
        }

        uint16_t * u = values;
        uint16_t * v = other.values;

        for (uint64_t i = 0; i < rows; i += 1) {
            for (uint64_t j = 0; j < cols; j += 1) {
                v[j * rows + i] = u[i * cols + j];
            }
        }
    }

    inline void transpose_parallel (CloverMatrix16 &other)
    {
        //
        // MKL does not offer 16-bit transpose, so let's use IPP
        //
        uint16_t * src = values;
        uint16_t * dst = other.values;
        IppiSize srcRoi = { (int) cols, (int) rows  };

        ippiTranspose_16u_C1R ( src, (int) cols * sizeof(uint16_t), dst, (int) rows * sizeof(uint16_t), srcRoi );
    }

    inline void transpose (CloverMatrix16 &other)
    {
        //
        // Make sure we are running a single thread
        //
        ippSetNumThreads(1);

        //
        // MKL does not offer 16-bit transpose, so let's use IPP
        //
        uint16_t * src = values;
        uint16_t * dst = other.values;
        IppiSize srcRoi = { (int) cols, (int) rows  };

        ippiTranspose_16u_C1R ( src, (int) cols * sizeof(uint16_t), dst, (int) rows * sizeof(uint16_t), srcRoi );

        //
        // Get back to the official number of threads
        //
        ippSetNumThreads(get_OpenMP_threads());
    }


    inline void clear ()
    {
        char * u = (char *) values;
        const uint64_t n0 = getBytes();
        const uint64_t n1 = n0 >> 6;
        for (uint64_t i = 0; i < n1; i += 32) {
            _mm256_storeu_si256((__m256i *)(u + i), _mm256_setzero_si256());
        }
        for (uint64_t i = n1; i < n0; i += 1) {
            u[i] = 0;
        }
    }

    ~CloverMatrix16()
    {
        free(values);
    }

};

#endif /* CLOVER_MATRIX16_H */
