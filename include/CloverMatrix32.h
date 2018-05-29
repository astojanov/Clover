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


#ifndef CLOVER_MATRIX32_H
#define CLOVER_MATRIX32_H

#include "CloverMatrix.h"
#include "CloverVector32.h"
#include "mkl.h"

#if defined(_OPENMP)
#include <omp.h>
#endif


class CloverMatrix32 : public CloverMatrix {

protected:
    float * values;

    void allocate()
    {
        uint64_t value_bytes = cols * rows * sizeof(float);
        const int ret = posix_memalign((void **) &values, get_system_pagesize(), value_bytes);
        if (ret != 0) {
            std::cout << "Could not allocate memory for CloverMatrix32. Exiting ..." << std::endl;
            exit(1);
        }
    }

public:

    CloverMatrix32 (uint64_t h, uint64_t w) : CloverMatrix(h, w)
    {
        allocate();
    }

    uint64_t getBitsLength () const {
        return 32;
    }

    float * getData () const
    {
        return values;
    }

    inline uint64_t getBytes () const
    {
        const uint64_t s = cols * rows;
        return s * sizeof(float);
    }

    inline std::string toString () const
    {
        std::stringstream sout;

        for (int i = 0; i < rows; i += 1) {
            for (int j = 0; j < cols; j += 1) {
                sout << std::setw(7) << std::setprecision(2) << values[i * cols + j] << " ";
            }
            sout << ";" << std::endl;
        }

        return sout.str();
    }


    inline void mvm_parallel(const CloverVector32 &productVector, CloverVector32 &result) const
    {
        if (productVector.size() != getCols()) {
            std::cout << "MVM can not be performed. Exiting ..." << std::endl;
            exit(1);
        }
        //
        // Setup the constants
        //
        const float * A = productVector.getData();
        float * y = result.getData();
        const int m = (int) getRows();
        const int n = (int) getCols();
        //
        // Just call MKL using BLAS SGEMV routine:
        //
        cblas_sgemv (CblasRowMajor, CblasNoTrans, m, n, 1, values, n, A, 1, 0, y, 1);
    }

    inline void mvm(const CloverVector32 &productVector, CloverVector32 &result) const
    {
        if (productVector.size() != getCols()) {
            std::cout << "Can't perform MVM: " << getRows() << " x " << getCols() << " Matrix times a " << productVector.size() << " vector to update a " << result.size() << " vector. Exiting..." << std::endl;
            exit(1);
        }
        //
        // Setup the constants
        //
        const float * A = productVector.getData();
        float * y = result.getData();
        const int m = (int) getRows();
        const int n = (int) getCols();
        //
        // Just call MKL using BLAS SGEMV routine:
        //
        mkl_set_num_threads(1);
        cblas_sgemv (CblasRowMajor, CblasNoTrans, m, n, 1, values, n, A, 1, 0, y, 1);
        mkl_set_num_threads(get_OpenMP_threads());
    }


    inline void mvm_scalar(const CloverVector32 &productVector, CloverVector32 &result) const
    {
        mvm_parallel(productVector, result);
    }


    /**
     * Quantize 32-bit matrix into 32-bit matrix. In reality this operation is
     * pointless, and we keep it for compatibility with the other class structures.
     * In this sense a quantization will result in a simple copy.
     *
     * @param other - 32-bit matrix that needs to be quantized
     */
    void inline quantize(const CloverMatrix32 &other)
    {
        const uint64_t n0 = size();
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


    inline void transpose_scalar (CloverMatrix32 &other)
    {
        float * u = values;
        float * v = other.values;

        for (uint64_t i = 0; i < rows; i += 1) {
            for (uint64_t j = 0; j < cols; j += 1) {
                v[j * rows + i] = u[i * cols + j];
            }
        }
    }

    inline void transpose_parallel (CloverMatrix32 &other)
    {
//        //
//        // Perform the transposition with Intel IPP
//        //
//        float * src = values;
//        float * dst = other.values;
//        IppiSize srcRoi = { (int) cols, (int) rows  };
//
//        ippiTranspose_32f_C1R ( src, (int) cols * sizeof(float), dst, (int) rows * sizeof(float), srcRoi );

        mkl_somatcopy ('R', 'T', rows, cols, 1, values, cols, other.values, rows);
    }


    inline void transpose (CloverMatrix32 &other)
    {
        //
        // Make sure we are running a single thread
        //
        ippSetNumThreads(1);

        //
        // Perform the transposition with Intel IPP
        //
        float * src = values;
        float * dst = other.values;
        IppiSize srcRoi = { (int) cols, (int) rows  };

        ippiTranspose_32f_C1R ( src, (int) cols * sizeof(float), dst, (int) rows * sizeof(float), srcRoi );

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
        const __m256i zero = _mm256_set1_epi8(0);
        for (uint64_t i = 0; i < n1; i += 32) {
            _mm256_storeu_si256((__m256i *)(u + i), zero);
        }
        for (uint64_t i = n1; i < n0; i += 1) {
            u[i] = 0;
        }
    }

    // ==============================================================================================================
    // = Random Matrix Initialization (testing purposes)
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
        const uint64_t length = size();
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
        const uint64_t length = size();
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

    float get(uint64_t i, uint64_t j) const {
        return values[i * cols + j];
    }
    void set(uint64_t i, uint64_t j, float alpha) {
        values[i * cols + j] = alpha;
    } 

    ~CloverMatrix32()
    {
        free(values);
    }

};

#endif /* CLOVER_MATRIX32_H */
