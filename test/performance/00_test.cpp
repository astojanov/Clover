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

#include <iostream>

#include <CloverVector32.h>
#include <CloverVector16.h>
#include <CloverVector8.h>
#include <CloverVector4.h>
#include <CloverMatrix32.h>
#include <CloverMatrix16.h>
#include <CloverMatrix8.h>
#include <simdxorshift128plus.h>
#include <CloverMatrix4.h>

#include "../../lib/perf.h"

#include "01_measure.h"
#include "02_bit04.h"
#include "02_bit08.h"
#include "02_bit16.h"
#include "02_bit32.h"
#include "03_iht_gd_util.h"

using namespace std;

uint64_t vector_ops_sizes [] = {
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
        268435456,
        536870912
};


uint64_t matrix_ops_sizes [] = {
        128 * 2,
        128 * 4,
        128 * 8,
        128 * 16,
        128 * 32,
        128 * 48,
        128 * 64,
        128 * 80,
        128 * 96,
        128 * 112,
        128 * 128,
        128 * 144,
        128 * 160,
        128 * 176,
        128 * 192,
        128 * 208,
        128 * 224,
        128 * 240,
        128 * 256
};

measurement_t empty_measurement = { {0, 0}, 0};

uint64_t * get_test_vector_ops_sizes ()
{
    return vector_ops_sizes;
}

uint64_t * get_test_matrix_ops_sizes ()
{
    return matrix_ops_sizes;
}

uint64_t get_test_vector_ops_sizes_length ()
{
    return sizeof(vector_ops_sizes) / sizeof(uint64_t);
}

uint64_t get_test_matrix_ops_sizes_length ()
{
    return sizeof(matrix_ops_sizes) / sizeof(uint64_t);
}

void print_header ()
{
    std::cout << "| ";
    std::cout << std::setw(10) << "N";
    std::cout << " | ";
    std::cout << std::setw(13) << "32-bit cycles";
    std::cout << " | ";
    std::cout << std::setw(13) << "16-bit cycles";
    std::cout << " | ";
    std::cout << std::setw(13) << " 8-bit cycles";
    std::cout << " | ";
    std::cout << std::setw(13) << " 4-bit cycles";

    std::cout << " |||| ";

    std::cout << std::setw(17) << std::fixed << std::setprecision(2) << "32-bit bandwidth";
    std::cout << " | ";
    std::cout << std::setw(17) << std::fixed << std::setprecision(2) << "16-bit bandwidth";
    std::cout << " | ";
    std::cout << std::setw(17) << std::fixed << std::setprecision(2) << " 8-bit bandwidth";
    std::cout << " | ";
    std::cout << std::setw(17) << std::fixed << std::setprecision(2) << " 4-bit bandwidth";
    std::cout << " |||| ";

    std::cout << std::setw(14) << std::fixed << std::setprecision(2) << "32-bit / 16-bit";
    std::cout << " | ";
    std::cout << std::setw(14) << std::fixed << std::setprecision(2) << "32-bit /  8-bit";
    std::cout << " | ";
    std::cout << std::setw(14) << std::fixed << std::setprecision(2) << "32-bit /  4-bit";
    std::cout << " | ";
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
    std::cout << std::endl;
}

std::string toStringMBPerSecond (double bytes)
{
    const double mb = 1024 * 1024;
    std::stringstream sout;
    if (bytes != 0) {
        sout << std::setw(12) << std::fixed << std::setprecision(2) << bytes / mb << " MB/s";
    } else {
        sout << std::setw(12) << std::fixed << std::setprecision(2) << "" << "     ";
    }
    return sout.str();
}

std::string toStringRatio (uint64_t cycles_1, uint64_t cycles_2)
{
    std::stringstream sout;
    if (cycles_1 == 0 || cycles_2 == 0) {
        std::cout << std::setw(14) << std::fixed << std::setprecision(2) << "" << " ";
    } else {
        std::cout << std::setw(14) << std::fixed << std::setprecision(2) << (double) cycles_1 / (double) cycles_2 << "x";
    }
    return sout.str();
}

void print_measurement (uint64_t size, measurement_t &m04, measurement_t &m08, measurement_t &m16, measurement_t &m32)
{
    const uint64_t c_vector_04 = m04.bench.cycles;
    const double   b_vector_04 = m04.bandwidth;
    const uint64_t c_vector_08 = m08.bench.cycles;
    const double   b_vector_08 = m08.bandwidth;
    const uint64_t c_vector_16 = m16.bench.cycles;
    const double   b_vector_16 = m16.bandwidth;
    const uint64_t c_vector_32 = m32.bench.cycles;
    const double   b_vector_32 = m32.bandwidth;


    std::cout << left << "| ";
    std::cout << std::setw(10) << size;
    std::cout << " | ";
    std::cout << std::setw(13) << c_vector_32;
    std::cout << " | ";
    std::cout << std::setw(13) << c_vector_16;
    std::cout << " | ";
    std::cout << std::setw(13) << c_vector_08;
    std::cout << " | ";
    std::cout << std::setw(13) << c_vector_04;

    std::cout << " |||| " << right;

    std::cout << toStringMBPerSecond(b_vector_32);
    std::cout << " | ";
    std::cout << toStringMBPerSecond(b_vector_16);
    std::cout << " | ";
    std::cout << toStringMBPerSecond(b_vector_08);
    std::cout << " | ";
    std::cout << toStringMBPerSecond(b_vector_04);
    std::cout << " |||| ";
    std::cout << toStringRatio(c_vector_32, c_vector_16);
    std::cout << " | ";
    std::cout << toStringRatio(c_vector_32, c_vector_08);
    std::cout << " | ";
    std::cout << toStringRatio(c_vector_32, c_vector_04);
    std::cout << " | ";
    std::cout << std::endl;
}

// ===============================================================================================================
// = Vector operations
// ===============================================================================================================

void testing_vector_quantize()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Vector Quantization" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_quantize_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_quantize_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_quantize_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_quantize_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_quantize_parallel()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);
    int nthr = CloverBase::get_OpenMP_threads();

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Vector Quantization running: " << nthr << " threads" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_quantize_parallel_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_quantize_parallel_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_quantize_parallel_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_quantize_parallel_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_get()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Vector get()" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_get_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_get_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_get_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_get_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_dot()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Vector Dot Product" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_dot_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_dot_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_dot_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_dot_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_dot_parallel()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);
    int nthr = CloverBase::get_OpenMP_threads();

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Vector Dot Product running: " << nthr << " threads" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_dot_parallel_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_dot_parallel_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_dot_parallel_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_dot_parallel_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_scaleandadd()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Vector Scale-and-Add" << endl;
    cout << "===================================================================================" << endl;

    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_scaleandadd_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_scaleandadd_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_scaleandadd_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_scaleandadd_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_scaleandadd_parallel()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t);
    int nthr = CloverBase::get_OpenMP_threads();

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Vector Scale-and-Add running: " << nthr << " threads" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_vector_scaleandadd_parallel_04(vector_ops_sizes[i]);
        measurement_t m08 = measure_vector_scaleandadd_parallel_08(vector_ops_sizes[i]);
        measurement_t m16 = measure_vector_scaleandadd_parallel_16(vector_ops_sizes[i]);
        measurement_t m32 = measure_vector_scaleandadd_parallel_32(vector_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_threshold()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t) * 2 / 3;

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Vector Thresholding" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {
        const uint64_t size = vector_ops_sizes[i];
        const uint64_t k = size / 16;

        measurement_t m04 = measure_vector_threshold_04(size, k);
        measurement_t m08 = measure_vector_threshold_08(size, k);
        measurement_t m16 = measure_vector_threshold_16(size, k);
        measurement_t m32 = measure_vector_threshold_32(size, k);

        print_measurement(size, m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_vector_threshold_parallel()
{
    int n = sizeof(vector_ops_sizes) / sizeof(uint64_t) * 2 / 3;
    int nthr = CloverBase::get_OpenMP_threads();

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Vector Thresholding running: " << nthr << " threads" << endl;
    cout << "===================================================================================" << endl;

    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        const uint64_t size = vector_ops_sizes[i];
        const uint64_t k = size / 16;

        measurement_t m04 = measure_vector_threshold_parallel_04(size, k);
        measurement_t m08 = measure_vector_threshold_parallel_08(size, k);
        measurement_t m16 = measure_vector_threshold_parallel_16(size, k);
        measurement_t m32 = measure_vector_threshold_parallel_32(size, k);

        print_measurement(size, m04, m08, m16, m32);
    }

    cout << endl << endl;
}


// ===============================================================================================================
// = Matrix operations
// ===============================================================================================================

void testing_matrix_quantize()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Matrix Quantization" << endl;
    cout << "===================================================================================" << endl;

    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_quantize_04(matrix_ops_sizes[i]);
        measurement_t m08 = measure_matrix_quantize_08(matrix_ops_sizes[i]);
        measurement_t m16 = measure_matrix_quantize_16(matrix_ops_sizes[i]);
        measurement_t m32 = measure_matrix_quantize_32(matrix_ops_sizes[i]);

        print_measurement(vector_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential MVM" << endl;
    cout << "===================================================================================" << endl;

    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_04(matrix_ops_sizes[i]);
        measurement_t m08 = measure_matrix_MVM_08(matrix_ops_sizes[i]);
        measurement_t m16 = measure_matrix_MVM_16(matrix_ops_sizes[i]);
        measurement_t m32 = measure_matrix_MVM_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM_mixed_mat04_vec08()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Mixed MVM: 4-bit Matrix and 8-bit Vector" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_mixed_mat04_vec08(matrix_ops_sizes[i]);
        measurement_t m08 = empty_measurement;
        measurement_t m16 = empty_measurement;
        measurement_t m32 = measure_matrix_MVM_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM_mixed_mat04_vec32()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Mixed MVM 4-bit Matrix and 32-bit Vector" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_mixed_mat04_vec32(matrix_ops_sizes[i]);
        measurement_t m08 = empty_measurement;
        measurement_t m16 = empty_measurement;
        measurement_t m32 = measure_matrix_MVM_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM_parallel()
{
    int nthr = CloverBase::get_OpenMP_threads();
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel MVM running: " << nthr << " Threads" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_parallel_04(matrix_ops_sizes[i]);
        measurement_t m08 = measure_matrix_MVM_parallel_08(matrix_ops_sizes[i]);
        measurement_t m16 = measure_matrix_MVM_parallel_16(matrix_ops_sizes[i]);
        measurement_t m32 = measure_matrix_MVM_parallel_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM_parallel_mixed_mat04_vec08()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Mixed MVM: 4-bit Matrix and 8-bit Vector" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_parallel_mixed_mat04_vec08(matrix_ops_sizes[i]);
        measurement_t m08 = empty_measurement;
        measurement_t m16 = empty_measurement;
        measurement_t m32 = measure_matrix_MVM_parallel_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_MVM_parallel_mixed_mat04_vec32()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Mixed MVM 4-bit Matrix and 32-bit Vector" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_MVM_parallel_mixed_mat04_vec32(matrix_ops_sizes[i]);
        measurement_t m08 = empty_measurement;
        measurement_t m16 = empty_measurement;
        measurement_t m32 = measure_matrix_MVM_parallel_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}


void testing_matrix_transpose()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Sequential Matrix Transpose" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();


    for (int i = 0; i < n; i += 1) {

        measurement_t m04 = measure_matrix_transpose_04(matrix_ops_sizes[i]);
        measurement_t m08 = measure_matrix_transpose_08(matrix_ops_sizes[i]);
        measurement_t m16 = measure_matrix_transpose_16(matrix_ops_sizes[i]);
        measurement_t m32 = measure_matrix_transpose_32(matrix_ops_sizes[i]);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_matrix_transpose_parallel()
{
    int n = sizeof(matrix_ops_sizes) / sizeof(uint64_t);

    cout << "===================================================================================" << endl;
    cout << "= Speed of Parallel Matrix Transpose" << endl;
    cout << "===================================================================================" << endl;
    cout << endl;
    print_header ();

    for (int i = 0; i < n; i += 1) {
        measurement_t m04 = measure_matrix_transpose_parallel_04(matrix_ops_sizes[i]);
        measurement_t m08 = empty_measurement; // measure_matrix_transpose_parallel_08(matrix_ops_sizes[i]);
        measurement_t m16 = empty_measurement; // measure_matrix_transpose_parallel_16(matrix_ops_sizes[i]);
        measurement_t m32 = measure_matrix_transpose_parallel_32(matrix_ops_sizes[i]);
        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

// ===============================================================================================================
// = Quantized Linear Algebra Application
// ===============================================================================================================


void testing_IHT(bool useMixedPrecision, bool useFixedIterations)
{
    int test_cases = sizeof(matrix_ops_sizes) / sizeof(uint64_t);
    experiment_setup_t * iht_experiments;

    if (useMixedPrecision) {
        iht_experiments = setup_IHT_experiment_mixed();
    } else {
        iht_experiments = setup_IHT_experiment_pure();
    }

    //
    // We would like to choose parameters such that we make the problem
    // as hard as we can. According to [1], particularly Figures 7.1 and 7.2
    // we can observe that reconstruction becomes harder for a matrix of
    // size m x n when n = 2 * m, or in other words, there are more columns
    // than rows. Also the problem is harder when sparsity is about 1/4 of
    // the vector that we want to re-construct.
    //
    // [1] Compressive sensing a summary of reconstruction algorithms
    //     Master thesis at ETH by Pope, Graeme
    //
    problem_type_t problem = QUANTIZED_ITERATIVE_HARD_THRESHOLDING;
    float M_ratio = 2.0f;
    float K_ratio = 1.0f / 4.0f;

    cout << "===================================================================================" << endl;
    cout << "= Performance test for Quantized Iterative Hard Thresholding" << endl;
    cout << "= " << endl;
    cout << "= Number of threads : " << CloverBase::get_OpenMP_threads() << endl;
    cout << "= cols / rows ratio : " << std::fixed << std::setprecision(2) << M_ratio << endl;
    cout << "= K ratio           : " << std::fixed << std::setprecision(1) << K_ratio * 100 << " %" << endl;
    cout << "= " << endl;
    if (useFixedIterations) {
        cout << "= In this particular experiment we are running fixed number of 100 iterations" << endl;
    } else {
        cout << "= In this particular experiment we try to run the best mu and determine the number" << endl;
        cout << "= of iterations. To achieve that we use offline grid search." << endl;
    }
    cout << "= " << endl;
    if (useMixedPrecision) {
        cout << "= Note that this while running this experiment, the 4-bit version uses the mixed" << endl;
        cout << "= precision MVM, producing an 8-bit vector as a result of the iterations" << endl;
    } else {
        cout << "= Note that this experiment, uses pure 4-bit precision" << endl;
    }
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < test_cases; i += 1)
    {
        //
        // Define the values for the matrix, as MxN
        //
        const uint64_t m_initial = matrix_ops_sizes[i];
        const uint64_t n_initial = (uint64_t)(m_initial * M_ratio);
        //
        // Let's initialize the matrix
        //
        CloverMatrix32 Phi_32(m_initial, n_initial);
        //
        // Due to padding these values might as well change
        //
        const uint64_t m = Phi_32.getRows();
        const uint64_t n = Phi_32.getCols();
        //
        // Value initialization
        //
        CloverVector32 x_32(n);
        CloverVector32 y_32(m);
        uint64_t K = (uint64_t) (n * K_ratio);
        initialize_random_IHT_values(x_32, Phi_32, y_32, K);

        measurement_t m04, m08, m16, m32;
        uint64_t iter04, iter08, iter16, iter32;

        //
        // Setup the mu-s for our experiment
        //
        float mu04 = iht_experiments[i].bit04.mu;
        float mu08 = iht_experiments[i].bit08.mu;
        float mu16 = iht_experiments[i].bit16.mu;
        float mu32 = iht_experiments[i].bit32.mu;

        //
        // Setup the number of iterations
        //
        if (useFixedIterations) {
            iter04 = 100;
            iter08 = 100;
            iter16 = 100;
            iter32 = 100;
        } else {
            iter04 = iht_experiments[i].bit04.iterations;
            iter08 = iht_experiments[i].bit08.iterations;
            iter16 = iht_experiments[i].bit16.iterations;
            iter32 = iht_experiments[i].bit32.iterations;
        }

        //
        // Now runt the whole test
        //
        if (useMixedPrecision) {
            m04 = measure_IHT_or_GD_mixed_mat4_vec8(problem, x_32, Phi_32, y_32, K, mu04, iter04);
        } else {
            m04 = measure_IHT_or_GD_04(problem, x_32, Phi_32, y_32, K, mu04, iter04);
        }
        m08 = measure_IHT_or_GD_08(problem, x_32, Phi_32, y_32, K, mu08, iter08);
        m16 = measure_IHT_or_GD_16(problem, x_32, Phi_32, y_32, K, mu16, iter16);
        m32 = measure_IHT_or_GD_32(problem, x_32, Phi_32, y_32, K, mu32, iter32);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_GD(bool useMixedPrecision, bool useFixedIterations)
{
    int test_cases = sizeof(matrix_ops_sizes) / sizeof(uint64_t);
    experiment_setup_t * gd_experiments;

    if (useMixedPrecision) {
        gd_experiments = setup_GD_experiment_mixed();
    } else {
        gd_experiments = setup_GD_experiment_pure();
    }

    //
    // Gradient Decent is kind of opposite of IHT. Therefore, we
    // will have more rows than columns. In this particular case
    // we like to see rows = M_ratio * cols.
    //
    problem_type_t problem = QUANTIZED_GRADIENT_DESCENT;
    float M_ratio = 1.5;

    cout << "===================================================================================" << endl;
    cout << "= Performance test for Quantized Gradient Descent" << endl;
    cout << "= " << endl;
    cout << "= Number of threads : " << CloverBase::get_OpenMP_threads() << endl;
    cout << "= rows / cols ratio : " << std::fixed << std::setprecision(2) << M_ratio << endl;
    cout << "= " << endl;
    if (useFixedIterations) {
        cout << "= In this particular experiment we are running fixed number of 100 iterations" << endl;
    } else {
        cout << "= In this particular experiment we try to run the best mu and determine the number" << endl;
        cout << "= of iterations. To achieve that we use offline grid search." << endl;
    }
    cout << "= " << endl;
    if (useMixedPrecision) {
        cout << "= Note that this while running this experiment, the 4-bit version uses the mixed" << endl;
        cout << "= precision MVM, producing an 8-bit vector as a result of the iterations" << endl;
    } else {
        cout << "= Note that this experiment, uses pure 4-bit precision" << endl;
    }
    cout << "===================================================================================" << endl;
    cout << endl;

    print_header ();

    for (int i = 0; i < test_cases; i += 1)
    {
        //
        // Define the values for the matrix, as MxN
        //
        const uint64_t n_initial = matrix_ops_sizes[i];
        const uint64_t m_initial = (uint64_t)(n_initial * M_ratio);
        //
        // Let's initialize the matrix
        //
        CloverMatrix32 Phi_32(m_initial, n_initial);
        //
        // Due to padding these values might as well change
        //
        const uint64_t m = Phi_32.getRows();
        const uint64_t n = Phi_32.getCols();
        //
        // Let's initialize the matrix
        //
        CloverVector32 x_32(n);
        CloverVector32 y_32(m);
        uint64_t K = 0; // not needed
        initialize_random_GD_values(x_32, Phi_32, y_32);

        measurement_t m04, m08, m16, m32;
        uint64_t iter04, iter08, iter16, iter32;

        float mu04 = gd_experiments[i].bit04.mu;
        float mu08 = gd_experiments[i].bit08.mu;
        float mu16 = gd_experiments[i].bit16.mu;
        float mu32 = gd_experiments[i].bit32.mu;

        //
        // Setup the number of iterations
        //
        if (useFixedIterations) {
            iter04 = 100;
            iter08 = 100;
            iter16 = 100;
            iter32 = 100;
        } else {
            iter04 = gd_experiments[i].bit04.iterations;
            iter08 = gd_experiments[i].bit08.iterations;
            iter16 = gd_experiments[i].bit16.iterations;
            iter32 = gd_experiments[i].bit32.iterations;
        }
        //
        // Now run the experiments
        //
        if (useMixedPrecision) {
            m04 = measure_IHT_or_GD_mixed_mat4_vec8(problem, x_32, Phi_32, y_32, K, mu04, iter04);
        } else {
            m04 = measure_IHT_or_GD_04(problem, x_32, Phi_32, y_32, K, mu04, iter04);
        }
        m08 = measure_IHT_or_GD_08(problem, x_32, Phi_32, y_32, K, mu08, iter08);
        m16 = measure_IHT_or_GD_16(problem, x_32, Phi_32, y_32, K, mu16, iter16);
        m32 = measure_IHT_or_GD_32(problem, x_32, Phi_32, y_32, K, mu32, iter32);

        print_measurement(matrix_ops_sizes[i], m04, m08, m16, m32);
    }

    cout << endl << endl;
}

void testing_IHT_fixed_pure    () { testing_IHT(false, true ); }
void testing_IHT_fixed_mixed   () { testing_IHT(true , true ); }
void testing_IHT_optimal_pure  () { testing_IHT(false, false); }
void testing_IHT_optimal_mixed () { testing_IHT(true , false); }

void testing_GD_fixed_pure     () { testing_GD(false, true ); }
void testing_GD_fixed_mixed    () { testing_GD(true , true ); }
void testing_GD_optimal_pure   () { testing_GD(false, false); }
void testing_GD_optimal_mixed  () { testing_GD(true , false); }


void test (int argc, const char* argv[]) {

    init_deterministic_keys();

    // ===============================================================================================================
    // = Vector operations
    // ===============================================================================================================

    testing_vector_quantize();
    testing_vector_quantize_parallel();
    testing_vector_get();
    testing_vector_dot();
    testing_vector_dot_parallel();
    testing_vector_scaleandadd();
    testing_vector_scaleandadd_parallel();
    testing_vector_threshold();
    testing_vector_threshold_parallel();

    // ===============================================================================================================
    // = Matrix operations
    // ===============================================================================================================

    testing_matrix_quantize();
    testing_matrix_MVM();
    testing_matrix_MVM_mixed_mat04_vec08();
    testing_matrix_MVM_mixed_mat04_vec32();
    testing_matrix_MVM_parallel();
    testing_matrix_MVM_parallel_mixed_mat04_vec08();
    testing_matrix_MVM_parallel_mixed_mat04_vec32();
    testing_matrix_transpose();
    testing_matrix_transpose_parallel();

    // ===============================================================================================================
    // = Quantized Linear Algebra Application
    // ===============================================================================================================

    testing_IHT_fixed_pure    ();
    testing_IHT_fixed_mixed   ();
    testing_IHT_optimal_pure  ();
    testing_IHT_optimal_mixed ();

    testing_GD_fixed_pure     ();
    testing_GD_fixed_mixed    ();
    testing_GD_optimal_pure   ();
    testing_GD_optimal_mixed  ();

}
