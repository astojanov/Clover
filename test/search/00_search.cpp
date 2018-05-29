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
#include <CloverMatrix4.h>
#include <CloverMatrix8.h>
#include <CloverMatrix16.h>

#include "../performance/03_iht_gd_util.h"
#include "../random/00_random.h"
#include "../performance/00_test.h"

using namespace std;


void print_experiment_params (int bitcount, uint64_t m, uint64_t n, uint64_t n_iter, float mu, float target_quality)
{
    std::cout << "Grid search complete";
    std::cout << " | ";
    std::cout << std::setw(2) << bitcount << "-bit";
    std::cout << " | ";
    std::cout << "Matrix: " << std::setw(5) << m << " x " << std::setw(5) << n;
    std::cout << " | ";
    std::cout << " Iterations:  " << std::setw(10) << n_iter;
    std::cout << " | ";
    std::cout << " mu:  " << std::setw(17) << std::fixed << std::setprecision(10) << mu;
    std::cout << " | ";
    std::cout << " Quality:  " << std::setw(17) << std::fixed << std::setprecision(10) << target_quality;
    std::cout << std::endl;
}

void GD_params(bool useMixedPrecision)
{
    uint64_t * matrix_ops_sizes = get_test_matrix_ops_sizes();
    uint64_t test_cases = get_test_matrix_ops_sizes_length();

    //
    // Gradient Decent is kind of opposite of IHT. Therefore, we
    // will have more rows than columns. In this particular case
    // we like to see rows = M_ratio * cols.
    //
    float M_ratio = 1.5;

    cout << "===================================================================================" << endl;
    cout << "= Performing Grid search for Quantized Gradient Descent" << endl;
    cout << "= Number of threads : " << CloverBase::get_OpenMP_threads() << endl;
    cout << "= rows / cols ratio : " << std::fixed << std::setprecision(2) << M_ratio << endl;
    if (useMixedPrecision) {
        cout << "= " << endl;
        cout << "= Note that this while running this experiment, the 4-bit version uses the mixed" << endl;
        cout << "= precision MVM, producing an 8-bit vector as a result of the iterations" << endl;
    } else {
        cout << "= Note that this experiment, uses pure 4-bit precision" << endl;
    }
    cout << "===================================================================================" << endl;
    cout << endl;

    for (int i = 0; i < test_cases; i += 1) {
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
        initialize_random_GD_values(x_32, Phi_32, y_32);

        float lo = 0.1;
        float hi = 0.5;
        float precision = 0.05;

        float quality_target, mu04, mu08, mu16,  mu32;
        uint64_t n_iter_04, n_iter_08, n_iter_16, n_iter_32;

        if (useMixedPrecision) {
            quality_target = GD_best_possible_quality<CloverMatrix4, CloverVector8> (x_32, Phi_32, y_32, lo, hi, precision) / 0.9f;
            n_iter_04 = GD_find_best_n_iterations<CloverMatrix4 , CloverVector8 > (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu04);
        } else {
            quality_target = GD_best_possible_quality<CloverMatrix4, CloverVector4> (x_32, Phi_32, y_32, lo, hi, precision) / 0.9f;
            n_iter_04 = GD_find_best_n_iterations<CloverMatrix4 , CloverVector4 > (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu04);
        }

        n_iter_08 = GD_find_best_n_iterations<CloverMatrix8 , CloverVector8 > (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu08);
        n_iter_16 = GD_find_best_n_iterations<CloverMatrix16, CloverVector16> (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu16);
        n_iter_32 = GD_find_best_n_iterations<CloverMatrix32, CloverVector32> (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu32);

        print_experiment_params (4 , m, n, n_iter_04, mu04, quality_target);
        print_experiment_params (8 , m, n, n_iter_08, mu08, quality_target);
        print_experiment_params (16, m, n, n_iter_16, mu16, quality_target);
        print_experiment_params (32, m, n, n_iter_32, mu32, quality_target);
        cout << endl;
    }

    cout << endl << endl;
}

void IHT_params(bool useMixedPrecision)
{
    uint64_t * matrix_ops_sizes = get_test_matrix_ops_sizes();
    uint64_t test_cases = get_test_matrix_ops_sizes_length();

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
    cout << "= Grid Search for Quantized Iterative Hard Thresholding" << endl;
    cout << "= " << endl;
    cout << "= Number of threads : " << CloverBase::get_OpenMP_threads() << endl;
    cout << "= cols / rows ratio : " << std::fixed << std::setprecision(2) << M_ratio << endl;
    cout << "= K ratio           : " << std::fixed << std::setprecision(1) << K_ratio * 100 << " %" << endl;
    cout << "= " << endl;
    if (useMixedPrecision) {
        cout << "= " << endl;
        cout << "= Note that this while running this experiment, the 4-bit version uses the mixed" << endl;
        cout << "= precision MVM, producing an 8-bit vector as a result of the iterations" << endl;
    } else {
        cout << "= Note that this experiment, uses pure 4-bit precision" << endl;
    }
    cout << "===================================================================================" << endl;
    cout << endl;


    for (int i = 0; i < test_cases; i += 1) {
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
        //
        // Set the control parameters
        //
        float lo = 0.000001;
        float hi = 0.5;
        float precision = 0.000001;

        float quality_reached, quality_target, mu32, mu16, mu08, mu04, mu04_initial;
        uint64_t n_iter_04_initial, n_iter_04, n_iter_08, n_iter_16, n_iter_32;

        //
        // Let's try to reached the best quality we can with the 4-bit
        //

        if (useMixedPrecision) {
            quality_reached = IHT_best_possible_quality<CloverMatrix4, CloverVector8>(
                    x_32, Phi_32, y_32, K, lo, hi, precision, n_iter_04_initial, mu04_initial
            );
        } else {
            quality_reached = IHT_best_possible_quality<CloverMatrix4, CloverVector4>(
                    x_32, Phi_32, y_32, K, lo, hi, precision, n_iter_04_initial, mu04_initial
            );
        }
        //
        // Once we reached that target, let's relax it, so the other versions can reach the same quality
        //
        quality_target = quality_reached / 0.98f;

        //
        // Now optimize for best number of iterations for the qiven quality target
        //
        if (useMixedPrecision) {
            n_iter_04 = IHT_find_best_n_iterations<CloverMatrix4, CloverVector8>(x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu04, K);
        } else {
            n_iter_04 = IHT_find_best_n_iterations<CloverMatrix4, CloverVector4>(x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu04, K);
        }
        n_iter_08 = IHT_find_best_n_iterations<CloverMatrix8 , CloverVector8 > (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu08, K);
        n_iter_16 = IHT_find_best_n_iterations<CloverMatrix16, CloverVector16> (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu16, K);
        n_iter_32 = IHT_find_best_n_iterations<CloverMatrix32, CloverVector32> (x_32, Phi_32, y_32, lo, hi, precision, quality_target, mu32, K);

        //
        // Since we already reached a better quality while searching for best quality, check whether those iterations
        // were in fact lower then once we optimized to find the best iterations, and adjust mu accordingly
        //
        if (n_iter_04_initial < n_iter_04) {
            n_iter_04 = n_iter_04_initial;
            mu04 = mu04_initial;
        }

        print_experiment_params (4 , m, n, n_iter_04, mu04, quality_target);
        print_experiment_params (8 , m, n, n_iter_08, mu08, quality_target);
        print_experiment_params (16, m, n, n_iter_16, mu16, quality_target);
        print_experiment_params (32, m, n, n_iter_32, mu32, quality_target);
        cout << endl;
    }

    cout << endl << endl;
}

void search(int argc, const char* argv[])
{
    init_deterministic_keys();

    //
    // First run the grid search for pure precision
    //
    GD_params(false);
    IHT_params(false);

    //
    // Then run the grid search for mixed precision
    //
    GD_params (true);
    IHT_params(true);

}