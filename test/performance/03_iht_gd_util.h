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


#ifndef CLOVER_IHT_GD_UTIL_H
#define CLOVER_IHT_GD_UTIL_H

#include <CloverVector32.h>
#include <CloverMatrix32.h>
#include "../accuracy/01_math.h"

typedef enum {
    QUANTIZED_ITERATIVE_HARD_THRESHOLDING,
    QUANTIZED_GRADIENT_DESCENT
} problem_type_t;

typedef struct {
    uint64_t iterations;
    float mu;
    float quality;
} problem_params_t;

typedef struct {
    problem_params_t bit04;
    problem_params_t bit08;
    problem_params_t bit16;
    problem_params_t bit32;
} experiment_setup_t;


template <class QMatrix, class QVector>
uint64_t determine_IHT_or_GD_iterations (
        problem_type_t problem_type,
        CloverVector32 &x_real,
        QMatrix &Phi,
        QVector &y,
        double mu,
        uint64_t K,
        uint64_t iteration_limit,
        //
        // Temporary variables, so we avoid reinitialization
        //
        CloverVector32 &x_restored,
        QVector &x,
        QMatrix &PhiT,
        QVector &t1,
        QVector &t2,
        QVector &t3,
        double x_real_norm2,
        float quality_target
) {
    //
    // Initialize
    //
    x.clear();
    double curr_loss;
    uint64_t i;
    for (i = 0; i < iteration_limit; i += 1)
    {
        Phi.mvm_parallel(x, t1);                        // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);          // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);                      // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, (float) mu);         // x = x + \mu * Phi' * (y - Phi * x)
        //
        // If dealing with IHT, just make sure that we threshold
        //
        switch (problem_type) {
            case QUANTIZED_ITERATIVE_HARD_THRESHOLDING: {
                x.threshold(K);
                break;
            }
            case QUANTIZED_GRADIENT_DESCENT: {
                //
                // Don't do anything
                //
                break;
            }
        }
        //
        // Calculate loss
        //
        x.restore(x_restored);
        x_restored.scaleAndAdd(x_real, -1);
        curr_loss = norm2(x_restored) / x_real_norm2;

        if (std::isnan(curr_loss)) {
            i = iteration_limit;
            break;
        }

        if (curr_loss <= quality_target){
            break;
        }
    }

    return i;
};

template <class QMatrix, class QVector>
bool is_IHT_or_GD_convergent(
        problem_type_t problem_type,
        CloverVector32 &x_real,
        QMatrix &Phi,
        QVector &y,
        double mu,
        uint64_t K,
        uint64_t iteration_limit,
        //
        // Temporary variables, so we avoid reinitialization
        //
        CloverVector32 &x_restored,
        QVector &x,
        QMatrix &PhiT,
        QVector &t1,
        QVector &t2,
        QVector &t3,
        double x_real_norm2,
        float &quality
) {

    bool   is_convergent = true;
    double prev_loss = std::numeric_limits<float>::max();
    double curr_loss = 0;
    double improvement;

    //
    // Initialize
    //
    // uint64_t iteration;
    x.clear();
    quality = -std::numeric_limits<float>::max();

    for (uint64_t i = 0; i < iteration_limit; i += 1)
    {
        Phi.mvm_parallel(x, t1);                        // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);          // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);                      // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, (float) mu);         // x = x + \mu * Phi' * (y - Phi * x)
        //
        // If dealing with IHT, just make sure that we threshold
        //
        switch (problem_type) {
            case QUANTIZED_ITERATIVE_HARD_THRESHOLDING: {
                x.threshold(K);
                break;
            }
            case QUANTIZED_GRADIENT_DESCENT: {
                //
                // Don't do anything
                //
                break;
            }
        }
        //
        // Now calculate the reconstruction rate
        //
        x.restore(x_restored);
        x_restored.scaleAndAdd(x_real, -1);             // t3 = x - x_real

        curr_loss = norm2(x_restored) / x_real_norm2;
        improvement = prev_loss - curr_loss;
        prev_loss = curr_loss;

        if (std::isnan(curr_loss)) {
            is_convergent = false;
            break;
        }

        if (improvement >= 0 && improvement < 0.001) {
            break;
        }
    }

    if (is_convergent) {
        is_convergent = curr_loss < 2;
    }

    if (is_convergent) {
        quality = (float) curr_loss;
    }

    return is_convergent;
};

template <class QMatrix, class QVector>
float GD_best_possible_quality (
        CloverVector32 &x_real,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        float lo, float hi, float precision
) {
    const uint64_t m = Phi_32.getRows();
    const uint64_t n = Phi_32.getCols();

    CloverVector32 x_restored(n);

    QMatrix Phi(m, n);
    QVector x(n);
    QVector y(m);
    QMatrix PhiT(n, m);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);

    Phi.quantize(Phi_32);
    y.quantize(y_32);
    Phi.transpose(PhiT);
    double x_real_norm2 = norm2(x_real);

    std::cout << "Looking for best possible quality  for ";
    std::cout << std::setw(2) << Phi.getBitsLength() << "-bit matrix and ";
    std::cout << std::setw(2) << x.getBitsLength() << "-bit vector ";
    std::cout << std::endl;
    uint64_t iterationLimit = 50;

    //
    // Fail fast if it doesn't converge for the low end of our grid search
    //
    float lo_quality = -std::numeric_limits<float>::max();
    bool lo_conv = is_IHT_or_GD_convergent (
            QUANTIZED_GRADIENT_DESCENT,
            x_real, Phi, y, lo, 0, iterationLimit, x_restored, x, PhiT, t1, t2, t3, x_real_norm2, lo_quality
    );
    if (!lo_conv) {
        std::cout << "is_GD_convergent fails because this GD does not converge for mu = " << lo << std::endl;
        std::cout << "This should never ever happen" << std::endl;
        exit(1);
    }

    //
    // Perform the grid search
    //
    float best_quality = lo_quality;
    float current_mu = lo + precision;

    while (current_mu < hi) {
        float current_quality;
        bool is_convergent = is_IHT_or_GD_convergent(
                QUANTIZED_GRADIENT_DESCENT,
                x_real, Phi, y, current_mu, 0, iterationLimit, x_restored, x, PhiT, t1, t2, t3, x_real_norm2, current_quality
        );
        if (is_convergent) {
            std::cout << "Convergence completed for mu = " << current_mu <<  " and quality parameter: " << current_quality  << std::endl;
            if (current_quality < best_quality){
                best_quality = current_quality;
            }
        } else {
            break;
        }
        current_mu += precision;
    }

    std::cout << std::endl;
    return best_quality;
}

template <class QMatrix, class QVector>
uint64_t GD_find_best_n_iterations (
        CloverVector32 &x_real,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        float lo, float hi, float precision,
        float quality_target,
        float &mu
) {
    const uint64_t m = Phi_32.getRows();
    const uint64_t n = Phi_32.getCols();

    CloverVector32 x_restored(n);

    QVector x(n);
    QMatrix Phi(m,n);
    QMatrix PhiT(n, m);
    QVector y(m);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);

    Phi.quantize(Phi_32);
    y.quantize(y_32);
    Phi.transpose(PhiT);
    double x_real_norm2 = norm2(x_real);

    std::cout << "Finding mu for ";
    std::cout << std::setw(2) << Phi.getBitsLength() << "-bit matrix and ";
    std::cout << std::setw(2) << x.getBitsLength() << "-bit vector ";
    std::cout << std::endl;

    uint64_t iteration_limit = 50;

    //
    // Perform the grid search
    //
    float best_mu = lo;
    uint64_t best_iterations = iteration_limit;

    for (float current_mu = lo; current_mu < hi; current_mu += precision)
    {
        uint64_t current_iterations = determine_IHT_or_GD_iterations(
                QUANTIZED_GRADIENT_DESCENT,
                x_real, Phi, y, current_mu, 0, iteration_limit, x_restored, x, PhiT, t1, t2, t3, x_real_norm2,
                quality_target
        );
        std::cout << "Convergence completed for mu = " << current_mu <<  " and iterations: " << current_iterations  << std::endl;
        if (current_iterations < best_iterations) {
            best_iterations =  current_iterations;
            best_mu = current_mu;
        }
    }
    std::cout << std::endl;

    mu = best_mu;
    return best_iterations;
}


template <class QMatrix, class QVector>
bool profile_IHT (
        QMatrix &Phi,
        QMatrix &PhiT,
        QVector &x,
        QVector &y,
        double mu,
        uint64_t K,
        //
        // Control parameters
        //
        CloverVector32 &x_real,
        double x_real_norm2,
        uint64_t iteration_limit,
        float quality_target,
        //
        // Tuning parameters
        //
        float &quality,
        uint64_t &iteration_count,
        //
        // Temporary variables, so we avoid reinitialization
        //
        CloverVector32 &x_restored,
        QVector &t1,
        QVector &t2,
        QVector &t3
) {

    bool  is_convergent = true;
    float curr_loss = 0;

    //
    // Initialize
    //
    x.clear();
    quality = std::numeric_limits<float>::max();

    for (uint64_t i = 0; i < iteration_limit; i += 1)
    {
        Phi.mvm_parallel(x, t1);                        // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);          // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);                      // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, (float) mu);         // x = x + \mu * Phi' * (y - Phi * x)
        x.threshold(K);

        //
        // Now calculate the reconstruction rate
        //
        x.restore(x_restored);
        x_restored.scaleAndAdd(x_real, -1);             // t3 = x - x_real
        curr_loss = (float) (norm2(x_restored) / x_real_norm2);

        if (std::isnan(curr_loss)) {
            //
            // If things seem funky, snap out of it
            //
            is_convergent = false;
            break;
        } else {
            //
            // Get the loss ratio with the already reached quality
            //
            float loss_ratio = quality / curr_loss;

            //
            // Once we have reached the minimal quality
            // make sure we note it down
            //
            if (loss_ratio > 1) {
                quality = curr_loss;
                iteration_count = i + 1;
                //
                // If we have reached the target quality, quit
                //
                if (curr_loss < quality_target) {
                    break;
                }
            }
//            else if (loss_ratio < 0.25) {
//                is_convergent = false;
//                break;
//            }
        }

    }

    //
    // Sanity check
    //
    if (is_convergent) {
        is_convergent = curr_loss < 2;
    }

    return is_convergent;
};

template <typename T>
uint64_t min_index (T * arr, uint64_t size)
{
    uint64_t min_idx = 0;
    for (uint64_t i = 1; i < size; i += 1) {
        if (arr[i] < arr[min_idx]) {
            min_idx = i;
        }
    }
    return min_idx;
}

template <class QMatrix, class QVector>
float IHT_best_possible_quality (
        CloverVector32 &x_real,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        uint64_t K,
        float lo_initial, float hi, float precision,
        uint64_t &n_iter, float &best_mu
) {
    //
    // Initialize the constraints
    //
    uint64_t iteration_count = 0;
    uint64_t best_iteration_count = 0;
    uint64_t iteration_limit = 50;
    uint64_t grid_size = 10;

    float mu, lo = lo_initial;
    float best_quality = -std::numeric_limits<float>::max();
    float curr_quality = -std::numeric_limits<float>::max();

    bool is_convergent     = false;
    bool upper_bound_found = false;

    //
    // Get the dimensions
    //
    const uint64_t m = Phi_32.getRows();
    const uint64_t n = Phi_32.getCols();
    //
    // Setup the matrices and vectors
    //
    QMatrix Phi(m, n);
    QMatrix PhiT(n, m);
    QVector x(n);
    QVector y(m);

    CloverVector32 x_restored(n);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);
    //
    // Feed the data
    //
    Phi.quantize(Phi_32);
    y.quantize(y_32);
    Phi.transpose(PhiT);
    const double x_real_norm2 = norm2(x_real);
    //
    // Let's begin
    //
    std::cout << "Looking for best possible quality for ";
    std::cout << std::setw(2) << Phi.getBitsLength() << "-bit matrix and ";
    std::cout << std::setw(2) << x.getBitsLength() << "-bit vector ";
    std::cout << std::endl;


    //
    // Fail fast if it doesn't converge for the low end of our grid search
    //
    is_convergent = profile_IHT (
            Phi, PhiT, x, y, lo, K,
            x_real, x_real_norm2, iteration_limit, -std::numeric_limits<float>::max(),
            curr_quality, iteration_count,
            x_restored, t1, t2, t3
    );
    if (!is_convergent) {
        std::cout << "IHT Convergence fails because this GD does not converge for mu = " << lo << std::endl;
        std::cout << "This should never ever happen" << std::endl;
        exit(1);
    }
    best_quality = curr_quality;

    //
    // Start a binary search to find the largest mu that this problem would converge
    //
    while (lo + precision <= hi)
    {
        mu = (lo + hi) / 2;

        is_convergent = profile_IHT (
                Phi, PhiT, x, y, mu, K,
                x_real, x_real_norm2, iteration_limit, -std::numeric_limits<float>::max(),
                curr_quality, iteration_count,
                x_restored, t1, t2, t3
        );

        if (is_convergent) {
            lo = mu;
            if (curr_quality < best_quality) {
                best_mu = mu;
                best_quality = curr_quality;
                best_iteration_count = iteration_count;
            }
            upper_bound_found = true;
        } else {
            hi = mu;
        }

        std::cout << "Convergence for mu = " << std::setw(17) << std::fixed << std::setprecision(10) << mu << ": " << (is_convergent ? "OK  " : "Fail");
        std::cout << " | ";
        std::cout << "Quality: " << curr_quality;
        std::cout << " | ";
        std::cout << "Iterations: " << iteration_count;
        std::cout << std::endl;
    }

    if (!upper_bound_found) {
        std::cout << "Upper bound was not found. Exiting ..." << std::endl;
        exit(1);
    } else {
        std::cout << "Upper bound found. Let's start with grid search ..." << std::endl << std::endl;
    }

    //
    // Now perform grid search such that the grid is binary adjusted withing a given range
    //
    lo = lo_initial;
    float grid_qualities[grid_size + 10];

    while (lo + precision <= hi) {

        float step = (hi - lo) / grid_size;

        for (uint64_t i = 0; i <= grid_size; i += 1) {
            mu = lo + step * i;
            is_convergent = profile_IHT (
                    Phi, PhiT, x, y, mu, K,
                    x_real, x_real_norm2, iteration_limit, -std::numeric_limits<float>::max(),
                    curr_quality, iteration_count,
                    x_restored, t1, t2, t3
            );

            std::cout << "Convergence for mu = " << std::setw(17) << std::fixed << std::setprecision(10) << mu << ": " << (is_convergent ? "OK  " : "Fail");
            std::cout << " | ";
            std::cout << "Quality: " << curr_quality;
            std::cout << " | ";
            std::cout << "Iterations: " << iteration_count;
            std::cout << std::endl;


            if (is_convergent) {
                grid_qualities[i] = curr_quality;
                if (curr_quality < best_quality) {
                    best_mu = mu;
                    best_quality = curr_quality;
                    best_iteration_count = iteration_count;
                }
            } else {
                grid_qualities[i] = std::numeric_limits<float>::max();
            }
        }

        uint64_t idx_1 = min_index(grid_qualities, grid_size + 1);
        grid_qualities[idx_1] = std::numeric_limits<float>::max();
        uint64_t idx_2 = min_index(grid_qualities, grid_size + 1);

        uint64_t idx_lo = (idx_1 <= idx_2) ? idx_1 : idx_2;
        uint64_t idx_hi = (idx_1 >  idx_2) ? idx_1 : idx_2;

        //
        // Set hi before you set lo, so we do not have to do explicit copy
        //
        hi = lo + step * idx_hi;
        lo = lo + step * idx_lo;

        std::cout << "Readjustment: " << lo << " - " << hi << std::endl;
    }

    std::cout << "" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Finished Search for Best Quality | mu = " << mu << " | Quality: " << best_quality;
    std::cout << " | Iterations: " << best_iteration_count << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "" << std::endl;

    n_iter = best_iteration_count;

    return best_quality;
}

template <class QMatrix, class QVector>
uint64_t IHT_find_best_n_iterations (
        CloverVector32 &x_real,
        CloverMatrix32 &Phi_32,
        CloverVector32 &y_32,
        float lo_initial, float hi, float precision,
        float quality_target,
        float &result_mu, float K
) {

    uint64_t grid_size = 10;
    float mu, lo = lo_initial;
    bool is_convergent = false;

    uint64_t iteration_count = 0;
    uint64_t iteration_limit = 50;
    uint64_t best_iteration_count = iteration_limit;

    float curr_quality = -std::numeric_limits<float>::max();

    //
    // Get the dimensions
    //
    const uint64_t m = Phi_32.getRows();
    const uint64_t n = Phi_32.getCols();
    //
    // Setup the matrices and vectors
    //
    QMatrix Phi(m, n);
    QMatrix PhiT(n, m);
    QVector x(n);
    QVector y(m);

    CloverVector32 x_restored(n);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);
    //
    // Feed the data
    //
    Phi.quantize(Phi_32);
    y.quantize(y_32);
    Phi.transpose(PhiT);
    const double x_real_norm2 = norm2(x_real);
    //
    // Let's begin
    //
    std::cout << "Optimizing for best iterations and looking for mu using ";
    std::cout << std::setw(2) << Phi.getBitsLength() << "-bit matrix and ";
    std::cout << std::setw(2) << x.getBitsLength() << "-bit vector ";
    std::cout << std::endl;

    //
    // Start a binary search to find the largest mu that this problem would converge
    //
    while (lo + precision <= hi)
    {
        mu = (lo + hi) / 2;

        is_convergent = profile_IHT (
                Phi, PhiT, x, y, mu, K,
                x_real, x_real_norm2, iteration_limit, quality_target,
                curr_quality, iteration_count,
                x_restored, t1, t2, t3
        );

        if (is_convergent) {
            lo = mu;
            if (curr_quality < quality_target && iteration_count < best_iteration_count) {
                result_mu = mu;
                best_iteration_count = iteration_count;
            }
        } else {
            hi = mu;
        }

        std::cout << "Convergence for mu = " << std::setw(17) << std::fixed << std::setprecision(10) << mu << ": " << (is_convergent ? "OK  " : "Fail");
        std::cout << " | ";
        std::cout << "Quality: " << curr_quality;
        std::cout << " | ";
        std::cout << "Target: " << quality_target;
        std::cout << " | ";
        std::cout << "Iterations: " << iteration_count;
        std::cout << std::endl;
    }


    //
    // Now restart and start looking at the grid
    //
    lo = lo_initial;
    uint64_t grid_iterations[grid_size + 10];

    while (lo + precision <= hi) {

        float step = (hi - lo) / grid_size;

        for (uint64_t i = 0; i <= grid_size; i += 1) {
            mu = lo + step * i;
            is_convergent = profile_IHT (
                    Phi, PhiT, x, y, mu, K,
                    x_real, x_real_norm2, iteration_limit, quality_target,
                    curr_quality, iteration_count,
                    x_restored, t1, t2, t3
            );

            std::cout << "Convergence for mu = " << std::setw(17) << std::fixed << std::setprecision(10) << mu << ": " << (is_convergent ? "OK  " : "Fail");
            std::cout << " | ";
            std::cout << "Quality: " << curr_quality;
            std::cout << " | ";
            std::cout << "Target: " << quality_target;
            std::cout << " | ";
            std::cout << "Iterations: " << iteration_count;
            std::cout << std::endl;


            if (is_convergent && curr_quality < quality_target) {
                grid_iterations[i] = iteration_count;
                if (iteration_count < best_iteration_count) {
                    best_iteration_count = iteration_count;
                }
            } else {
                grid_iterations[i] = iteration_limit;
            }
        }
        //
        // Look at the range of best iteration and do another grid search into that region
        //
        uint64_t idx_1 = min_index(grid_iterations, grid_size + 1);
        grid_iterations[idx_1] = iteration_limit;
        uint64_t idx_2 = min_index(grid_iterations, grid_size + 1);
        uint64_t idx_lo = (idx_1 <= idx_2) ? idx_1 : idx_2;
        uint64_t idx_hi = (idx_1 >  idx_2) ? idx_1 : idx_2;
        if (idx_lo == 0 && idx_hi == grid_size) {
            idx_lo = 1; // Sanity check
        }

        hi = lo + step * idx_hi; // Set hi before you set lo, so we do not have to do explicit copy
        lo = lo + step * idx_lo;

        std::cout << "Readjustment: " << lo << " - " << hi << std::endl;
    }

    std::cout << "" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Finished Search for Optimal Iterations | mu = " << mu << " | Target: " << quality_target;
    std::cout << " | Iterations: " << best_iteration_count << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "" << std::endl;

    std::cout << std::endl;
    return best_iteration_count;
}

experiment_setup_t * setup_IHT_experiment_mixed();
experiment_setup_t * setup_GD_experiment_mixed();
experiment_setup_t * setup_IHT_experiment_pure();
experiment_setup_t * setup_GD_experiment_pure();

void initialize_random_IHT_values(
        CloverVector32 &x_32, CloverMatrix32 &Phi_32, CloverVector32 &y_32, uint64_t &K
);

void initialize_random_GD_values(
        CloverVector32 &x_32, CloverMatrix32 &Phi_32, CloverVector32 &y_32
);

#endif /* CLOVER_IHT_GD_UTIL_H */
