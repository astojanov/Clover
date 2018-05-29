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

#ifndef CLOVER_GD_ACCURACY_H
#define CLOVER_GD_ACCURACY_H

#include "../performance/01_measure.h"

template <class QMatrix, class QVector>
inline void Q_GD_norm2(
        const CloverVector32 &x_real,  // The real vector that we try to reconstruct
        QVector &x,                 // The vector that we like to reconstruct
        QMatrix &Phi,               // Measurements matrix \Phi [MxN]
        QVector &y,                 // Real valued vector of observations [M]
        uint64_t numberOfIterations,   // Number of iterations
        float mu
) {
    //
    // Get the sizes
    //
    uint64_t m = Phi.getRows();
    uint64_t n = Phi.getCols();
    //
    // Restored vector
    //
    CloverVector32 x_iteration(n);
    //
    // Define the temporary vectors
    //
    QMatrix PhiT(n, m);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);

    //
    // Make sure that the initial time signal is set to zero.
    //
    x.clear();

    //
    // Perform the matrix transposition
    //
    Phi.transpose(PhiT);

    //
    // The norm of the real vector
    //
    double x_real_norm2 = norm2(x_real);

    for(int iteration = 0; iteration < numberOfIterations; iteration++)
    {
        Phi.mvm_parallel(x, t1);                // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);  // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);              // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, mu);         // x = x + \mu * Phi' * (y - Phi * x)
        //
        // Now calculate the norm
        //
        x.restore(x_iteration);
        x_iteration.scaleAndAdd(x_real, -1);
        double norm_diff = norm2(x_iteration);

        std::cout << std::setw(10) << iteration + 1 << " | ";
        std::cout << std::fixed << std::setprecision(7) << std::setw(20) << norm_diff / x_real_norm2 << std::endl;
    }
}

template <class QMatrix, class QVector>
void test_gd_accuracy (
        const CloverVector32 &x_32,
        const CloverMatrix32 &Phi_32,
        const CloverVector32 &y_32,
        const uint64_t numberOfIterations,
        float mu
) {
    uint64_t m = Phi_32.getRows();
    uint64_t n = Phi_32.getCols();

    //
    // Lets Initialize those things.
    //
    QMatrix Phi(m, n);
    QVector x(n);
    QVector y(m);
    Phi.quantize(Phi_32);
    y.quantize(y_32);

    std::cout << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "Testing            : " << y.getBitsLength() << "-bit Matrix of size: " << m << " x " << n << std::endl;
    std::cout << "Running            : " << numberOfIterations << " iterations" << std::endl;;
    std::cout << "Learning Rate (mu) : " << mu << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "Iteration      |  Reconstruction Rate" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    //
    // Perform the benchmark
    //
    Q_GD_norm2(x_32, x, Phi, y, numberOfIterations, mu);

    std::cout << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "The original vector (left) compared to the reconstructed vector (right)" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << compare(x_32.toString(), x.toString()) << std::endl;
    std::cout << std::endl;
}


#endif /* CLOVER_GD_ACCURACY_H */
