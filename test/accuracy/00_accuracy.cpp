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

#include <CloverVector4.h>
#include <CloverMatrix4.h>
#include <CloverVector8.h>
#include <CloverMatrix8.h>
#include <CloverMatrix16.h>
#include "01_math.h"
#include "02_iht_accuracy.h"
#include "03_gd_accuracy.h"

using namespace std;

void test_iht(bool should_search_for_mu)
{
    uint64_t m = 512 , n = 1024;
    uint64_t numberOfEpochs = 200;
    uint64_t K = n / 16;

    cout << "======================================================================" << endl;
    cout << "= Performing IHT accuracy testing ..." << endl;
    cout << "----------------------------------------------------------------------" << endl;

    CloverMatrix32 Phi(m, n);
    CloverVector32 x(n);
    CloverVector32 y(m);
    initialize_random_IHT_values(x, Phi, y, K);

    //
    // Set the control parameters
    //
    float lo = 0.000001;
    float hi = 0.5;
    float precision = 0.000001;

    float mu32, mu16, mu08, mu04, mu48;
    uint64_t n_iter;

    if (should_search_for_mu) {
        //
        // Let's try to reached the best quality we can
        //
        IHT_best_possible_quality<CloverMatrix4, CloverVector8>(x, Phi, y, K, lo, hi, precision, n_iter, mu48);
        IHT_best_possible_quality<CloverMatrix4, CloverVector4>(x, Phi, y, K, lo, hi, precision, n_iter, mu04);
        IHT_best_possible_quality<CloverMatrix8, CloverVector8>(x, Phi, y, K, lo, hi, precision, n_iter, mu08);
        IHT_best_possible_quality<CloverMatrix16, CloverVector16>(x, Phi, y, K, lo, hi, precision, n_iter, mu16);
        IHT_best_possible_quality<CloverMatrix32, CloverVector32>(x, Phi, y, K, lo, hi, precision, n_iter, mu32);
    } else {
        //
        // Hard-code the mu-s (numbers obtained from a previous iteration)
        //
        mu48 = 0.0051299492f;
        mu04 = 0.0042842f;
        mu08 = 0.0042007f;
        mu16 = 0.0048838f;
        mu32 = 0.0048838;
    }

    //
    // Call the four versions
    //
    test_iht_accuracy<CloverMatrix4 , CloverVector8 > (x, Phi, y, numberOfEpochs, K, mu48);
    test_iht_accuracy<CloverMatrix4 , CloverVector4 > (x, Phi, y, numberOfEpochs, K, mu04);
    test_iht_accuracy<CloverMatrix8 , CloverVector8 > (x, Phi, y, numberOfEpochs, K, mu08);
    test_iht_accuracy<CloverMatrix16, CloverVector16> (x, Phi, y, numberOfEpochs, K, mu16);
    test_iht_accuracy<CloverMatrix32, CloverVector32> (x, Phi, y, numberOfEpochs, K, mu32);

    cout << "======================================================================" << endl << endl;
}

void test_gd(bool should_search_for_mu)
{
    uint64_t m = 384 , n = 256;
    uint64_t numberOfEpochs = 500;
    float mu = 0.4000000358f;

    cout << "======================================================================" << endl;
    cout << "= Performing GD accuracy testing ..." << endl;
    cout << "----------------------------------------------------------------------" << endl;

    CloverMatrix32 Phi(m, n);
    CloverVector32 x(n);
    CloverVector32 y(m);
    initialize_random_GD_values(x, Phi, y);


    //
    // Call the four versions
    //
    test_gd_accuracy<CloverMatrix4 , CloverVector8 > (x, Phi, y, numberOfEpochs, mu);
    test_gd_accuracy<CloverMatrix4 , CloverVector4 > (x, Phi, y, numberOfEpochs, mu);
    test_gd_accuracy<CloverMatrix8 , CloverVector8 > (x, Phi, y, numberOfEpochs, mu);
    test_gd_accuracy<CloverMatrix16, CloverVector16> (x, Phi, y, numberOfEpochs, mu);
    test_gd_accuracy<CloverMatrix32, CloverVector32> (x, Phi, y, numberOfEpochs, mu);

    cout << "======================================================================" << endl << endl;
}

void test_accuracy ()
{
    //
    // For reproducibility
    //
    init_deterministic_keys();
    test_iht(false);
    // test_gd(false);
}
