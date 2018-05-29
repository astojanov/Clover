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

#include <simdxorshift128plus.h>
#include <iostream>
#include "00_random.h"

using namespace std;

__m256i random_FVector_key1, random_FVector_key2;

void init_deterministic_keys()
{
    //
    // Initialize the XOR-Shift
    //
    cout << "======================================================================" << endl;
    cout << "= Initialization of the single XOR shift used for random vectors: Done" << endl;
    cout << "======================================================================" << endl;
    cout << endl;
    avx_xorshift128plus_init(445560390295639063UL, 2935984234003016713UL, random_FVector_key1, random_FVector_key2);
}
