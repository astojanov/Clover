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
#include "01_endianess.h"
#include "02_vector.h"
#include "03_matrix.h"
#include "../random/00_random.h"

void print_disable_stohastic_rounding_warning()
{
#ifndef CLOVER_STOCHASTIC_ROUNDING_DISABLED
    std::cout << "======================================================================" << std::endl;
    std::cout << "= Validaton Warning:" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
    std::cout << "This validation is only possible if this project is compiled " << std::endl;
    std::cout << "such that STOCHASTIC_ROUNDING_DISABLED flag is enabled," << std::endl;
    std::cout << "otherwise, most of the validation steps will be skipped!" << std::endl;
    std::cout << std::endl;
    std::cout << "To enable it, use the following: " << std::endl;
    std::cout << std::endl;
    std::cout << "\t" << "mkdir build" << std::endl;
    std::cout << "\t" << "cd build" << std::endl;
    std::cout << "\t" << "cmake -DSTOCHASTIC_ROUNDING_DISABLED=1 .." << std::endl;
    std::cout << "\t" << "cd .." << std::endl;
    std::cout << "\t" << "cmake --build build --config Release" << std::endl;
    std::cout << std::endl;
#endif
}


void validate (int argc, const char* argv[])
{
    init_deterministic_keys();
    test_endianness();
    print_disable_stohastic_rounding_warning();
    validate_vector_ops();
    validate_matrix_ops();
}
