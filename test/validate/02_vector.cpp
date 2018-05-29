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
#include <CloverVector4.h>
#include <CloverVector16.h>
#include <CloverVector8.h>

using namespace std;

template<class QVector, class QSupport>
void validate_vector_apply_support (int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Apply Support " << ": ";


    for (uint64_t size = 128; size < 1024; size += 1) {

        CloverVector32 samples_vector_32(size);
        CloverVector32 testing_vector_32(size);
        samples_vector_32.setRandomInteger(-2, 2);
        testing_vector_32.setRandomInteger(1, 5);

        //
        // Create the vectors for testing
        //
        QVector samples_vector(size);
        QVector testing_vector(size);
        samples_vector.quantize(samples_vector_32);
        testing_vector.quantize(testing_vector_32);
        QVector testing_vector_clone(testing_vector);

        QSupport support(size);

        samples_vector.support(support);
        testing_vector.apply_support(support);
        testing_vector_clone.apply_support_scalar(support);

        for (uint64_t i = 0; i < size; i += 1) {
            float v1 = testing_vector.get(i);
            float v2 = testing_vector_clone.get(i);
            if (v1 != v2) {
                cout << "Failed" << endl << endl;
                cout << "Operation fails at position: " << i << endl;
                cout << endl;
                std::string supps = compare(testing_vector.toString(), testing_vector_clone.toString());
                cout << compare(supps, support.toString()) << endl;
                exit(1);
            }
        }
    }
    cout << "Good" << endl;
}

template<class QVector, class QSupport>
void validate_vector_support (int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Support " << ": ";


    for (uint64_t size = 128; size < 2048; size += 1) {
        CloverVector32 v_32(size);
        QVector  v_01(size);
        QSupport supp_scalar(size);
        QSupport supp_vector(size);

        v_32.setRandomFloats(-10, 10);
        v_01.quantize(v_32);
        v_01.support(supp_vector);
        v_01.support_scalar(supp_scalar);

        for (uint64_t i = 0; i < size; i += 1) {
            bool v1 = supp_vector.get(i);
            bool v2 = supp_scalar.get(i);
            if (v1 != v2) {
                cout << "Failed" << endl << endl;
                cout << "Operation fails at position: " << i << endl;
                cout << endl;
                std::string supps = compare(supp_scalar.toString(), supp_vector.toString());
                cout << compare(v_01.toString(), supps) << endl;
                exit(1);
            }
        }
    }
    cout << "Good" << endl;
}


template<class QVector>
void validate_vector_quantization (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Quantization " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 1024; size += 1) {
            CloverVector32 v_32(size);
            QVector v_01(size);
            QVector v_02(size);

            v_32.setRandomInteger(10);
            v_01.quantize(v_32);
            v_02.quantize_scalar(v_32);

            for (uint64_t i = 0; i < size; i += 1) {
                float v1 = v_01.get(i);
                float v2 = v_02.get(i);
                if (v1 != v2) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << compare(v_01.toString(), v_02.toString()) << endl;
                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_quantization_parallel (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Parallel Quantization " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 1024; size += 1) {
            CloverVector32 v_32(size);
            QVector v_01(size);
            QVector v_02(size);

            v_32.setRandomInteger(10);
            v_01.quantize_parallel(v_32);
            v_02.quantize_scalar(v_32);

            for (uint64_t i = 0; i < size; i += 1) {
                float v1 = v_01.get(i);
                float v2 = v_02.get(i);
                if (v1 != v2) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << compare(v_01.toString(), v_02.toString()) << endl;
                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_consistency(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Consistency " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 2048; size += 1) {

            CloverVector32 v_32(size);
            CloverVector32 u_32(size);
            QVector v_01(size);

            v_32.setRandomInteger(7);

            v_01.quantize(v_32);
            v_01.restore(u_32);

            for (uint64_t i = 0; i < size; i += 1) {
                float v1 = v_32.get(i);
                float v2 = u_32.get(i);
                if (fabsf(v1 - v2) > 1) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_32[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "u_32[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << endl;
                    cout << "Quantization result: " << endl;
                    cout << compare(v_32.toString(), v_01.toString()) << endl;
                    cout << endl;
                    cout << "Restoration result: " << endl;
                    cout << compare(v_32.toString(), u_32.toString()) << endl;
                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_restore (int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Restore " << ": ";

    for (uint64_t size = 128; size < 1024; size += 1) {

        CloverVector32 v_32(size);
        QVector         q_01(size);

        CloverVector32 v_01(size);
        CloverVector32 v_02(size);

        v_32.setRandomInteger(10);
        q_01.quantize(v_32);
        q_01.restore(v_01);
        q_01.restore_scalar(v_02);

        for (uint64_t i = 0; i < size; i += 1) {
            float v1 = v_01.get(i);
            float v2 = v_02.get(i);
            if (v1 != v2) {
                cout << "Failed" << endl << endl;
                cout << "Operation fails at position: " << i << endl;
                cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                cout << endl;
                cout << compare(v_01.toString(), v_02.toString()) << endl;
                exit(1);
            }
        }
    }
    cout << "Good" << endl;
}

template<class QVector>
void validate_vector_dot (int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Dot Product " << ": ";

    for (uint64_t size = 128; size < 2048; size += 1) {

        CloverVector32 v_32(size);
        CloverVector32 u_32(size);

        QVector         q_01(size);
        QVector         q_02(size);

        v_32.setRandomInteger(7);
        u_32.setRandomInteger(7);

        q_01.quantize(v_32);
        q_02.quantize(u_32);

        float v1 = q_01.dot(q_02);
        float v2 = q_01.dot_scalar(q_02);

        //
        // For the dot product computation, we are simply not aiming at binary compatibility
        // Our order of computations is vastly different anyways.
        //
        if (fabs(v1 - v2) > 0.02) {
            cout << "Failed" << endl << endl;
            cout << "\t" << "Dot Product       : " << CloverBase::float2hex(v1) << " " << v1 << endl;
            cout << "\t" << "Dot Product Scalar: " << CloverBase::float2hex(v2) << " " << v2 << endl;
            cout << endl;
            cout << compare(q_01.toString(), q_02.toString()) << endl;
            exit(1);
        }

    }
    cout << "Good" << endl;
}


template<class QVector>
void validate_vector_dot_parallel (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Parallel Dot Product " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 2048; size += 1) {

            CloverVector32 v_32(size);
            CloverVector32 u_32(size);

            QVector q_01(size);
            QVector q_02(size);

            v_32.setRandomInteger(7);
            u_32.setRandomInteger(7);

            q_01.quantize(v_32);
            q_02.quantize(u_32);

            float v1 = q_01.dot_parallel(q_02);
            float v2 = q_01.dot_scalar(q_02);

            //
            // For the dot product computation, we are simply not aiming at binary compatibility
            // Our order of computations is vastly different anyways.
            //
            if (fabs(v1 - v2) > 0.02) {
                cout << "Failed" << endl << endl;
                cout << "\t" << "Dot Product       : " << CloverBase::float2hex(v1) << " " << v1 << endl;
                cout << "\t" << "Dot Product Scalar: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                cout << endl;
                cout << compare(q_01.toString(), q_02.toString()) << endl;
                exit(1);
            }

        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_scaleAndAdd (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Scale And Add " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 2048; size += 1) {

            CloverVector32 v_32(size);
            CloverVector32 u_32(size);

            QVector q_01(size);
            QVector q_02(size);

            v_32.setRandomInteger(40);
            u_32.setRandomInteger(10);

            q_01.quantize(v_32);
            q_02.quantize(u_32);

            QVector q_01_copy = q_01;

            q_01.scaleAndAdd(q_02, 0.5f);
            q_01_copy.scaleAndAdd_scalar(q_02, 0.5f);


            for (uint64_t i = 0; i < size; i += 1) {
                float v1 = q_01.get(i);
                float v2 = q_01_copy.get(i);
                if (v1 != v2) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << compare(q_01.toString(), q_01_copy.toString()) << endl;
                    cout << endl;
                    cout << endl;
                    cout << "Initial vectors are defined as: " << endl;
                    cout << compare(v_32.toString(), u_32.toString()) << endl;
                    cout << "Result vectors is defined as: " << endl;
                    v_32.scaleAndAdd(u_32, 0.5f);
                    cout << v_32.toString() << endl;

                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_scaleAndAdd_parallel (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Scale And Add Parallel " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 2048; size += 1) {

            CloverVector32 v_32(size);
            CloverVector32 u_32(size);

            QVector q_01(size);
            QVector q_02(size);

            v_32.setRandomInteger(40);
            u_32.setRandomInteger(10);

            q_01.quantize(v_32);
            q_02.quantize(u_32);

            QVector q_01_copy = q_01;

            q_01.scaleAndAdd_parallel(q_02, 0.5f);
            q_01_copy.scaleAndAdd_scalar(q_02, 0.5f);


            for (uint64_t i = 0; i < size; i += 1) {
                float v1 = q_01.get(i);
                float v2 = q_01_copy.get(i);
                if (v1 != v2) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << compare(q_01.toString(), q_01_copy.toString()) << endl;
                    cout << endl;
                    cout << endl;
                    cout << "Initial vectors are defined as: " << endl;
                    cout << compare(v_32.toString(), u_32.toString()) << endl;
                    cout << "Result vectors is defined as: " << endl;
                    v_32.scaleAndAdd(u_32, 0.5f);
                    cout << v_32.toString() << endl;

                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QVector>
void validate_vector_threshold (int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Threshold " << ": ";


    for (uint64_t size = 128; size < 2048; size += 1) {
        int k = 64;

        CloverVector32 v_32(size);
        QVector q_01(size);
        QVector q_02(size);
        CloverVector32 r_32(size);

        v_32.setRandomInteger(40);

        q_01.quantize(v_32);
        QVector q_01_copy = q_01;
        q_01_copy.restore(v_32);

        q_01.threshold(k);
        q_01.restore(r_32);
        std::sort(v_32.getData(), v_32.getData() + size, [](float a, float b) { return (std::abs(a) > std::abs(b)); });
        std::sort(r_32.getData(), r_32.getData() + size, [](float a, float b) { return (std::abs(a) > std::abs(b)); });

        for (uint64_t i = 0; i < k; i += 1) {
            float v1 = v_32.get(i);
            float v2 = r_32.get(i);

            if (std::abs(std::abs(v1) - std::abs(v2)) / std::max(v1,v2) > 0.1) {
                cout << "Failed" << endl << endl;
                cout << "Operation fails at position: " << i << endl;
                cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                cout << endl;
                cout << compare(v_32.toString(), r_32.toString()) << endl;
                cout << endl;
                cout << endl;
                cout << "Initial vector is defined as: " << endl;
                cout << q_01_copy.toString() << endl;
                cout << "Result vector is defined as: " << endl;
                cout << q_01.toString() << endl;

                exit(1);
            }
        }
    }
    cout << "Good" << endl;

}


template<class QVector>
void validate_vector_threshold_parallel (int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Parallel Threshold " << ": ";

    if (!skip_validation) {
        for (uint64_t size = 128; size < 2048; size += 1) {
            int k = 64;

            CloverVector32 v_32(size);
            QVector q_01(size);
            QVector q_02(size);
            CloverVector32 r_32(size);

            v_32.setRandomInteger(40);

            q_01.quantize(v_32);
            QVector q_01_copy = q_01;
            q_01_copy.restore(v_32);

            q_01.threshold_parallel(k);
            q_01.restore(r_32);
            std::sort(v_32.getData(), v_32.getData() + size,
                      [](float a, float b) { return (std::abs(a) > std::abs(b)); });
            std::sort(r_32.getData(), r_32.getData() + size,
                      [](float a, float b) { return (std::abs(a) > std::abs(b)); });

            for (uint64_t i = 0; i < k; i += 1) {
                float v1 = v_32.get(i);
                float v2 = r_32.get(i);

                if (std::abs(std::abs(v1) - std::abs(v2)) / std::max(v1, v2) > 0.1) {
                    cout << "Failed" << endl << endl;
                    cout << "Operation fails at position: " << i << endl;
                    cout << "\t" << "v_01[" << i << "]: " << CloverBase::float2hex(v1) << " " << v1 << endl;
                    cout << "\t" << "v_02[" << i << "]: " << CloverBase::float2hex(v2) << " " << v2 << endl;
                    cout << endl;
                    cout << compare(v_32.toString(), r_32.toString()) << endl;
                    cout << endl;
                    cout << endl;
                    cout << "Initial vector is defined as: " << endl;
                    cout << q_01_copy.toString() << endl;
                    cout << "Result vector is defined as: " << endl;
                    cout << q_01.toString() << endl;

                    exit(1);
                }
            }
        }
        cout << "Good" << endl;
    }  else {
        cout << "Skipped" << endl;
    }
}


void validate_vector_ops ()
{
#if defined(_OPENMP)
    bool openMPEnabled = true;
#else
    bool openMPEnabled = false;
#endif

#ifdef CLOVER_STOCHASTIC_ROUNDING_DISABLED
    bool stochastic_rounding_disabled = true;
#else
    bool stochastic_rounding_disabled = false;
#endif

    cout << "======================================================================" << endl;
    cout << "= Validating Vector operations: " << endl;
    cout << "----------------------------------------------------------------------" << endl;
    //
    // Validate the quantization process, for this we need stochastic rounding
    // to be in fact disabled.
    //
    validate_vector_quantization<CloverVector4> ( 4, !stochastic_rounding_disabled);
    validate_vector_quantization<CloverVector8> ( 8, !stochastic_rounding_disabled);
    validate_vector_quantization<CloverVector16>(16, false);
    //
    // Validate quantization in parallel
    //
    validate_vector_quantization_parallel<CloverVector4> ( 4, !openMPEnabled || !stochastic_rounding_disabled);
    validate_vector_quantization_parallel<CloverVector8> ( 8, !openMPEnabled || !stochastic_rounding_disabled);
    validate_vector_quantization_parallel<CloverVector16>(16, !openMPEnabled);
    validate_vector_quantization_parallel<CloverVector32>(32, !openMPEnabled);
    //
    // Validate the restore routine product
    //
    validate_vector_restore<CloverVector4> ( 4);
    validate_vector_restore<CloverVector8> ( 8);
    validate_vector_restore<CloverVector16>(16);
    //
    // Validate consistency
    //
    validate_vector_consistency<CloverVector4> ( 4, !stochastic_rounding_disabled);
    validate_vector_consistency<CloverVector8> ( 8, !stochastic_rounding_disabled);
    validate_vector_consistency<CloverVector16>(16, false);
    //
    // Validate the dot product
    //
    validate_vector_dot<CloverVector4> ( 4);
    validate_vector_dot<CloverVector8> ( 8);
    validate_vector_dot<CloverVector16>(16);
    //
    // Validate dot product parallel
    //
    validate_vector_dot_parallel<CloverVector4> ( 4, !openMPEnabled);
    validate_vector_dot_parallel<CloverVector8> ( 8, !openMPEnabled);
    validate_vector_dot_parallel<CloverVector16>(16, !openMPEnabled);
    validate_vector_dot_parallel<CloverVector32>(32, !openMPEnabled);
    //
    // Validate scale and add (sequential)
    //
    validate_vector_scaleAndAdd<CloverVector4> ( 4, !stochastic_rounding_disabled);
    validate_vector_scaleAndAdd<CloverVector8> ( 8, !stochastic_rounding_disabled);
    validate_vector_scaleAndAdd<CloverVector16>(16, false);
    //
    // Validate scale and add (parallel)
    //
    validate_vector_scaleAndAdd_parallel<CloverVector4 >( 4, !openMPEnabled || !stochastic_rounding_disabled);
    validate_vector_scaleAndAdd_parallel<CloverVector8 >( 8, !openMPEnabled || !stochastic_rounding_disabled);
    validate_vector_scaleAndAdd_parallel<CloverVector16>(16, !openMPEnabled);
    //
    // Validate thresholding
    //
    validate_vector_threshold<CloverVector4 >( 4);
    validate_vector_threshold<CloverVector8 >( 8);
    validate_vector_threshold<CloverVector16>(16);
    //
    // Validate thresholding (parallel)
    //
    validate_vector_threshold_parallel<CloverVector4 >( 4, !openMPEnabled);
    validate_vector_threshold_parallel<CloverVector8 >( 8, !openMPEnabled);
    validate_vector_threshold_parallel<CloverVector16>(16, !openMPEnabled);
    validate_vector_threshold_parallel<CloverVector32>(32, !openMPEnabled);

    cout << "======================================================================" << endl;
    cout << endl;
}
