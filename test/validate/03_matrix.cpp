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
#include <cassert>

#include <CloverMatrix32.h>
#include <CloverMatrix4.h>
#include <CloverMatrix16.h>
#include <CloverMatrix8.h>

#include "../random/00_random.h"
#include "00_validate.h"

using namespace std;

template<class QMatrix>
void validate_matrix_quantization(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Quantization " << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                m32.setRandomInteger(10);

                QMatrix m_01(m, n);
                QMatrix m_02(m, n);

                m_01.quantize(m32);
                m_02.quantize_scalar(m32);


                for (int ki = 0; ki < m; ki += 1) {
                    for (int kj = 0; kj < n; kj += 1) {

                        float x1 = m_01.get(ki, kj);
                        float x2 = m_02.get(ki, kj);

                        if (x1 != x2) {
                            cout << "Fails at position: (" << ki << ", " << kj << ")" << endl;
                            cout << endl << endl;

                            cout << "The initial version: " << endl;
                            cout << m32.toString() << endl;
                            cout << endl << endl;

                            cout << "Vector version: " << endl;
                            cout << m_01.toString() << endl;
                            cout << endl << endl;

                            cout << "Scalar version: " << endl;
                            cout << m_02.toString() << endl;
                            cout << endl << endl;

                            cout << "Exiting ... " << endl;
                            exit(1);
                        }
                    }
                }

            }
        }

        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }

}


template<class QMatrix>
void validate_matrix_consistency(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Consistency " << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                m32.setRandomInteger(7);

                QMatrix m_01(m, n);
                m_01.quantize(m32);

                for (int ki = 0; ki < m; ki += 1) {
                    for (int kj = 0; kj < n; kj += 1) {

                        float x1 = m32.get(ki, kj);
                        float x2 = m_01.get(ki, kj);

                        if (fabs(x1 - x2) > 1) {
                            cout << "Fails at position: (" << ki << ", " << kj << ")" << endl;
                            cout << endl << endl;

                            cout << "The initial version: " << endl;
                            cout << m32.toString() << endl;
                            cout << endl << endl;

                            cout << "Quantized: " << endl;
                            cout << m_01.toString() << endl;
                            cout << endl << endl;

                            cout << "Exiting ... " << endl;
                            exit(1);
                        }
                    }
                }

            }
        }

        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }

}



template<class QMatrix>
void validate_matrix_transpose(int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Transpose " << ": ";

    for (int i = 1; i <= 10; i += 1) {
        for (int j = 1; j <= 10; j += 1) {

            int m = i * 128;
            int n = j * 128;

            CloverMatrix32 m32(m, n);
            m32.setRandomInteger(10);

            QMatrix qm(m, n);
            QMatrix qt(n, m);
            qm.quantize(m32);
            qm.transpose(qt);

            for (int ki = 0; ki < m; ki += 1) {
                for (int kj = 0; kj < n; kj += 1) {

                    float x1 = qm.get(ki, kj);
                    float x2 = qt.get(kj, ki);

                    if (x1 != x2) {
                        cout << "Fails at position: (" << ki << ", " << kj << ")" << endl;

                        cout << endl << endl;
                        cout << qm.toString() << endl;
                        cout << endl << endl;
                        cout << qt.toString() << endl;
                        cout << endl << endl;

                        cout << "Exiting ... " << endl;
                        exit(1);
                    }
                }
            }

        }
    }

    cout << "Good" << endl;
}

template<class QMatrix>
void validate_matrix_transpose_parallel(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Transpose Parallel" << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                m32.setRandomInteger(10);

                QMatrix qm(m, n);
                QMatrix qt(n, m);
                qm.quantize(m32);
                qm.transpose_parallel(qt);

                for (int ki = 0; ki < m; ki += 1) {
                    for (int kj = 0; kj < n; kj += 1) {

                        float x1 = qm.get(ki, kj);
                        float x2 = qt.get(kj, ki);

                        if (x1 != x2) {
                            cout << "Fails at position: (" << ki << ", " << kj << ")" << endl;

                            cout << endl << endl;
                            cout << qm.toString() << endl;
                            cout << endl << endl;
                            cout << qt.toString() << endl;
                            cout << endl << endl;

                            cout << "Exiting ... " << endl;
                            exit(1);
                        }
                    }
                }

            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }
}

template<class QMatrix, class QVector>
void validate_matrix_MVM(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "MVM Vectorized" << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                CloverVector32 v32(n);
                CloverVector32 r32(m);
                m32.setRandomInteger(10);
                v32.setRandomInteger(10);

                QMatrix m_01(m, n);
                QVector v_01(n);

                QVector result_scalar(m);
                QVector result_vector(m);

                m_01.quantize(m32);
                v_01.quantize(v32);

                m_01.mvm(v_01, result_vector);
                m_01.mvm_scalar(v_01, result_scalar);


                for (int k = 0; k < m; k += 1) {

                    float x1 = result_vector.get(k);
                    float x2 = result_scalar.get(k);

                    if (x1 != x2) {
                        cout << "Fails at position: " << k << endl;
                        cout << endl << endl;

                        cout << "Resulting vectors: " << endl;
                        cout << compare(result_vector.toString(), result_scalar.toString()) << endl;
                        cout << endl << endl;

                        cout << "Quantized matrix is: " << endl;
                        cout << m_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Quantized vector is: " << endl;
                        cout << v_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Matrix is: " << endl;
                        cout << m32.toString() << endl;
                        cout << endl << endl;

                        cout << "Vector is: " << endl;
                        cout << v32.toString() << endl;
                        cout << endl << endl;

                        m32.mvm(v32, r32);
                        cout << "32-bit result is: " << endl;
                        cout << r32.toString() << endl;
                        cout << endl << endl;

                        cout << "Exiting ... " << endl;
                        exit(1);
                    }

                }

            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }

}

void validate_matrix_MVM_mixed_mat4_vec8(int skip_validation)
{
    cout << right << std::setw(2) << 4 << "-bit | " << left << std::setw(30) << "MVM_8 Vectorized" << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                CloverVector32 v32(n);
                CloverVector32 r32(m);
                m32.setRandomInteger(10);
                v32.setRandomInteger(10);

                CloverMatrix4 m_01(m, n);
                CloverVector8 v_01(n);

                CloverVector8 result_scalar(m);
                CloverVector8 result_vector(m);

                m_01.quantize(m32);
                v_01.quantize(v32);

                m_01.mvm(v_01, result_vector);
                m_01.mvm_scalar(v_01, result_scalar);


                for (int k = 0; k < m; k += 1) {

                    float x1 = result_vector.get(k);
                    float x2 = result_scalar.get(k);

                    bool equal = false;
                    if (x2 == 0) {
                        if (x1 == 0) {
                            equal = true;
                        } else {
                            equal = abs(result_vector.getBits(k)) == 1;
                        }
                    } else {
                        equal = fabs(x1 - x2) / fabs(x2) < 0.016;
                    }

                    if (!equal) {
                        cout << "Fails at position: " << k << endl;
                        cout << endl << endl;

                        cout << "Resulting vectors: " << endl;
                        cout << compare(result_vector.toString(), result_scalar.toString()) << endl;
                        cout << endl << endl;

                        cout << "Quantized matrix is: " << endl;
                        cout << m_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Quantized vector is: " << endl;
                        cout << v_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Matrix is: " << endl;
                        cout << m32.toString() << endl;
                        cout << endl << endl;

                        cout << "Vector is: " << endl;
                        cout << v32.toString() << endl;
                        cout << endl << endl;

                        m32.mvm(v32, r32);
                        cout << "32-bit result is: " << endl;
                        cout << r32.toString() << endl;
                        cout << endl << endl;

                        cout << "Exiting ... " << endl;
                        exit(1);
                    }

                }

            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }

}


template<class QMatrix>
void validate_matrix_vector_product_32(int bitcount)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "Mixed MVM_32 Vectorized" << ": ";


    for (int i = 1; i <= 10; i += 1) {
        for (int j = 1; j <= 10; j += 1) {

            int m = i * 128;
            int n = j * 128;

            CloverMatrix32 matrix_32(m, n);
            CloverVector32 vector_32(n);

            matrix_32.setRandomInteger(10);
            vector_32.setRandomInteger(10);

            QMatrix matrix_lo(m, n);

            CloverVector32 result_scalar(m);
            CloverVector32 result_vector(m);

            matrix_lo.quantize(matrix_32);

            matrix_lo.mvm(vector_32, result_vector);
            matrix_lo.mvm_scalar(vector_32, result_scalar);


            for (int k = 0; k < m; k += 1) {

                float x1 = result_vector.get(k);
                float x2 = result_scalar.get(k);

                if (fabs(x1 - x2) > 0.01) {
                    cout << "Fails at position: " << k << endl;
                    cout << endl << endl;

                    cout << "Resulting vectors: " << endl;
                    cout << compare(result_vector.toString(), result_scalar.toString()) << endl;
                    cout << endl << endl;
//
//                    cout << "Quantized matrix is: " << endl;
//                    cout << matrix_lo.toString() << endl;
//                    cout << endl << endl;
//
//                    cout << "Quantized vector is: " << endl;
//                    cout << vector_lo.toString() << endl;
//                    cout << endl << endl;
//
//                    cout << "Matrix is: " << endl;
//                    cout << matrix_32.toString() << endl;
//                    cout << endl << endl;
//
//                    cout << "Vector is: " << endl;
//                    cout << vector_32.toString() << endl;
//                    cout << endl << endl;
//
//                    matrix_32.mvm(vector_32, result_32);
//                    cout << "32-bit result is: " << endl;
//                    cout << result_32.toString() << endl;
//                    cout << endl << endl;

                    cout << "Exiting ... " << endl;
                    exit(1);
                }

            }

        }
    }
    cout << "Good" << endl;
}



template<class QMatrix, class QVector>
void validate_matrix_MVM_parallel(int bitcount, int skip_validation)
{
    cout << right << std::setw(2) << bitcount << "-bit | " << left << std::setw(30) << "MVM Parallel" << ": ";

    if (!skip_validation) {
        for (int i = 1; i <= 10; i += 1) {
            for (int j = 1; j <= 10; j += 1) {

                int m = i * 128;
                int n = j * 128;

                CloverMatrix32 m32(m, n);
                CloverVector32 v32(n);
                CloverVector32 r32(m);
                m32.setRandomInteger(10);
                v32.setRandomInteger(10);

                QMatrix m_01(m, n);
                QVector v_01(n);

                QVector result_scalar(m);
                QVector result_parallel(m);

                m_01.quantize(m32);
                v_01.quantize(v32);

                m_01.mvm(v_01, result_parallel);
                m_01.mvm_parallel(v_01, result_scalar);


                for (int k = 0; k < m; k += 1) {

                    float x1 = result_parallel.get(k);
                    float x2 = result_scalar.get(k);

                    if (x1 != x2) {
                        cout << "Fails at position: " << k << endl;
                        cout << endl << endl;

                        cout << "Resulting vectors: " << endl;
                        cout << compare(result_parallel.toString(), result_scalar.toString()) << endl;
                        cout << endl << endl;

                        cout << "Quantized matrix is: " << endl;
                        cout << m_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Quantized vector is: " << endl;
                        cout << v_01.toString() << endl;
                        cout << endl << endl;

                        cout << "Matrix is: " << endl;
                        cout << m32.toString() << endl;
                        cout << endl << endl;

                        cout << "Vector is: " << endl;
                        cout << v32.toString() << endl;
                        cout << endl << endl;

                        m32.mvm(v32, r32);
                        cout << "32-bit result is: " << endl;
                        cout << r32.toString() << endl;
                        cout << endl << endl;

                        cout << "Exiting ... " << endl;
                        exit(1);
                    }

                }

            }
        }
        cout << "Good" << endl;
    } else {
        cout << "Skipped" << endl;
    }

}

void validate_matrix_ops ()
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
    cout << "= Validating Matrix operations:" << endl;
    cout << "----------------------------------------------------------------------" << endl;

    //
    // Validate quantization
    //
    validate_matrix_quantization<CloverMatrix4> ( 4, !stochastic_rounding_disabled);
    validate_matrix_quantization<CloverMatrix8> ( 8, !stochastic_rounding_disabled);
    validate_matrix_quantization<CloverMatrix16>(16, false);
    //
    // Validate consitency
    //
    validate_matrix_consistency<CloverMatrix4> (4, !stochastic_rounding_disabled);
    validate_matrix_consistency<CloverMatrix8> (4, !stochastic_rounding_disabled);
    validate_matrix_consistency<CloverMatrix16>(16, false);

    validate_matrix_MVM<CloverMatrix4 , CloverVector4 >(4, !stochastic_rounding_disabled);
    validate_matrix_MVM<CloverMatrix8 , CloverVector8 >(8, !stochastic_rounding_disabled);
    validate_matrix_MVM<CloverMatrix16, CloverVector16>(16, false);
    validate_matrix_MVM<CloverMatrix32, CloverVector32>(32, false);
    validate_matrix_MVM_mixed_mat4_vec8(!stochastic_rounding_disabled);

    validate_matrix_MVM_parallel<CloverMatrix4 , CloverVector4 >(4 , !openMPEnabled || !stochastic_rounding_disabled);
    validate_matrix_MVM_parallel<CloverMatrix8 , CloverVector8 >(8 , !openMPEnabled || !stochastic_rounding_disabled);
    validate_matrix_MVM_parallel<CloverMatrix16, CloverVector16>(16, !openMPEnabled);

    validate_matrix_vector_product_32<CloverMatrix4 >(4);
    validate_matrix_vector_product_32<CloverMatrix8 >(8);
    validate_matrix_vector_product_32<CloverMatrix16>(16);
    validate_matrix_vector_product_32<CloverMatrix32>(32);

    validate_matrix_transpose<CloverMatrix4> ( 4);
    validate_matrix_transpose<CloverMatrix8> ( 8);
    validate_matrix_transpose<CloverMatrix16>(16);
    validate_matrix_transpose<CloverMatrix32>(32);

    validate_matrix_transpose_parallel<CloverMatrix4> ( 4, !openMPEnabled);
    validate_matrix_transpose_parallel<CloverMatrix8> ( 8, !openMPEnabled);
    validate_matrix_transpose_parallel<CloverMatrix16>(16, !openMPEnabled);
    validate_matrix_transpose_parallel<CloverMatrix32>(32, !openMPEnabled);

    cout << "======================================================================" << endl;
    cout << endl;

}
