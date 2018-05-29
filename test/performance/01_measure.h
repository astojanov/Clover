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

#ifndef CLOVER_MEASURE_H
#define CLOVER_MEASURE_H

#include <cmath>
#include <tuple>
#include <thread>
#include <algorithm>
#include <CloverVector32.h>
#include <CloverMatrix32.h>
#include "../../lib/perf.h"
#include "../random/00_random.h"

#include "03_iht_gd_util.h"

#define MEASURE_REPETITIONS 15
#define MEASURE_THRESHOLD 1e6

typedef struct {
    benchmark_t bench;
    double bandwidth;
} measurement_t;

// ===============================================================================================================
// = Individual Matrix / Vector operations
// ===============================================================================================================

template <class QVector>
inline benchmark_t benchmark_dot_scalar(QVector const &u, QVector const &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            tmp += u.dot_scalar(v);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            tmp += u.dot_scalar(v);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
inline benchmark_t benchmark_vector_dot(QVector const &u, QVector const &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            tmp += u.dot(v);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            tmp += u.dot(v);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
inline benchmark_t benchmark_vector_dot_parallel(QVector const &u, QVector const &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            tmp += u.dot_parallel(v);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            tmp += u.dot_parallel(v);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
benchmark_t benchmark_quantize_scalar(CloverVector32 const &u, QVector &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            v.quantize_scalar(u);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            v.quantize_scalar(u);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class Raw, class Quantized>
benchmark_t benchmark_quantize(Raw const &u, Quantized &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            v.quantize(u);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            v.quantize(u);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class Quantized>
benchmark_t benchmark_vector_get(const Quantized &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    const uint64_t n = v.size_pad();
    float acc1 = 0;
    float acc2 = 0;
    float acc3 = 0;
    float acc4 = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            for (uint64_t idx = 0; idx < n; idx += 4) {
                acc1 += v.get(idx + 0);
                acc2 += v.get(idx + 1);
                acc3 += v.get(idx + 2);
                acc4 += v.get(idx + 3);
            }
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);

    acc1 /= 1000000.0f;
    acc2 /= 1000000.0f;
    acc3 /= 1000000.0f;
    acc4 /= 1000000.0f;

    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            for (uint64_t idx = 0; idx < n; idx += 4) {
                acc1 += v.get(idx + 0);
                acc2 += v.get(idx + 1);
                acc3 += v.get(idx + 2);
                acc4 += v.get(idx + 3);
            }
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }

    tmp = acc1 + acc2 + acc3 + acc4;

    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}


template <class Raw, class Quantized>
benchmark_t benchmark_quantize_parallel(Raw const &u, Quantized &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            v.quantize_parallel(u);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            v.quantize_parallel(u);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QMatrix, class QVector>
benchmark_t benchmark_matrix_MVM(QMatrix &u, const QVector &v, QVector &r)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.mvm(v, r);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.mvm(v, r);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QMatrix, class QVector>
benchmark_t benchmark_matrix_MVM_parallel(QMatrix &u, const QVector &v, QVector &r)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.mvm_parallel(v, r);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.mvm_parallel(v, r);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QMatrix>
benchmark_t benchmark_matrix_transpose(QMatrix &u, QMatrix &r)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.transpose(r);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.transpose(r);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QMatrix>
benchmark_t benchmark_matrix_transpose_parallel(QMatrix &u, QMatrix &r)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.transpose_parallel(r);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.transpose_parallel(r);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
benchmark_t benchmark_vector_scaleandadd(QVector &u, QVector const &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.scaleAndAdd(v, CloverRandom::get_random_float());
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.scaleAndAdd(v, CloverRandom::get_random_float());
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
benchmark_t benchmark_vector_scaleandadd_parallel(QVector &u, QVector const &v)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.scaleAndAdd_parallel(v, CloverRandom::get_random_float());
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            u.scaleAndAdd_parallel(v, CloverRandom::get_random_float());
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
benchmark_t benchmark_vector_threshold(QVector &u, const CloverVector32 &init, int k)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;

        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.threshold(k);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        u.quantize(init);

        cycles_count_start();
        u.threshold(k);
        score = cycles_count_stop();

        score.cycles = (uint64_t)((double) score.cycles );
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}

template <class QVector>
benchmark_t benchmark_vector_threshold_parallel(QVector &u, const CloverVector32 &init, int k)
{
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[MEASURE_REPETITIONS + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;

        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            u.threshold_parallel(k);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < MEASURE_REPETITIONS; i += 1) {
        u.quantize(init);

        cycles_count_start();
        u.threshold_parallel(k);
        score = cycles_count_stop();

        score.cycles = (uint64_t)((double) score.cycles );
        results[i] = score;
    }
    results[MEASURE_REPETITIONS].cycles = (uint64_t) tmp;
    std::sort(results, results + MEASURE_REPETITIONS, cmp_benchmark_t);
    return results[MEASURE_REPETITIONS / 2];
}


template <class QVector>
measurement_t measure_vector_quantize(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    QVector qa(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    vector.bench = benchmark_quantize(a, qa);
    vector.bandwidth = (double) (a.getBytes() + qa.getBytes()) / (double) vector.bench.time;

    return vector;
}


template <class QVector>
measurement_t measure_vector_quantize_parallel(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    QVector qa(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    vector.bench = benchmark_quantize_parallel(a, qa);
    vector.bandwidth = (double) (a.getBytes() + qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QVector>
measurement_t measure_vector_get(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    QVector qa(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    qa.quantize_parallel(a);

    vector.bench = benchmark_vector_get(qa);
    vector.bandwidth = (double) (qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QMatrix>
measurement_t measure_matrix_quantize(uint64_t size)
{
    measurement_t vector;

    CloverMatrix32 a(size, size);
    QMatrix qa(size, size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    vector.bench = benchmark_quantize(a, qa);
    vector.bandwidth = (double) (a.getBytes() + qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QMatrix, class QVector>
measurement_t measure_matrix_MVM(uint64_t size)
{
    measurement_t vector;

    CloverMatrix32 a(size, size);
    CloverVector32 v(size);
    QMatrix qa(size, size);
    QVector qv(size);
    QVector qr(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    v.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qv.quantize(v);

    vector.bench = benchmark_matrix_MVM(qa, qv, qr);
    vector.bandwidth = (double) (qv.getBytes() + qa.getBytes() + qr.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QMatrix, class QVector>
measurement_t measure_matrix_MVM_parallel(uint64_t size)
{
    measurement_t vector;

    CloverMatrix32 a(size, size);
    CloverVector32 v(size);
    QMatrix qa(size, size);
    QVector qv(size);
    QVector qr(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    v.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qv.quantize(v);

    vector.bench = benchmark_matrix_MVM_parallel(qa, qv, qr);
    vector.bandwidth = (double) (qv.getBytes() + qa.getBytes() + qr.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QMatrix>
measurement_t measure_matrix_transpose(uint64_t size)
{
    measurement_t vector;

    CloverMatrix32 a(size, size);

    QMatrix qa(size, size);
    QMatrix qt(size, size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);

    vector.bench = benchmark_matrix_transpose(qa, qt);
    vector.bandwidth = (double) (qa.getBytes() + qt.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QMatrix>
measurement_t measure_matrix_transpose_parallel(uint64_t size)
{
    measurement_t vector;

    CloverMatrix32 a(size, size);

    QMatrix qa(size, size);
    QMatrix qt(size, size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);

    vector.bench = benchmark_matrix_transpose_parallel(qa, qt);
    vector.bandwidth = (double) (qa.getBytes() + qt.getBytes()) / (double) vector.bench.time;

    return vector;
}


template <class QVector>
measurement_t measure_vector_dot(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    CloverVector32 b(size);

    QVector qa(size);
    QVector qb(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    b.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qb.quantize(b);

    vector.bench = benchmark_vector_dot<QVector>(qa, qb);
    vector.bandwidth = (double) (2 * qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QVector>
measurement_t measure_vector_dot_parallel(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    CloverVector32 b(size);

    QVector qa(size);
    QVector qb(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    b.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qb.quantize(b);

    vector.bench = benchmark_vector_dot_parallel<QVector>(qa, qb);
    vector.bandwidth = (double) (2 * qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QVector>
measurement_t measure_vector_scaleandadd(uint64_t size)
{
    measurement_t vector;

    CloverVector32 a(size);
    CloverVector32 b(size);

    QVector qa(size);
    QVector qb(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    b.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qb.quantize(b);

    vector.bench = benchmark_vector_scaleandadd<QVector>(qa, qb);
    vector.bandwidth = (double) (3 * qa.getBytes()) / (double) vector.bench.time;

    return vector;
}

template <class QVector>
measurement_t measure_vector_scaleandadd_parallel(uint64_t size)
{
    measurement_t measure;

    CloverVector32 a(size);
    CloverVector32 b(size);

    QVector qa(size);
    QVector qb(size);

    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    b.setRandomInteger(10, random_FVector_key1, random_FVector_key2);

    qa.quantize(a);
    qb.quantize(b);

    measure.bench = benchmark_vector_scaleandadd_parallel<QVector>(qa, qb);
    measure.bandwidth = (double) (3 * qa.getBytes()) / (double) measure.bench.time;

    return measure;
}


template <class QVector>
measurement_t measure_vector_threshold(uint64_t size, uint64_t k)
{
    measurement_t vector;

    CloverVector32 a(size);
    QVector qa(size);
    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    qa.quantize(a);

    vector.bench = benchmark_vector_threshold<QVector>(qa, a, k);
    vector.bandwidth = (double) (2 * qa.getBytes()) / (double) vector.bench.time;

    return vector;
}


template <class QVector>
measurement_t measure_vector_threshold_parallel(uint64_t size, uint64_t k)
{
    measurement_t measure;

    CloverVector32 a(size);
    QVector qa(size);
    a.setRandomInteger(10, random_FVector_key1, random_FVector_key2);
    qa.quantize(a);

    measure.bench = benchmark_vector_threshold_parallel<QVector>(qa, a, k);
    measure.bandwidth = (double) (2 * qa.getBytes()) / (double) measure.bench.time;

    return measure;
}

// ===============================================================================================================
// = Quantized Linear Algebra Application
// ===============================================================================================================

/**
 * Quantized Iterative Hard Thresholding to perform Compressive Sensing. Implementation based on
 * "Compressive sensing a summary of reconstruction algorithms" by Pope Graeme [1], equation 5.60
 * (page 66).
 *
 * [1] https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/150999/eth-41464-01.pdf
 */
template <class QMatrix, class QVector>
inline void Q_IHT(
        QMatrix &Phi,            // Measurements matrix \Phi [MxN]
        QMatrix &PhiT,           // \Phi, but transposed
        QVector &x,              // Real valued, finite length, one-dimensional, discrete time signal [N]
        QVector &y,              // Real valued vector of observations [M]
        QVector &t1,             // temporary vector
        QVector &t2,             // temporary vector
        QVector &t3,             // temporary vector
        const uint64_t iterations,  // Number of iterations
        const uint64_t K,           // Thresholding parameter
        const float mu
) {
    x.clear();

    for(uint64_t i = 0; i < iterations; i += 1)
    {
        Phi.mvm_parallel(x, t1);                // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);  // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);              // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, mu);         // x = x + \mu * Phi' * (y - Phi * x)
        x.threshold_parallel(K);                // perform hard tresholding
    }
}

template <class QMatrix, class QVector>
benchmark_t benchmark_IHT(
        QMatrix &Phi,                // Measurements matrix \Phi [MxN]
        QMatrix &PhiT,               // \Phi, but transposed
        QVector &x,                  // Real valued, finite length, one-dimensional, discrete time signal [N]
        QVector &y,                  // Real valued vector of observations [M]
        QVector &t1,                 // temporary vector
        QVector &t2,                 // temporary vector
        QVector &t3,                 // temporary vector
        const uint64_t iterations,      // Number of iterations
        const uint64_t K,               // Thresholding parameter
        const float mu
) {

    int measure_repetitions = 4;
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[measure_repetitions + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            Q_IHT(Phi, PhiT, x, y, t1, t2, t3, iterations, K, mu);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < measure_repetitions; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            Q_IHT(Phi, PhiT, x, y, t1, t2, t3, iterations, K, mu);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[measure_repetitions].cycles = (uint64_t) tmp;
    std::sort(results, results + measure_repetitions, cmp_benchmark_t);
    return results[measure_repetitions / 2];
}



template <class QMatrix, class QVector>
inline void Q_GD(
        QMatrix &Phi,            // Measurements matrix \Phi [MxN]
        QMatrix &PhiT,           // \Phi, but transposed
        QVector &x,              // Real valued, finite length, one-dimensional, discrete time signal [N]
        QVector &y,              // Real valued vector of observations [M]
        QVector &t1,             // temporary vector
        QVector &t2,             // temporary vector
        QVector &t3,             // temporary vector
        const uint64_t iterations,  // Number of iterations
        const float mu
) {
    x.clear();

    for(uint64_t i = 0; i < iterations; i += 1)
    {
        Phi.mvm_parallel(x, t1);                // t1 = Phi * x
        y.scaleAndAdd_parallel(t1, -1.0f, t2);  // t2 = y - Phi * x
        PhiT.mvm_parallel(t2, t3);              // t3 = Phi' * (y - Phi * x)
        x.scaleAndAdd_parallel(t3, mu);         // x = x + \mu * Phi' * (y - Phi * x)
    }
}

template <class QMatrix, class QVector>
benchmark_t benchmark_GD(
        QMatrix &Phi,                // Measurements matrix \Phi [MxN]
        QMatrix &PhiT,               // \Phi, but transposed
        QVector &x,                  // Real valued, finite length, one-dimensional, discrete time signal [N]
        QVector &y,                  // Real valued vector of observations [M]
        QVector &t1,                 // temporary vector
        QVector &t2,                 // temporary vector
        QVector &t3,                 // temporary vector
        const uint64_t iterations,      // Number of iterations
        const float mu
) {

    int measure_repetitions = 4;
    int runs = 1;
    int multiplier = 1;
    uint64_t cycles;
    benchmark_t score;

    benchmark_t results[measure_repetitions + 1];
    double tmp = 0;

    do {
        runs = runs * multiplier;
        cycles_count_start();
        for (int i = 0; i < runs; i += 1) {
            Q_GD(Phi, PhiT, x, y, t1, t2, t3, iterations, mu);
        }
        cycles = cycles_count_stop().cycles;
        multiplier = (int) ceil (  (MEASURE_THRESHOLD * 100.0) / (cycles * 1.0 * runs)  + 1.0 );
    } while (multiplier > 2);


    for (int i = 0; i < measure_repetitions; i += 1) {
        cycles_count_start();
        for (int j = 0; j < runs; ++j) {
            Q_GD(Phi, PhiT, x, y, t1, t2, t3, iterations, mu);
        }
        score = cycles_count_stop();
        score.cycles = (uint64_t)((double) score.cycles / runs);
        score.time   /= (double) runs;
        results[i] = score;
    }
    results[measure_repetitions].cycles = (uint64_t) tmp;
    std::sort(results, results + measure_repetitions, cmp_benchmark_t);
    return results[measure_repetitions / 2];
}



template <class QMatrix, class QVector>
measurement_t measure_IHT_or_GD(
        problem_type_t problem_type,
        CloverVector32 &x_32, CloverMatrix32 &Phi_32, CloverVector32 &y_32, uint64_t &K, float &mu, uint64_t &iterations
) {
    uint64_t bytes = 0;
    measurement_t measure;
    //
    // Get the dimension os the original matrix
    //
    const uint64_t m = Phi_32.getRows();
    const uint64_t n = Phi_32.getCols();
    //
    // Create the memory spaces
    //
    QMatrix Phi (m, n);
    QMatrix PhiT(n, m);
    QVector x(n);
    QVector y(m);
    QVector t1(m);
    QVector t2(m);
    QVector t3(n);
    //
    // Quantize what is necessary
    //
    Phi.quantize(Phi_32);
    x.quantize(x_32);
    y.quantize(y_32);
    //
    // Perform transposition in advance
    //
    Phi.transpose_parallel(PhiT);
    //
    // Let the experiment run
    //
    switch (problem_type) {
        case QUANTIZED_ITERATIVE_HARD_THRESHOLDING: {
            measure.bench = benchmark_IHT(Phi, PhiT, x, y, t1, t2, t3, iterations, K, mu);
            break;
        }
        case QUANTIZED_GRADIENT_DESCENT: {
            measure.bench = benchmark_GD(Phi, PhiT, x, y, t1, t2, t3, iterations, mu);
            break;
        }
    }

    //
    // Let's define a bandwidth estimation (this is just an very rough approximation
    //
    bytes += Phi.getBytes();
    bytes += PhiT.getBytes();
    bytes += x.getBytes();
    bytes += y.getBytes();
    bytes += t1.getBytes();
    bytes += t2.getBytes();
    bytes += t3.getBytes();

    measure.bandwidth = (double) bytes / (double) measure.bench.time * iterations;
    return measure;
};


#endif /* CLOVER_MEASURE_H */
