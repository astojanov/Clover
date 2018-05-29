#ifndef SIMD_DEBUG_H
#define SIMD_DEBUG_H

#include <immintrin.h>
#include <string>

void print_epi8  (__m256i a);
void print_epi16 (__m256i a);
void print_epi32 (__m256i a);
void print_ps    (__m256  a);

std::string compare(std::string a, std::string b);

#endif // SIMD_DEBUG_H
