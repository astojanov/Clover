#include <stdio.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <sstream>
#include <iomanip>
#include "simd_debug.h"

void print_epi8(__m256i a)
{
    int i; int8_t tmp[32];
    _mm256_storeu_si256( (__m256i*) tmp, a);
    for (i = 0; i < 32; i += 1) {
        printf("%3d ", (int) tmp[i]);
    }
    printf("\n");
}

void print_epi16(__m256i a)
{
    int i; int16_t tmp[16];
    _mm256_storeu_si256( (__m256i*) tmp, a);

    for (i = 0; i < 16; i += 1) {
        printf("%8d ", (int) tmp[i]);
    }
    printf("\n");
}

void print_epi32(__m256i a)
{
    int i; int32_t tmp[8];
    _mm256_storeu_si256( (__m256i*) tmp, a);
    for (i = 0; i < 8; i += 1) {
        printf("%d ", (int) tmp[i]);
    }
    printf("\n");
}

void print_epu32_hex(__m256i a)
{
    int i; uint32_t tmp[8];
    _mm256_storeu_si256( (__m256i*) tmp, a);
    for (i = 0; i < 8; i += 1) {
        printf("0x%08x ", (uint32_t) tmp[i]);
    }
    printf("\n");
}

void print_ps(__m256 a)
{
    int i; float tmp[8];
    _mm256_storeu_ps( tmp, a);
    for (i = 0; i < 8; i += 1) {
        std::cout << std::setw(7) << std::fixed << std::setprecision(2) << tmp[i] << " ";
//        printf("%02.2f ", tmp[i]);
    }
    printf("\n");
}

std::vector<std::string> split_string(
        const std::string& str,
        const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}


std::string compare(std::string a, std::string b)
{
    std::stringstream sout;
    std::vector<std::string> sa = split_string(a, "\n");
    std::vector<std::string> sb = split_string(b, "\n");

    for (int i = 0; i < sa.size(); i += 1) {
        sout << sa[i] << "  ||||||||  " << sb[i] << std::endl;
    }

    return sout.str();
}
