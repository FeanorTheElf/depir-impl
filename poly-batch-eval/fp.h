#pragma once
#include <cstdint>
#include <cmath>
#include <assert.h>
#include <string>

#include "primes.h"

inline void test_fp(uint16_t prime) {
    int64_t p = static_cast<int64_t>(prime);
    const float pf = static_cast<float>(p);
    const float p_inv = 1.f / static_cast<float>(p);
    const int64_t max_value = std::max(p * p + p, ((int64_t)1 << 16) - 1);
    for (int64_t a = -max_value; a <= max_value; ++a) {
        int64_t expected = a % p;
        float af = static_cast<float>(a);
        float actual = (af - round(af * p_inv) * p);
        
        if ((static_cast<int64_t>(actual) - expected) % p != 0) {
            throw "failed congruence";
        }
        else if (actual < -pf || actual > pf) {
            throw "failed reducedness";
        }
    }
}

inline void test_all_fp() {
    for (uint16_t p : primes) {
        try {
            test_fp(p);
            std::cout << p << std::endl;
        }
        catch (const char* exception) {
            throw std::string("failed for ") + std::to_string(p) + "; " + exception;
        }
    }
}

typedef float FpEl;

struct Fp {

private:
    float p;
    float p_inv;

public:
    __device__ __host__ inline Fp(uint16_t p) : p(static_cast<float>(p)), p_inv(1.f / static_cast<float>(p)) {
        // run test_all_fp() to see that it fails from this value on (makes sense, as it will never work for primes of more than 12 bits)
        assert(p <= 4093);
    }

    Fp(const Fp&) = default;
    Fp(Fp&&) = default;
    ~Fp() = default;

    Fp& operator=(const Fp&) = default;
    Fp& operator=(Fp&&) = default;

    __device__ __host__ inline float reduce(float value) const {
        return value - p * round(value * p_inv);
    }

    __device__ __host__ inline float mul(float lhs, float rhs) const {
        assert(lhs >= -p && lhs <= p);
        assert(rhs >= -p && lhs <= p);
        return reduce(lhs * rhs);
    }

    __device__ __host__ inline float from_int(int16_t n) const {
        return reduce(static_cast<float>(n));
    }

    __device__ __host__ inline float from_int(int64_t n) const {
        int64_t result = n;
        // each statement reduces the size by almost 24 bits
        result -= static_cast<uint64_t>(p) * static_cast<int64_t>(round(static_cast<float>(result) * p_inv));
        result -= static_cast<uint64_t>(p) * static_cast<int64_t>(round(static_cast<float>(result) * p_inv));
        return reduce(static_cast<float>(result));
    }

    __device__ __host__ inline float mul_add(float lhs, float rhs, float add) const {
        return reduce(lhs * rhs + add);
    }

    __device__ __host__ inline uint16_t modulus() const {
        return static_cast<uint16_t>(p);
    }

    __device__ __host__ inline uint16_t lift(FpEl x) const {
        uint16_t result = static_cast<uint16_t>(x + p);
        if (result >= p) {
            result -= static_cast<uint16_t>(p);
        }
        if (result >= p) {
            result -= static_cast<uint16_t>(p);
        }
        return result;
    }
};