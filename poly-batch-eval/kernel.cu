
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#include <cstddef>
#include <iostream>
#include <assert.h>
#include <numeric>
#include <chrono>
#include <vector>

#include "fp.h"

#define cuda_check(ans) { cuda_error_check((ans), __FILE__, __LINE__); }

inline void cuda_error_check(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(code) << "  at " << file << ":" << line << std::endl;
        if (abort) {
            exit(code);
        }
    }
}

typedef unsigned char byte;

size_t binomial(size_t n, size_t k) {
    assert(k <= n);
    k = std::min(k, n - k);
    size_t num = 1;
    size_t den = 1;
    for (size_t i = 0; i < k; ++i) {
        num *= (n - i);
        den *= (i + 1);
    }
    return num / den;
}

template<size_t k>
__device__ __host__ constexpr size_t static_binomial(size_t n) {
    if (k > n) {
        return 0;
    }
    size_t num = 1;
    size_t den = 1;
    for (size_t i = 0; i < k; ++i) {
        num *= (n - i);
        den *= (i + 1);
    }
    return num / den;
}

template<>
__device__ __host__ constexpr size_t static_binomial<0>(size_t n) {
    return 1;
}

int64_t read_int_from_poly_file(const byte* data) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; --i) {
        result = (result << 8) + static_cast<uint64_t>(data[i]);
    }
    return static_cast<int64_t>(result);
}

std::vector<int16_t> read_poly(const char* poly_identifier, size_t d, size_t m) {
    auto filename = std::string("poly_") + poly_identifier;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: \"" << filename << "\": " << std::strerror(errno) << std::endl;
        throw std::exception();
    }
    std::vector<int16_t> result;
    result.reserve(binomial(d + m, m));
    
    byte buffer[8];
    for (size_t i = 0; i < binomial(d + m, m); ++i) {
        file.read(reinterpret_cast<char*>(buffer), 8);
        int64_t value = read_int_from_poly_file(buffer);
        assert(abs(value) < static_cast<int64_t>(std::numeric_limits<int16_t>().max()));
        result.push_back(static_cast<int16_t>(value));

    }
    return result;
}

void write_result(const char* poly_identifier, size_t p, const std::vector<uint16_t>& data, bool append = false) {
    auto filename = std::string("poly_db_") + std::to_string(p) + "_" + poly_identifier;
    std::ofstream file(filename, std::ios::binary | (append ? std::ios_base::app : 0));
    if (!file) {
        std::cerr << "Error: \"" << filename << "\": " << std::strerror(errno) << std::endl;
        throw std::exception();
    }

    file.write(reinterpret_cast<const char*>(&data[0]), 2 * data.size());
    file.close();
}

template<size_t m>
struct RevBlockIter {

private:
    size_t d;
    int current_index;
    size_t current_start;
    size_t current_end;

    __device__ __host__ inline RevBlockIter(size_t d, int current_index, size_t current_start, size_t current_end) : d(d), current_index(current_index), current_start(current_start), current_end(current_end) {}

public:

    RevBlockIter(const RevBlockIter&) = default;
    RevBlockIter(RevBlockIter&&) = default;
    ~RevBlockIter() = default;

    RevBlockIter& operator=(const RevBlockIter&) = default;
    RevBlockIter& operator=(RevBlockIter&&) = default;

    __device__ __host__ inline bool operator!=(const RevBlockIter& other) const {
        return current_index != other.current_index;
    }

    __device__ __host__ inline RevBlockIter& operator++() {
        static_assert(m >= 1, "m must not be zero");
        current_index -= 1;
        current_end = current_start;
        current_start = current_start - static_binomial<m - 1>(d - current_index + m - 1);
        return *this;
    }

    __device__ __host__ inline size_t operator*() const {
        return current_index;
    }

    __device__ __host__ inline size_t range_start() const {
        return current_start;
    }

    __device__ __host__ inline size_t range_end() const {
        return current_end;
    }

    __device__ __host__ static inline RevBlockIter begin(size_t d) {
        return RevBlockIter(d, d, static_binomial<m>(d + m) - 1, static_binomial<m>(d + m));
    }

    __device__ __host__ static inline RevBlockIter end(size_t d) {
        return RevBlockIter(d, -1, 0, 0);
    }
};

template<size_t m>
__device__ __host__ inline FpEl evaluate_poly(const Fp& field, const int16_t* poly, const FpEl* values, const size_t d) {
    RevBlockIter<m> it = RevBlockIter<m>::begin(d);
    const RevBlockIter<m> end = RevBlockIter<m>::end(d);
    FpEl current = 0.;
    for (; it != end; ++it) {
        current = field.mul_add(current, values[m - 1], evaluate_poly<m - 1>(field, poly + it.range_start(), values, d - *it));
    }
    return current;
}

template<>
__device__ __host__ inline FpEl evaluate_poly<0>(const Fp& field, const int16_t* poly, const FpEl* values, const size_t d) {
    return poly[0];
}

template<size_t m>
__device__ __host__ inline void index_to_point(const Fp& field, int64_t index, FpEl* output) {
    for (int i = m - 1; i >= 0; --i) {
        output[i] = field.from_int(index);
        index = (index - field.lift(output[i])) / field.modulus();
    }
    assert(index == 0);
}

template<size_t m>
__global__ void evaluate_poly_parallel(Fp field, const int16_t* poly, const size_t d, const int64_t start, const int64_t end, uint16_t* out) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int gridlen = blockDim.x * gridDim.x;
    FpEl point[m];
    for (int64_t current = start + tid; current < end; current += gridlen) {
        index_to_point<m>(field, current, point);
        out[current - start] = field.lift(evaluate_poly<m>(field, poly, point, d));
    }
}

template<size_t m>
std::vector<uint16_t> evaluate_poly_parallel(Fp field, const std::vector<int16_t>& poly, const size_t start, const size_t end, const size_t d) {
    assert(poly.size() == static_binomial<m>(d + m));
    int16_t* poly_device;
    cuda_check(cudaMalloc(&poly_device, poly.size() * sizeof(int16_t)));
    cuda_check(cudaMemcpy(poly_device, &poly[0], poly.size() * sizeof(int16_t), cudaMemcpyHostToDevice));

    size_t out_len = end - start;
    uint16_t* result_device;
    cuda_check(cudaMalloc(&result_device, out_len * sizeof(uint16_t)));

    evaluate_poly_parallel <m> <<<64, 256>>> (field, poly_device, d, start, end, result_device);
    cuda_check(cudaDeviceSynchronize());

    std::vector<uint16_t> result;
    result.resize(out_len);
    cuda_check(cudaMemcpy(&result[0], result_device, out_len * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(poly_device));
    cuda_check(cudaFree(result_device));

    return result;
}

int main()
{
    constexpr size_t d = 18;
    constexpr size_t m = 4;
    const char* filehash = "sX9KMQ";
    std::vector<int16_t> poly = read_poly(filehash, d, m);
    auto start = std::chrono::steady_clock::now();
    for (uint16_t p : primes) {
        if (p > 191) {
            auto end = std::chrono::steady_clock::now();
            std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(start - end).count() << std::endl;
            return 0;
        }
        size_t start = 0;
        const size_t total_len = std::pow(p, m);
        const size_t batch_len = 1 << 30;
        size_t end = std::min(total_len, batch_len);
        std::vector<uint16_t> result = evaluate_poly_parallel<m>(Fp{ p }, poly, start, end, d);
        write_result(filehash, p, result);
        while (end < total_len) {
            start = end;
            end = std::min(total_len, end + batch_len);
            std::vector<uint16_t> result = evaluate_poly_parallel<m>(Fp{ p }, poly, start, end, d);
            write_result(filehash, p, result, true);
        }
        std::cout << "done " << p << std::endl;
    }
    return 0;
}
