#pragma once
#include <set>
#include <vector>
#include "gcd.h"
#include "index.h"
#include "util.h"

template<typename Fn>
static void scatter_cycles(
    const Fn &f, std::vector<uint32_t> *heads, std::vector<uint32_t> *lens) {
  uint32_t len = f.len();
  std::set<uint32_t> unvisited;
  for (uint32_t i = 0; i < len; i++) {
    unvisited.insert(i);
  }
  while (!unvisited.empty()) {
    uint32_t idx = *unvisited.begin();
    unvisited.erase(unvisited.begin());
    uint32_t dest = f(idx);
    if (idx != dest) {
      heads->push_back(idx);
      uint32_t start = idx;
      uint32_t len = 1;
      while (dest != start) {
        idx = dest;
        unvisited.erase(idx);
        dest = f(idx);
        len++;
      }
      lens->push_back(len);
    }
  }
}

template<typename F, uint32_t U>
__device__ __forceinline__ void unroll_cycle_row_permute(
    F f, row_major_index rm, float* data, uint32_t i, uint32_t j, uint32_t l) {
  float src = data[rm(i, j)];
  float loaded[U + 1];
  loaded[0] = src;
  for (uint32_t k = 0; k < l / U; k++) {
    uint32_t rows[U];
    #pragma unroll
    for (uint32_t x = 0; x < U; x++) {
      i = f(i);
      rows[x] = i;
    }
    #pragma unroll
    for (uint32_t x = 0; x < U; x++) {
      loaded[x + 1] = data[rm(rows[x], j)];
    }
    #pragma unroll
    for (uint32_t x = 0; x < U; x++) {
      data[rm(rows[x], j)] = loaded[x];
    }
    loaded[0] = loaded[U];
  }
  float tmp = loaded[0];
  for (uint32_t k = 0; k < l % U; k++) {
    i = f(i);
    float new_tmp = data[rm(i, j)];
    data[rm(i, j)] = tmp;
    tmp = new_tmp;
  }
}

template<typename F, uint32_t U>
__global__ void cycle_row_permute(F f, float* data, uint32_t* heads,
                                  uint32_t* lens, uint32_t n_heads) {
  volatile uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  volatile uint32_t h = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t n = f.n;
  row_major_index rm(f.m, f.n);

  if ((j < n) && (h < n_heads)) {
    uint32_t i = heads[h];
    uint32_t l = lens[h];
    unroll_cycle_row_permute<T, F, U>(f, rm, data, i, j, l);
  }
}

template<typename F>
static void scatter_permute(
    F f, uint32_t height, uint32_t n, float* data, uint32_t* tmp) {
  std::vector<uint32_t> heads;
  std::vector<uint32_t> lens;
  scatter_cycles(f, heads, lens);
  uint32_t* d_heads = tmp;
  uint32_t* d_lens = tmp + height / 2;
  cudaMemcpy(d_heads, heads.data(), sizeof(uint32_t) * heads.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_lens, lens.data(), sizeof(uint32_t) * lens.size(),
             cudaMemcpyHostToDevice);
  uint32_t n_threads_x = 256;
  uint32_t n_threads_y = 1024 / n_threads_x;
  uint32_t n_blocks_x = div_up(n, n_threads_x);
  uint32_t n_blocks_y = div_up(heads.size(), n_threads_y);
  cycle_row_permute<F, 4>
      <<<dim3(n_blocks_x, n_blocks_y), dim3(n_threads_x, n_threads_y)>>>
      (f, data, d_heads, d_lens, heads.size());
}

