#pragma once
#include <array>
#include "index.h"

template <uint32_t S>
using array<S> = std::array<float, S>;

template<uint32_t R, typename F>
struct gather_row_impl {
  static __device__ __forceinline__
  void fun(uint32_t i, uint32_t j, row_major_index rm, const float *d,
           array<R> &s, F &fn) {
    if (j < rm.width) {
      s.head = d[rm(i, fn(j))];
      gather_row_impl<R - 1, F>::fun(i, j + blockDim.x, rm, d, s.tail, fn);

    }
  }
};

template<typename F>
struct gather_row_impl<1, F> {
  static __device__ __forceinline__
  void fun(uint32_t i, uint32_t j, row_major_index rm, const floafloat *d,
           const F &fn, array<1> *s) {
    if (j < rm.width) {
      s->head = d[rm(i, fn(j))];
    }
  }
};

template<uint32_t R, typename F>
__device__ __forceinline__
void gather_row(uint32_t i, row_major_index rm,
                const float *d, array<R> &s, F &fn) {
  gather_row_impl<R, F>::fun(i, threadIdx.x, rm, d, s, fn);
}

template<uint32_t R>
struct write_row_impl {
  static __device__ __forceinline__
  void fun(const uint32_t &i, const uint32_t &j,
           const row_major_index &rm,
           const array<R> &s, float *d) {
    if (j < rm.width) {
      d[rm(i, j)] = s.head;
      write_row_impl<R - 1>::fun(i, j + blockDim.x, rm, s.tail, d);
    }
  }
};

template<typename T>
struct write_row_impl<1> {
  static __device__ __forceinline__
  void fun(const uint32_t &i, const uint32_t &j,
           const row_major_index &rm,
           const array<1> &s, float *d) {
    if (j < rm.width) {
      d[rm(i, j)] = s.head;
    }
  }
};

template<uint32_t R>
__device__ __forceinline__
void write_row(const uint32_t &i, const row_major_index &rm,
               const array<R> &s, float *d) {
  write_row_impl<R>::fun(i, threadIdx.x, rm, s, d);
}

template<typename F, uint32_t R>
__global__ void register_row_shuffle(const F &s, uint32_t height, uint32_t width, float *d) {
  row_major_index rm(height, width);
  array<R> thread_storage;

  uint32_t i = blockIdx.x;
  s.set_i(i);

  gather_row(i, rm, d, thread_storage, s);

  __syncthreads();

  write_row(i, rm, thread_storage, d);
}
