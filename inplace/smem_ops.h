#pragma once
#include "index.h"

template<typename F>
__global__ void smem_row_shuffle(F s, uint32_t height, uint32_t width, float *d) {
  extern __shared__ float *shared_row;
  for (uint32_t i = blockIdx.x; i < height; i += gridDim.x) {
    row_major_index rm(height, width);
    s.set_i(i);
    for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
      shared_row[j] = d[rm(i, j)];
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
      d[rm(i, j)] = shared_row[s(j)];
    }
    __syncthreads();
  }
}
