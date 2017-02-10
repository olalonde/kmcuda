#pragma once

#include "inplace/util.h"
#include "inplace/equations.h"

__device__ __forceinline__
uint32_t ctz(uint32_t x) {
  return __ffs(x) - 1;
}

__device__ __forceinline__
uint32_t gcd(uint32_t x, uint32_t y) {
  if (x == 0) {
    return y;
  }
  if (y == 0) {
    return x;
  }
  uint32_t cf2 = ctz(x | y);
  x >>= ctz(x);
  while (true) {
    y >>= ctz(y);
    if (x == y) {
      break;
    }
    if (x > y) {
      uint32_t t = x;
      x = y;
      y = t;
    }
    if (x == 1) {
      break;
    }
    y -= x;
  }
  return x << cf2;
}

template<typename F>
__global__ void coarse_col_rotate(F fn, reduced_divisor m, uint32_t n, float *d) {
  uint32_t warp_id = threadIdx.x & 0x1f;
  uint32_t global_index = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t rotation_amount = fn(fn.master(global_index, warp_id, 32));
  uint32_t col = global_index;

  __shared__ T smem[32 * 16];

  if ((col < n) && (rotation_amount > 0)) {
    row_major_index rm(m, n);
    uint32_t c = gcd(rotation_amount, m.get());
    uint32_t l = m.get() / c;
    uint32_t inc = m.get() - rotation_amount;
    uint32_t smem_write_idx = threadIdx.y * 32 + threadIdx.x;
    uint32_t max_col = (l > 16) ? 15 : l - 1;
    uint32_t smem_read_col = (threadIdx.y == 0) ? max_col : (threadIdx.y - 1);
    uint32_t smem_read_idx = smem_read_col * 32 + threadIdx.x;

    for (uint32_t b = 0; b < c; b++) {
      uint32_t x = threadIdx.y;
      uint32_t pos = m.mod(b + x * inc);
      smem[smem_write_idx] = d[rm(pos, col)];
      __syncthreads();
      T prior = smem[smem_read_idx];
      if (x < l) {
        d[rm(pos, col)] = prior;
      }
      __syncthreads();
      uint32_t n_rounds = l / 16;
      for (uint32_t i = 1; i < n_rounds; i++) {
        x += blockDim.y;
        uint32_t pos = m.mod(b + x * inc);
        if (x < l) {
          smem[smem_write_idx] = d[rm(pos, col)];
        }
        __syncthreads();
        T incoming = smem[smem_read_idx];
        T outgoing = (threadIdx.y == 0) ? prior : incoming;
        if (x < l) {
          d[rm(pos, col)] = outgoing;
        }
        prior = incoming;
        __syncthreads();
      }
      //Last round/cleanup
      x += blockDim.y;
      pos = m.mod(b + x * inc);
      if (x <= l) {
        smem[smem_write_idx] = d[rm(pos, col)];
      }
      __syncthreads();
      uint32_t remainder_length = (l % 16);
      uint32_t fin_smem_read_col =
          (threadIdx.y == 0) ? remainder_length : threadIdx.y - 1;
      uint32_t fin_smem_read_idx = fin_smem_read_col * 32 + threadIdx.x;
      T incoming = smem[fin_smem_read_idx];
      T outgoing = (threadIdx.y == 0) ? prior : incoming;
      if (x <= l) {
        d[rm(pos, col)] = outgoing;
      }
    }
  }
}

template<typename F>
__global__ void fine_col_rotate(F fn, uint32_t m, uint32_t n, float *d) {
  __shared__ T smem[32 * 32];

  uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col < n) {
    uint32_t warp_id = threadIdx.x & 0x1f;
    uint32_t coarse_rotation_amount = fn(fn.master(col, warp_id, 32));
    uint32_t overall_rotation_amount = fn(col);
    uint32_t
        fine_rotation_amount = overall_rotation_amount - coarse_rotation_amount;
    if (fine_rotation_amount < 0) fine_rotation_amount += m;
    //If the whole warp is rotating by 0, early exit
    uint32_t warp_vote = __ballot(fine_rotation_amount > 0);
    if (warp_vote > 0) {
      uint32_t row = threadIdx.y;
      uint32_t idx = row * n + col;
      T *read_ptr = d + idx;

      uint32_t smem_idx = threadIdx.y * 32 + threadIdx.x;

      T first = -2;
      if (row < m) {
        first = *read_ptr;
      }

      bool first_phase = (threadIdx.y >= fine_rotation_amount);
      uint32_t smem_row = threadIdx.y - fine_rotation_amount;
      if (!first_phase) {
        smem_row += 32;
      }

      uint32_t smem_write_idx = smem_row * 32 + threadIdx.x;

      if (first_phase) {
        smem[smem_write_idx] = first;
      }

      T *write_ptr = read_ptr;
      uint32_t ptr_inc = 32 * n;
      read_ptr += ptr_inc;
      //Loop over blocks that are guaranteed not to fall off the edge
      for (uint32_t i = 0; i < (m / 32) - 1; i++) {
        T tmp = *read_ptr;
        if (!first_phase) {
          smem[smem_write_idx] = tmp;
        }
        __syncthreads();
        *write_ptr = smem[smem_idx];
        __syncthreads();
        if (first_phase) {
          smem[smem_write_idx] = tmp;
        }
        write_ptr = read_ptr;
        read_ptr += ptr_inc;
      }

      //Final block (read_ptr may have fallen off the edge)
      uint32_t remainder = m % 32;
      T tmp = -3;
      if (threadIdx.y < remainder) {
        tmp = *read_ptr;
      }
      uint32_t tmp_dest_row = 32 - fine_rotation_amount + threadIdx.y;
      if ((tmp_dest_row >= 0) && (tmp_dest_row < 32)) {
        smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
      }
      __syncthreads();
      uint32_t
          first_dest_row = 32 + remainder - fine_rotation_amount + threadIdx.y;
      if ((first_dest_row >= 0) && (first_dest_row < 32)) {
        smem[first_dest_row * 32 + threadIdx.x] = first;
      }

      __syncthreads();
      *write_ptr = smem[smem_idx];
      write_ptr = read_ptr;
      __syncthreads();
      //Final incomplete block
      tmp_dest_row -= 32;
      first_dest_row -= 32;
      if ((tmp_dest_row >= 0) && (tmp_dest_row < 32)) {
        smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
      }
      __syncthreads();
      if ((first_dest_row >= 0) && (first_dest_row < 32)) {
        smem[first_dest_row * 32 + threadIdx.x] = first;
      }
      __syncthreads();
      if (threadIdx.y < remainder) {
        *write_ptr = smem[smem_idx];
      }
    }
  }
}

template<typename F>
static void rotate(F fn, uint32_t height, uint32_t width, float *data) {
  uint32_t n_blocks = div_up(width, 32);
  dim3 block_dim(32, 32);
  if (fn.fine()) {
    fine_col_rotate<<<n_blocks, block_dim>>>(fn, height, width, data);
  }
  coarse_col_rotate<<<n_blocks, dim3(32, 16)>>>(fn, height, width, data);
}
