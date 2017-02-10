#pragma once

#include "index.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"

namespace c2r {

struct fused_preop {
  reduced_divisor m;
  reduced_divisor b;
  __host__ fused_preop(uint32_t _m, uint32_t _b) : m(_m), b(_b) {}
  __host__ __device__
  uint32_t operator()(const uint32_t &i, const uint32_t &j) {
    return (uint32_t) m.mod(i + (uint32_t) b.div(j));
  }
};

// This shuffler exists for cases where m, n are large enough to cause overflow
struct long_shuffle {
  uint32_t m, n, k;
  reduced_divisor_64 b;
  reduced_divisor c;
  __host__
  long_shuffle(uint32_t _m, uint32_t _n, uint32_t _c, uint32_t _k) :
      m(_m), n(_n), k(_k), b(_n / _c), c(_c) {}
  uint32_t i;
  __host__ __device__
  void set_i(const uint32_t &_i) {
    i = _i;
  }
  __host__ __device__
  uint32_t f(const uint32_t &j) {
    uint32_t r = j + i * (n - 1);
    if (i - static_cast<int64_t>(c.mod(j)) <= m - static_cast<int64_t>(c.get())) {
      return r;
    } else {
      return r + m;
    }
  }

  __host__ __device__
  uint32_t operator()(const uint32_t &j) {
    uint32_t fij = f(j);
    uint32_t fijdivc, fijmodc;
    c.divmod(fij, fijdivc, fijmodc);
    uint32_t term_1 = b.mod(static_cast<int64_t>(k) * static_cast<int64_t>(fijdivc));
    uint32_t term_2 = fijmodc * b.get();
    return term_1 + term_2;
  }
};

struct fused_postop {
  reduced_divisor m;
  uint32_t n, c;
  __host__
  fused_postop(uint32_t _m, uint32_t _n, uint32_t _c) : m(_m), n(_n), c(_c) {}
  __host__ __device__
  uint32_t operator()(const uint32_t &i, const uint32_t &j) {
    return m.mod(i * n - m.div(i * c) + j);
  }
};

}

namespace r2c {

struct fused_preop {
  reduced_divisor a;
  reduced_divisor c;
  reduced_divisor m;
  uint32_t q;
  __host__
  fused_preop(uint32_t _a, uint32_t _c, uint32_t _m, uint32_t _q)
      : a(_a), c(_c), m(_m), q(_q) {}
  __host__ __device__ __forceinline__
  uint32_t p(const uint32_t &i) {
    uint32_t cm1 = c.get() - 1;
    uint32_t term_1 = a.get() * c.mod(cm1 * i);
    uint32_t term_2 = a.mod(c.div(cm1 + i) * q);
    return term_1 + term_2;

  }
  __host__ __device__
  uint32_t operator()(const uint32_t &i, const uint32_t &j) {
    uint32_t idx = m.mod(i + m.get() - m.mod(j));
    return p(idx);
  }
};

struct fused_postop {
  reduced_divisor m;
  reduced_divisor b;
  __host__ fused_postop(uint32_t _m, uint32_t _b) : m(_m), b(_b) {}
  __host__ __device__
  uint32_t operator()(const uint32_t &i, const uint32_t &j) {
    return m.mod(i + m.get() - b.div(j));
  }
};

}

template<typename F, uint32_t U>
__global__ void long_row_shuffle(
    F s, uint32_t height, uint32_t width, uint32_t i, float *d, float *tmp) {
  row_major_index rm(height, width);
  s.set_i(i);
  uint32_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t grid_size = gridDim.x * blockDim.x;
  uint32_t j = global_id;
  while (j + U * grid_size < width) {
    #pragma unroll
    for (uint32_t k = 0; k < U; k++) {
      tmp[j] = d[rm(i, s(j))];
      j += grid_size;
    }
  }
  while (j < width) {
    tmp[j] = d[rm(i, s(j))];
    j += grid_size;
  }
}

template<typename F>
__global__ void short_column_permute(F s, uint32_t height, uint32_t width, float *d) {
  extern __shared__ float *smem;
  row_major_index rm(height, width);
  row_major_index blk(blockDim.y, blockDim.x);
  uint32_t i = threadIdx.y; // One block tall by REQUIREMENT
  uint32_t grid_size = blockDim.x * gridDim.x;

  if (i < height) {
    for (uint32_t j = threadIdx.x + blockIdx.x * blockDim.x;
         j < width; j += grid_size) {
      smem[blk(i, threadIdx.x)] = d[rm(i, j)];
      __syncthreads();
      d[rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
      __syncthreads();
    }
  }
}

template<typename F>
void skinny_row_op(F s, uint32_t height, uint32_t width, float *d, float *tmp) {
  for (uint32_t i = 0; i < m; i++) {
    long_row_shuffle<F, 4><<<(width - 1) / (256 * 4) + 1, 256>>>(
        height, width, i, d, tmp, s);
    cudaMemcpy(d + width * i, tmp, sizeof(float) * width, cudaMemcpyDeviceToDevice);
  }
}

template<typename F>
void skinny_col_op(F s, uint32_t height, uint32_t width, float *d) {
  uint32_t n_threads = 32;
  uint32_t n_blocks = n_sms() * 8;
  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads, height);
  short_column_permute<<<grid_dim, block_dim,
      sizeof(float) * height * n_threads>>>(height, width, d, s);
}

namespace c2r {

static void skinny_transpose(uint32_t height, uint32_t width, float *data,
                             float *buffer) {
  assert(height <= 32);
  uint32_t c, t, k;
  extended_gcd(height, width, c, t);
  if (c > 1) {
    extended_gcd(height / c, width / c, t, k);
  } else {
    k = t;
  }
  if (c > 1) {
    skinny_col_op(fused_preop(height, width / c), height, width, data);
  }
  skinny_row_op(long_shuffle(height, width, c, k), height, width, data, buffer);
  skinny_col_op(fused_postop(height, width, c), height, width, data);
}

}

namespace r2c {

static void skinny_transpose(uint32_t height, uint32_t width, float *data,
                             float *buffer) {
  assert(height <= 32);
  uint32_t c, t, q;
  extended_gcd(width, height, c, t);
  if (c > 1) {
    extended_gcd(width / c, height / c, t, q);
  } else {
    q = t;
  }
  skinny_col_op(fused_preop(height / c, c, height, q), height, width, data);
  skinny_row_op(shuffler(height, width, c, 0), height, width, data, buffer);
  if (c > 1) {
    skinny_col_op(fused_postop(height, width / c), height, width, data);
  }
}

}
