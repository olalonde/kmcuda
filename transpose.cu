#include "inplace/gcd.h"
#include "inplace/rotate.h"
#include "inplace/permute.h"
#include "inplace/skinny.h"
#include "inplace/register_ops.h"

#include "private.h"

template<typename F>
void shuffle(float* data, uint32_t height, uint32_t width, F s) {
  #if CUDA_ARCH >= 50
  if (width < 12288) {
    smem_row_shuffle<<<height, 256, sizeof(float) * width>>>(height, width, data, s);
  } else if (width < 30720) {
    register_row_shuffle<F, 60><<<height, 512>>>(height, width, data, s);
  }
  #else
  if (width < 6144) {
    smem_row_shuffle<<<height, 256, sizeof(float) * width>>>(height, width, data, s);
  } else if (width < 11326) {
    register_row_shuffle<F, 31><<<height, 512>>>(height, width, data, s);
  } else if (n < 30720) {
    register_row_shuffle<F, 60><<<height, 512>>>(height, width, data, s);
  }
  #endif
  else {
    assert(false && "width must be < 30720");
  }
}

namespace c2r {

void transpose(bool row_major, uint32_t height, uint32_t width, float* data,
               uint32_t *buffer) {
  if (!row_major) {
    uint32_t tmp = height;
    height = width;
    width = tmp;
  }
  uint32_t c, t, k;
  extended_gcd(height, width, c, t);
  if (c > 1) {
    extended_gcd(height / c, width / c, t, k);
  } else {
    k = t;
  }
  if (c > 1) {
    rotate(prerotator(width / c), height, width, data);
  }
  shuffle(data, height, width, shuffler(height, width, c, k));
  rotate(postrotator(height), height, width, data);
  scatter_permute(scatter_postpermuter(m, n, c), m, n, data, buffer);
}

}

namespace r2c {

void transpose(bool row_major, uint32_t height, uint32_t width, float* data,
               uint32_t *buffer) {
  if (row_major) {
    uint32_t tmp = height;
    height = width;
    width = tmp;
  }
  uint32_t c, t, k;
  extended_gcd(height, width, c, t);
  if (c > 1) {
    extended_gcd(height / c, n / c, t, k);
  } else {
    k = t;
  }
  scatter_permute(scatter_prepermuter(height, width, c), height, width, data, buffer);
  rotate(prerotator(height), height, width, data);
  shuffle(data, height, width, shuffler(height, width, c, k));
  if (c > 1) {
    rotate(postrotator(width / c, height), height, width, data);
  }
}

}

__global__ void copy_sample_t(
    uint32_t index, uint32_t samples_size, uint16_t features_size,
    const float *__restrict__ samples, float *__restrict__ dest) {
  uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= features_size) {
    return;
  }
  dest[ti] = samples[static_cast<uint64_t>(samples_size) * static_cast<uint64_t>(ti) + index];
}

static KMCUDAResult inplace_transpose(
    bool row_major, uint32_t height, uint32_t width, float *data,
    float *buffer) {
  bool small_height = height < 32;
  bool small_width = width < 32;
  //Heuristic to choose the fastest implementation
  //based on size of matrix and data layout
  if (!small_height && small_width) {
    if (!row_major) {
      return c2r::skinny_transpose(width, height, data, buffer);
    } else {
      return r2c::skinny_transpose(width, height, data, buffer);
    }
  } else if (small_height) {
    if (!row_major) {
      return r2c::skinny_transpose(height, width, data, buffer);
    } else {
      return c2r::skinny_transpose(height, width, data, buffer);
    }
  } else if ((height > width) ^ row_major) {
    return r2c::transpose(row_major, data, height, width);
  } else {
    return c2r::transpose(row_major, data, height, width);
  }
}

extern "C" {

KMCUDAResult cuda_copy_sample_t(
    uint32_t index, uint32_t offset, uint32_t samples_size, uint16_t features_size,
    const std::vector<int> &devs, int verbosity, const udevptrs<float> &samples,
    udevptrs<float> *dest) {
  FOR_EACH_DEVI(
    dim3 block(min(1024, features_size), 1, 1);
    dim3 grid(upper(static_cast<unsigned>(features_size), block.x), 1, 1);
    copy_sample_t<<<grid, block>>>(
        index, samples_size, features_size, samples[devi].get(),
        (*dest)[devi].get() + offset);
  );
  return kmcudaSuccess;
}

KMCUDAResult cuda_inplace_transpose(
    uint32_t samples_size, uint16_t features_size, bool forward,
    const std::vector<int> &devs, int verbosity, udevptrs<float> *samples,
    udevptrs<uint32_t> *buffer) {
  INFO("transposing the samples inplace...\n");
  FOR_EACH_DEVI(
    inplace_transpose(forward, samples_size, features_size,
                      (*samples)[devi].get(), (*buffer)[devi].get());
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

}  // extern "C"