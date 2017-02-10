#ifndef KMCUDA_PRIVATE_H
#define KMCUDA_PRIVATE_H

#include "kmcuda.h"
#include <cinttypes>
#include <tuple>
#include "wrappers.h"
#include "macros.h"

template <typename T>
inline T upper(T size, T each) {
  T div = size / each;
  if (div * each == size) {
    return div;
  }
  return div + 1;
}

using plan_t = std::vector<std::tuple<uint32_t, uint32_t>>;

/// @brief (offset, size) pairs.
inline plan_t distribute(
    uint32_t amount, uint32_t size_each, const std::vector<int> &devs) {
  if (devs.size() == 0) {
    return {};
  }
  if (devs.size() == 1) {
    return {std::make_tuple(0, amount)};
  }
  const uint32_t alignment = 512;
  uint32_t a = size_each, b = alignment, gcd = 0;
  for (;;) {
    if (a == 0) {
      gcd = b;
      break;
    }
    b %= a;
    if (b == 0) {
      gcd = a;
      break;
    }
    a %= b;
  }
  uint32_t stride = alignment / gcd;
  uint32_t offset = 0;
  std::vector<std::tuple<uint32_t, uint32_t>> res;
  for (size_t i = 0; i < devs.size() - 1; i++) {
    float step = (amount - offset + .0f) / (devs.size() - i);
    uint32_t len = roundf(step / stride) * stride;
    res.emplace_back(offset, len);
    offset += len;
  }
  res.emplace_back(offset, amount - offset);
  return res;
}

inline uint32_t max_distribute_length(
    uint32_t amount, uint32_t size_each, const std::vector<int> &devs) {
  auto plan = distribute(amount, size_each, devs);
  uint32_t max_length = 0;
  for (auto& p : plan) {
    uint32_t length = std::get<1>(p);
    if (length > max_length) {
      max_length = length;
    }
  }
  return max_length;
}

inline void print_plan(
    const char *name, const std::vector<std::tuple<uint32_t, uint32_t>>& plan) {
  printf("%s: [", name);
  bool first = true;
  for (auto& p : plan) {
    if (!first) {
      printf(", ");
    }
    first = false;
    printf("(%" PRIu32 ", %" PRIu32 ")", std::get<0>(p), std::get<1>(p));
  }
  printf("]\n");
}

extern "C" {

KMCUDAResult cuda_copy_sample_t(
    uint32_t index, uint32_t offset, uint32_t samples_size, uint16_t features_size,
    const std::vector<int> &devs, int verbosity, const udevptrs<float> &samples,
    udevptrs<float> *dest);

KMCUDAResult cuda_inplace_transpose(
    uint32_t samples_size, uint16_t features_size, bool forward,
    const std::vector<int> &devs, int verbosity, udevptrs<float> *samples,
    udevptrs<uint32_t> *buffer);

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t samples_size, uint32_t features_size, uint32_t cc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<float> *dists, float *host_dists, float *dists_sum);

KMCUDAResult kmeans_cuda_afkmc2_calc_q(
    uint32_t samples_size, uint32_t features_size, uint32_t firstc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *d_q,
    float *h_q);

KMCUDAResult kmeans_cuda_afkmc2_random_step(
    uint32_t k, uint32_t m, uint64_t seed, int verbosity, const float *q,
    uint32_t *d_choices, uint32_t *h_choices, float *d_samples, float *h_samples);

KMCUDAResult kmeans_cuda_afkmc2_min_dist(
    uint32_t k, uint32_t m, KMCUDADistanceMetric metric, int fp16x2,
    int32_t verbosity, const float *samples, const uint32_t *choices,
    const float *centroids, float *d_min_dists, float *h_min_dists);

KMCUDAResult kmeans_cuda_setup(
    uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
    uint32_t yy_groups_size, const std::vector<int> &devs, int32_t verbosity);

KMCUDAResult kmeans_init_centroids(
    KMCUDAInitMethod method, const void *init_params, uint32_t samples_size,
    uint16_t features_size, uint32_t clusters_size, KMCUDADistanceMetric metric,
    uint32_t seed, const std::vector<int> &devs, int device_ptrs, int fp16x2,
    int32_t verbosity, const float *host_centroids,  const udevptrs<float> &samples,
    udevptrs<float> *dists, udevptrs<float> *aux, udevptrs<float> *centroids);

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t yy_groups_size, uint32_t samples_size,
    uint32_t clusters_size, uint16_t features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, udevptrs<float> *bounds_yy,
    udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy);

KMCUDAResult kmeans_cuda_calc_average_distance(
    uint32_t samples_size, uint16_t features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int32_t verbosity, const udevptrs<float> &samples,
    const udevptrs<float> &centroids, const udevptrs<uint32_t> &assignments,
    float *average_distance);

KMCUDAResult knn_cuda_setup(
    uint32_t samples_size, uint16_t features_size, uint32_t clusters_size,
    const std::vector<int> &devs, int32_t verbosity);

KMCUDAResult knn_cuda_calc(
    uint16_t k, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int verbosity,
    const udevptrs<float> &samples, const udevptrs<float> &centroids,
    const udevptrs<uint32_t> &assignments, const udevptrs<uint32_t> &inv_asses,
    const udevptrs<uint32_t> &inv_asses_offsets, udevptrs<float> *distances,
    udevptrs<float>* sample_dists, udevptrs<float> *radiuses,
    udevptrs<uint32_t> *neighbors);

int knn_cuda_neighbors_mem_multiplier(uint16_t k, int dev, int verbosity);
}  // extern "C"

#endif  // KMCUDA_PRIVATE_H
