#pragma once

struct column_major_index {
  uint32_t height;
  uint32_t width;

  __host__ __device__
  column_major_index(const uint32_t &_height, const uint32_t &_width) :
      height(_height), width(_width) {}

  __host__ __device__
  uint64_t operator()(const uint32_t &i, const uint32_t &j) const {
    return i + static_cast<uint64_t>(j) * static_cast<uint64_t>(height);
  }
};

struct row_major_index {
  uint32_t height;
  uint32_t width;

  __host__ __device__
  row_major_index(const uint32_t &_height, const uint32_t &_width) :
      height(_height), width(_width) {}

  __host__ __device__
  row_major_index(const reduced_divisor &_height, const uint32_t &_width) :
      height(_height.get()), width(_width) {}

  __host__ __device__
  uint64_t operator()(const uint32_t &i, const uint32_t &j) const {
    return j + static_cast<uint64_t>(i) * static_cast<uint64_t>(width);
  }
};
