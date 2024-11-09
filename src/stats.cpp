#include "stats.hpp"
#include <vector>
#include <numeric>
#include <cmath>

double average(const std::vector<double>& data)
{
  return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double sample_standard_deviation(const std::vector<double>& data)
{
  double mean = average(data);

  double sum_sqr_diffs = 0.0;
  for (const auto& x : data) {
    sum_sqr_diffs += (x - mean) * (x - mean);
  }

  double variance = sum_sqr_diffs / (data.size() - 1);
  return std::sqrt(variance);
}
