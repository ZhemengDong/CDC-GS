#include "spatial.h"
#include "simple_knn.h"

torch::Tensor distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

  int k = 7;  // 可自由设置
  SimpleKNN knn;
  knn.knn(P, (float3*)points.contiguous().data_ptr<float>(), means.contiguous().data_ptr<float>(), k);

  return means;
}