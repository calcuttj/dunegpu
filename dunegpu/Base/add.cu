#include "add.cuh"

namespace gpu {
__global__ void add(SimEnergyDepCuda * deps) {
  printf("%f\n", deps[0].Energy());
  return;
}

  void wrapper(SimEnergyDepCuda * deps) {
    add<<<1, 1>>>(deps);
  }
}


//void gpu::add(int n, float *x, float *y) {
//  for (int i = 0; i < n; i++)
//      y[i] = x[i] + y[i];
//}
