#include "add.cuh"

namespace gpu {
  __global__ void setup_kernel(curandState * state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
  }

  __global__ void add(int n_op,
                      SimEnergyDepCuda * deps,
                      float * visibility,
                      curandState * state) {
  int op_det = threadIdx.x;
  if (op_det < n_op) {
    curandState localState = state[op_det];
    printf("Inside kernel\n");
    printf("Op Channel %i\n", op_det);
    int vid = deps[0].VoxelID();
    float vis = visibility[vid*n_op + op_det];
    if (vis > 1.e-9) {
      printf("Vis %f\n", visibility[vid*n_op + op_det]);
      float lf = deps[0].NumFPhotons()*vis;
      float ls = deps[0].NumSPhotons()*vis;
      int nphot_fast = curand_poisson(&localState, lf);
      int nphot_slow = curand_poisson(&localState, ls);
      printf("L Fast, slow: %f, %f\n", lf, ls);
      printf("N Fast, slow: %i, %i\n", nphot_fast, nphot_slow);
    }
  }



  return;
}

  void wrapper(int n_op, SimEnergyDepCuda * deps, float * vis,
               curandState * state) {
    add<<<1, 256>>>(n_op, deps, vis, state);
  }

  void setup_rng_wrapper(curandState * state) {
    std::cout << "Setup rng" << std::endl;
    setup_kernel<<<1, 256>>>(state);
  }
}


//void gpu::add(int n, float *x, float *y) {
//  for (int i = 0; i < n; i++)
//      y[i] = x[i] + y[i];
//}
