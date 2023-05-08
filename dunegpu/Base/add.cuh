#ifndef ADD_H
#define ADD_H
#include "cuda_runtime.h"
#include "curand_kernel.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include "SimEnergyDepCuda.cuh"

namespace gpu {
  void wrapper(int n_op, SimEnergyDepCuda * deps, float * visibility,
               curandState * state);
  void setup_rng_wrapper(curandState * state);
};
#endif
