#ifndef ADD_H
#define ADD_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "SimEnergyDepCuda.cuh"

namespace gpu {
  void wrapper(SimEnergyDepCuda * deps);
};
#endif
