#include "thrust/tuple.h"
#include "add.cuh"

namespace gpu {
  __global__ void setup_kernel(curandState * state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
  }

  __global__ void add(int n_op,
                      size_t n_deps,
                      SimEnergyDepCuda * deps,
                      std::pair<int, int> * det_phots,
                      float * visibility,
                      curandState * state) {
    int op_det = threadIdx.x;
    if (op_det < n_op) {
      curandState localState = state[op_det];
      printf("Inside kernel\n");
      printf("Op Channel %i\n", op_det);

      for (size_t i = 0; i < n_deps; ++i) {
        int vid = deps[i].VoxelID();
        float vis = visibility[vid*n_op + op_det];
        if (vis > 1.e-9) {
          //printf("Vis %f\n", visibility[vid*n_op + op_det]);
          float lf = deps[0].NumFPhotons()*vis;
          float ls = deps[0].NumSPhotons()*vis;
          int nphot_fast = curand_poisson(&localState, lf);
          int nphot_slow = curand_poisson(&localState, ls);
          det_phots[i*n_op + op_det].first = nphot_fast;
          det_phots[i*n_op + op_det].second = nphot_slow;
          if ((nphot_fast + nphot_slow) > 0)
            printf("N Fast, slow: %i, %i\n", nphot_fast, nphot_slow);
          //printf("L Fast, slow: %f, %f\n", lf, ls);
          //printf("N Fast, slow: %i, %i\n", nphot_fast, nphot_slow);
        }

      }
    }

    return;
  }

  void wrapper(int n_op,
               size_t n_deps,
               SimEnergyDepCuda * deps,
               std::pair<int, int> * det_phots,
               float * vis,
               curandState * state) {
    add<<<1, 256>>>(n_op, n_deps, deps, det_phots, vis, state);
  }

  void setup_rng_wrapper(curandState * state) {
    std::cout << "Setup rng" << std::endl;
    setup_kernel<<<1, 256>>>(state);
  }

  void run_vis_functor(float * vis_table, int n_op_dets, int op_id,
                       std::vector<SimEnergyDepCuda> & deps) {
    std::cout << "Running vis functor" << std::endl;
    thrust::host_vector<SimEnergyDepCuda> deps_host(deps.begin(), deps.end());
    std::cout << "Made Deps host " << deps_host.size() << std::endl;

    thrust::device_vector<SimEnergyDepCuda> deps_dev(deps_host.begin(),
                                                     deps_host.end());
    std::cout << "Made Deps device " << deps_dev.size() << std::endl;
    thrust::device_vector<BTRHelper> results(deps.size());
    std::cout << "Made Results " << results.size() << std::endl;
    thrust::transform_if(deps_dev.begin(), deps_dev.end(), results.begin(),
                         vis_functor(vis_table, n_op_dets, op_id),
                         is_visible(vis_table, n_op_dets, op_id));
    printf("Results size: %zu\n", results.size());
    std::cout << ((BTRHelper)results[0]).nPhotFast << " " <<
                 ((BTRHelper)results[0]).nPhotSlow << " " <<
                 ((BTRHelper)results[0]).trackID << std::endl;
  }
}


//void gpu::add(int n, float *x, float *y) {
//  for (int i = 0; i < n; i++)
//      y[i] = x[i] + y[i];
//}
