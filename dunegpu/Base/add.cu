#include "thrust/tuple.h"
#include "thrust/copy.h"
#include "thrust/remove.h"
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

  void run_vis_functor(float * vis_table, int n_op_dets,
                       double fast_time, double slow_time,
                       int op_id,
                       std::vector<SimEnergyDepCuda> & deps,
                       std::vector<BTRHelper> & btrs) {




    //make device deps vector
    thrust::device_vector<SimEnergyDepCuda> deps_dev(deps.begin(), deps.end());

    //Make rngs equal to deps vec size
    thrust::device_vector<curandState> rng_vec(deps_dev.size());

    auto pInit = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::counting_iterator<int>(0), rng_vec.begin()));
    // initialize random generator
    thrust::for_each_n(pInit, rng_vec.size(), curand_setup());

    auto deps_rngs_start = thrust::make_zip_iterator(
        thrust::make_tuple(deps_dev.begin(), rng_vec.begin()));

    auto deps_rngs_end = thrust::make_zip_iterator(
        thrust::make_tuple(deps_dev.end(), rng_vec.end()));

    //Make results vector
    thrust::device_vector<BTRHelper> results(deps.size());

    //Do transformation
    thrust::transform(deps_rngs_start, deps_rngs_end, results.begin(),
                      vis_functor(vis_table, n_op_dets,
                                  fast_time, slow_time, op_id));


    //std::cout << ((BTRHelper)results[0]).nPhotFast << " " <<
    //             ((BTRHelper)results[0]).nPhotSlow << " " <<
    //             ((BTRHelper)results[0]).trackID << std::endl;

    //std::cout << "N results: " << results.size() << std::endl;
    auto removed_end = thrust::remove_if(results.begin(), results.end(), no_photons());
    
    thrust::host_vector<BTRHelper> host_results(results.begin(), removed_end);

    btrs.insert(btrs.begin(), host_results.begin(), host_results.end());
  }
}


//void gpu::add(int n, float *x, float *y) {
//  for (int i = 0; i < n; i++)
//      y[i] = x[i] + y[i];
//}
