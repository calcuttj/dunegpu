#ifndef ADD_H
#define ADD_H
#include "cuda_runtime.h"
#include "curand_kernel.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include "SimEnergyDepCuda.cuh"
#include "thrust/pair.h"
#include "thrust/tuple.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/transform.h"
#include "BTRHelper.cuh"

namespace gpu {
  void wrapper(int n_op,
               size_t n_deps,
               SimEnergyDepCuda * deps,
               std::pair<int, int> * det_phots,
               float * visibility,
               curandState * state);
  void setup_rng_wrapper(curandState * state);

  struct no_photons {
    no_photons() {}

    __host__ __device__
    bool operator()(const BTRHelper & btrh) {
      return ((btrh.nPhotFast + btrh.nPhotSlow) == 0);
    }
  };


  class is_visible {
    private:
      float * fVisibilityTable;
      int fNOpDets;
      int fOpID;
    public:
      is_visible(float * vis_table, int n_opdets, int op_id)
        : fVisibilityTable(vis_table),
          fNOpDets(n_opdets),
          fOpID(op_id) {}

      __host__ __device__
      bool operator()(SimEnergyDepCuda & dep) {
        return (fVisibilityTable[dep.VoxelID()*fNOpDets + fOpID] > 1.e-9);
      }
  };

  void run_vis_functor(float * vis_table, int n_op_dets,
                       double fast_time, double slow_time, //TODO -- remove these
                       int op_id, std::vector<SimEnergyDepCuda> & deps,
                       std::vector<BTRHelper> & btrs);

  class vis_functor {
    private:
      float * fVisibilityTable;
      int fNOpDets;
      double fFastDecayTime, fSlowDecayTime;//TODO -- remove these
      int fOpID;
    public:
      vis_functor(float * vis_table, int n_opdets,
                  double fast_time, double slow_time, //TODO -- remove these
                  int op_id)
        : fVisibilityTable(vis_table),
          fNOpDets(n_opdets),
          fFastDecayTime(fast_time),//TODO -- remove these
          fSlowDecayTime(slow_time),//TODO -- remove these
          fOpID(op_id) {}

      using DepRNG = thrust::tuple<SimEnergyDepCuda &, curandState &>;
      __device__
      BTRHelper operator()(DepRNG input) {

        auto & dep = thrust::get<0>(input);
        auto & rng = thrust::get<1>(input);
        int vox_id = dep.VoxelID();
        float vis = fVisibilityTable[fNOpDets*vox_id + fOpID];

        int n_fast = 0;
        int n_slow = 0;
        if (vis > 1.e-9) {
          n_fast = curand_poisson(&rng, dep.NumFPhotons()*vis);
          n_slow = curand_poisson(&rng, dep.NumSPhotons()*vis);

        }

        BTRHelper result(
           n_fast,
           n_slow,
           dep.TrackID(),
           dep.MidPointX(),
           dep.MidPointY(),
           dep.MidPointZ(),
           dep.StartT(),
           dep.Energy()
        );
        return result;
      }
  };

  struct curand_setup {
    using init_tuple = thrust::tuple<int, curandState &>;
    __device__
    void operator()(init_tuple t) {
      curandState s;
      int id = thrust::get<0>(t);
      curand_init(1234, id, 0, &s);
      thrust::get<1>(t) = s;
    }
  };

};
#endif
