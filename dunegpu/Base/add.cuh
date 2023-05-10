#ifndef ADD_H
#define ADD_H
#include "cuda_runtime.h"
#include "curand_kernel.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include "SimEnergyDepCuda.cuh"
#include "thrust/pair.h"
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

  void run_vis_functor(float * vis_table, int n_op_dets, int op_id,
                       std::vector<SimEnergyDepCuda> & deps);

  class vis_functor {
    private:
      float * fVisibilityTable;
      int fNOpDets;
      int fOpID;
    public:
      vis_functor(float * vis_table, int n_opdets, int op_id)
        : fVisibilityTable(vis_table),
          fNOpDets(n_opdets),
          fOpID(op_id) {}

      __host__ __device__
      /*thrust::pair<float, float>*/
      BTRHelper operator()(SimEnergyDepCuda & dep) {

        int vox_id = dep.VoxelID();
        float vis = fVisibilityTable[fNOpDets*vox_id + fOpID];

        //auto result = thrust::make_pair<float, float>(dep.NumFPhotons()*vis,
        //                                              dep.NumSPhotons()*vis);
        BTRHelper result(
           (int)dep.NumFPhotons()*vis,
           (int)dep.NumSPhotons()*vis,
           dep.TrackID(),
           dep.MidPointX(),
           dep.MidPointY(),
           dep.MidPointZ(),
           0.,
           0.,
           dep.Energy()
        );
        return result;
      }
  };
};
#endif
