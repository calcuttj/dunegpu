////////////////////////////////////////////////////////////////////////
// Class:       GPUAnalyzer
// Plugin Type: analyzer (Unknown Unknown)
// File:        GPUAnalyzer_module.cc
//
// Generated at Fri May  5 13:11:11 2023 by Jacob Calcutt using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larsim/PhotonPropagation/PhotonVisibilityService.h"

#include "add.cuh"

class GPUAnalyzer;


class GPUAnalyzer : public art::EDAnalyzer {
public:
  explicit GPUAnalyzer(fhicl::ParameterSet const& p);
  // The compiler-generated destructor is fine for non-base
  // classes without bare pointers or other resource use.

  // Plugins should not be copied or assigned.
  GPUAnalyzer(GPUAnalyzer const&) = delete;
  GPUAnalyzer(GPUAnalyzer&&) = delete;
  GPUAnalyzer& operator=(GPUAnalyzer const&) = delete;
  GPUAnalyzer& operator=(GPUAnalyzer&&) = delete;

  // Required functions.
  void analyze(art::Event const& e) override;

private:

  // Declare member data here.
  art::ServiceHandle<phot::PhotonVisibilityService const> fPVS;
  std::string fSimulationTag;

};


GPUAnalyzer::GPUAnalyzer(fhicl::ParameterSet const& p)
  : EDAnalyzer{p},
    fPVS(art::ServiceHandle<phot::PhotonVisibilityService const>()),
    fSimulationTag(p.get<std::string>("SimulationTag")) {
  // Call appropriate consumes<>() for any products to be retrieved by this module.
}

void GPUAnalyzer::analyze(art::Event const& e)
{
  std::cout << e.id().event() << std::endl;
  // Implementation of required member function here.


  auto allDeps = e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(fSimulationTag);
  size_t ndeps = allDeps->size();

  SimEnergyDepCuda * cuda_deps;
  cudaMallocManaged(&cuda_deps, ndeps*sizeof(SimEnergyDepCuda));
  for (size_t i = 0; i < ndeps; ++i) {
    cuda_deps[i] = SimEnergyDepCuda((*allDeps)[i]);
  }

  std::cout << "Calling wrapper" << std::endl;
  gpu::wrapper(cuda_deps); 
}

DEFINE_ART_MODULE(GPUAnalyzer)
