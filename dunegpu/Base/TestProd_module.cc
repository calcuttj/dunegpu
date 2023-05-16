////////////////////////////////////////////////////////////////////////
// Class:       TestProd
// Plugin Type: producer (Unknown Unknown)
// File:        TestProd_module.cc
//
// Generated at Mon May 15 20:21:14 2023 by Jacob Calcutt using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "SimEnergyDepCuda.cuh"

#include <memory>

class TestProd;


class TestProd : public art::EDProducer {
public:
  explicit TestProd(fhicl::ParameterSet const& p);
  // The compiler-generated destructor is fine for non-base
  // classes without bare pointers or other resource use.

  // Plugins should not be copied or assigned.
  TestProd(TestProd const&) = delete;
  TestProd(TestProd&&) = delete;
  TestProd& operator=(TestProd const&) = delete;
  TestProd& operator=(TestProd&&) = delete;

  // Required functions.
  void produce(art::Event& e) override;

private:

  // Declare member data here.
  std::string fSimulationTag;

};


TestProd::TestProd(fhicl::ParameterSet const& p)
  : EDProducer{p},
    fSimulationTag(p.get<std::string>("SimulationTag"))
  // More initializers here.
{
  produces<std::vector<SimEnergyDepCuda>>();
  // Call appropriate produces<>() functions here.
  // Call appropriate consumes<>() for any products to be retrieved by this module.
}

void TestProd::produce(art::Event& e)
{
  // Implementation of required member function here.
  auto dep_col = std::make_unique<std::vector<SimEnergyDepCuda>>();
   
  auto allDeps = *(e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(fSimulationTag));

  for (const auto & dep : allDeps) {
    dep_col->emplace_back(SimEnergyDepCuda(dep));
  }

  e.put(move(dep_col));
}

DEFINE_ART_MODULE(TestProd)
