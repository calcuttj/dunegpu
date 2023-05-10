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
#include "art/Utilities/make_tool.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "fhiclcpp/ParameterSet.h"

#include "larsim/PhotonPropagation/PhotonVisibilityService.h"
#include "larcore/Geometry/Geometry.h"
#include "larsim/PhotonPropagation/IPhotonLibrary.h"

#include "add.cuh"

#include "TFile.h"
#include "TTree.h"
//#include <ROOT/TTreeProcessorMT.hxx>
//#include "TTreeReader.h"

#include <cmath>

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

  void beginJob() override;
  void endJob() override;

  // Required functions.
  void analyze(art::Event const& e) override;

private:

  // Declare member data here.
  art::ServiceHandle<phot::PhotonVisibilityService const> fPVS;
  TFile * fLibraryFile;
  TTree * fLibraryTree;
  std::string fLibraryFileName;
  std::string fSimulationTag;
  size_t fNThreads;

  std::unique_ptr<phot::IPhotonMappingTransformations> fMapping;

  float * fVisibilityTable;
};


GPUAnalyzer::GPUAnalyzer(fhicl::ParameterSet const& p)
  : EDAnalyzer{p},
    fPVS(art::ServiceHandle<phot::PhotonVisibilityService const>()),
    fLibraryFileName(p.get<std::string>("LibraryFileName")),
    fSimulationTag(p.get<std::string>("SimulationTag")),
    fNThreads(p.get<size_t>("NThreads", 1)) {
  // Call appropriate consumes<>() for any products to be retrieved by this module.
    fhicl::ParameterSet mapDefaultSet;
    mapDefaultSet.put("tool_type", "PhotonMappingIdentityTransformations");
    fMapping = art::make_tool<phot::IPhotonMappingTransformations>(
      p.get<fhicl::ParameterSet>("Mapping", mapDefaultSet));

}

void GPUAnalyzer::beginJob() {
  //ROOT::EnableImplicitMT(fNThreads);
  std::cout << "Loading library" << std::endl;
  fLibraryFile = TFile::Open(fLibraryFileName.c_str());
  //TODO -- Wrap in exception
  std::cout << "File: " << fLibraryFile << std::endl;
  fLibraryTree = (TTree*)fLibraryFile->Get("PhotonLibraryData");
  std::cout << "Tree: " << fLibraryTree << std::endl;

  art::ServiceHandle<geo::Geometry const> geom;
  int NOpDets = geom->NOpDets();
  size_t NVoxels = fPVS->GetVoxelDef().GetNVoxels();

  size_t n_entries = NOpDets*NVoxels;
  cudaMallocManaged(&fVisibilityTable, n_entries*sizeof(float));
  int Voxel;
  int OpChannel;
  float Visibility;

  fLibraryTree->SetBranchAddress("Voxel", &Voxel);
  fLibraryTree->SetBranchAddress("OpChannel", &OpChannel);
  fLibraryTree->SetBranchAddress("Visibility", &Visibility);
  for (int i = 0; i < fLibraryTree->GetEntries(); ++i) {
    fLibraryTree->GetEntry(i);
    int index = Voxel*NOpDets + OpChannel;
    //std::cout << Voxel << " " << NOpDets << " " << OpChannel << std::endl;
    //std::cout << "Got entry " << i << " " << index << " " << n_entries << std::endl;
    fVisibilityTable[index] = Visibility;
  }
  std::cout << "Loaded library" << std::endl;

  /*
  int n_split = ceil(n_entries/fNThreads);
  std::vector<TEntryList> split_entries(fNThreads);
  size_t start = 0;
  for (size_t i = 0; i < fNThreads; ++i) {
    size_t max = ( i != fNThreads-1 ? n_split : n_entries % fNThreads);
    for (size_t j = 0; j < max; ++j) {
      split_entries.back().Enter(start + j);
    }

    start += max;
  }

  auto myFunction = [&](TTreeReader &myReader) {
    TTreeReaderValue<int> VoxelRV(myReader, "Voxel");
    TTreeReaderValue<int> OpChannelRV(myReader, "OpChannel");
    TTreeReaderValue<float> VisibilityRV(myReader, "Visibility");

    std::cout << "Starting thread" << std::endl;
    while (myReader.Next()) {
      auto voxel = *VoxelRV;
      auto opchan = *OpChannelRV;
      auto visibility = *VisibilityRV;


      //std::lock_guard<std::mutex> guard(themutex);
      fVisibilityTable[voxel*NOpDets + opchan] = visibility;
    }
  }

  std::vector<ROOT::TTreeProcessorMT> processors;
  for (size_t i = 0; i < fNThreads; ++i) {
    processors.push_back(ROOT::TTreeProcessorMT(*fLibraryTree, split_entries[i]));
  }
  */
}

void GPUAnalyzer::endJob() {
  fLibraryFile->Close();
}


void GPUAnalyzer::analyze(art::Event const& e)
{
  // Implementation of required member function here.
  auto allDeps = e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(fSimulationTag);
  size_t ndeps = allDeps->size();

  std::vector<SimEnergyDepCuda> cuda_deps;
  for (size_t i = 0; i < ndeps; ++i) {
    cuda_deps.push_back(SimEnergyDepCuda((*allDeps)[i]));
    int voxel_id = fPVS->GetVoxelDef().GetVoxelID(
        fMapping->detectorToLibrary((*allDeps)[i].MidPoint()));
    cuda_deps[i].SetVoxelID(voxel_id);
  }

  /*SimEnergyDepCuda * cuda_deps;
  cudaMallocManaged(&cuda_deps, ndeps*sizeof(SimEnergyDepCuda));
  for (size_t i = 0; i < ndeps; ++i) {
    cuda_deps[i] = SimEnergyDepCuda((*allDeps)[i]);
    //geo::Point_t p = {dep.MidPointX(), dep.MidPointY(), dep.MidPointZ()};
    int voxel_id = fPVS->GetVoxelDef().GetVoxelID(
        fMapping->detectorToLibrary((*allDeps)[i].MidPoint()));
    cuda_deps[i].SetVoxelID(voxel_id);
  }*/

  curandState * rng;
  cudaMallocManaged((void **)&rng, 256*sizeof(curandState));
  gpu::setup_rng_wrapper(rng);

  std::cout << "Calling wrapper" << std::endl;
  art::ServiceHandle<geo::Geometry const> geom;
  size_t NOpDets = geom->NOpDets();
  printf("%zu Op Dets\n", NOpDets);

  int op_id = 0;
  //thrust::transform(cuda_deps_dev.begin(), cuda_deps_dev.end(), results.begin(),
  //                  gpu::vis_functor(fVisibilityTable, NOpDets, op_id));
  gpu::run_vis_functor(fVisibilityTable, NOpDets, op_id, cuda_deps);
  std::cout << "Done" << std::endl;
  //std::cout << results.size() << " " << results[0] << std::endl;
  

  /*
  std::pair<int, int> * det_phots;
  cudaMallocManaged(&det_phots, ndeps*NOpDets*sizeof(std::pair<int,int>));

  gpu::wrapper(NOpDets, ndeps, cuda_deps, det_phots, fVisibilityTable, rng);
  */
}

DEFINE_ART_MODULE(GPUAnalyzer)
