////////////////////////////////////////////////////////////////////////
// Class:       GPUProducer
// Plugin Type: producer (Unknown Unknown)
// File:        GPUProducer_module.cc
//
// Generated at Fri May  5 13:11:11 2023 by Jacob Calcutt using cetskelgen
// from  version .
////////////////////////////////////////////////////////////////////////

#include "art/Framework/Core/EDProducer.h"
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

#include "lardataobj/Simulation/OpDetBacktrackerRecord.h"
#include "lardataobj/Simulation/SimPhotons.h"

#include "add.cuh"

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
//#include <ROOT/TTreeProcessorMT.hxx>
//#include "TTreeReader.h"

#include <cmath>

class GPUProducer;


class GPUProducer : public art::EDProducer {
public:
  explicit GPUProducer(fhicl::ParameterSet const& p);
  // The compiler-generated destructor is fine for non-base
  // classes without bare pointers or other resource use.

  // Plugins should not be copied or assigned.
  GPUProducer(GPUProducer const&) = delete;
  GPUProducer(GPUProducer&&) = delete;
  GPUProducer& operator=(GPUProducer const&) = delete;
  GPUProducer& operator=(GPUProducer&&) = delete;

  void beginJob() override;
  void endJob() override;

  // Required functions.
  void produce(art::Event & e) override;

private:
  void AddOpDetBTR(std::vector<sim::OpDetBacktrackerRecord>& opbtr,
                   std::vector<int>& ChannelMap,
                   const sim::OpDetBacktrackerRecord& btr) const;

  // Declare member data here.
  art::ServiceHandle<phot::PhotonVisibilityService const> fPVS;
  TFile * fLibraryFile;
  TTree * fLibraryTree;
  std::string fLibraryFileName;
  std::string fSimulationTag;
  size_t fNThreads;
  double fFastDecayTime, fSlowDecayTime;
  TRandom3 fRNG;

  std::unique_ptr<phot::IPhotonMappingTransformations> fMapping;

  float * fVisibilityTable;
};


GPUProducer::GPUProducer(fhicl::ParameterSet const& p)
  : EDProducer{p},
    fPVS(art::ServiceHandle<phot::PhotonVisibilityService const>()),
    fLibraryFileName(p.get<std::string>("LibraryFileName")),
    fSimulationTag(p.get<std::string>("SimulationTag")),
    fNThreads(p.get<size_t>("NThreads", 1)),
    fFastDecayTime(p.get<double>("FastDecayTime", 0.)), 
    fSlowDecayTime(p.get<double>("SlowDecayTime", 0.)),
    fRNG(0) {
  // Call appropriate consumes<>() for any products to be retrieved by this module.
    fhicl::ParameterSet mapDefaultSet;
    mapDefaultSet.put("tool_type", "PhotonMappingIdentityTransformations");
    fMapping = art::make_tool<phot::IPhotonMappingTransformations>(
      p.get<fhicl::ParameterSet>("Mapping", mapDefaultSet));

    produces<std::vector<sim::SimPhotonsLite>>();
    produces<std::vector<sim::OpDetBacktrackerRecord>>();
}

void GPUProducer::beginJob() {
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

void GPUProducer::endJob() {
  fLibraryFile->Close();
}


void GPUProducer::produce(art::Event & e) {

  art::ServiceHandle<geo::Geometry const> geom;
  size_t NOpDets = geom->NOpDets();
  auto phlit = std::make_unique<std::vector<sim::SimPhotonsLite>>();
  auto opbtr = std::make_unique<std::vector<sim::OpDetBacktrackerRecord>>();
  auto& dir_phlitcol(*phlit);
  dir_phlitcol.resize(NOpDets);
  std::vector<int> PDChannelToSOCMapDirect(NOpDets, -1);
  for (unsigned int i = 0; i < NOpDets; i++) {
    dir_phlitcol[i].OpChannel = i;
  }


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

  curandState * rng;
  cudaMallocManaged((void **)&rng, 256*sizeof(curandState));
  gpu::setup_rng_wrapper(rng);

  std::cout << "Calling wrapper" << std::endl;
  printf("%zu Op Dets\n", NOpDets);

  for (size_t op_id = 0; op_id < NOpDets; ++op_id) {
    std::cout << "Channel " << op_id << std::endl;
    std::vector<gpu::BTRHelper> btr_results;
    gpu::run_vis_functor(fVisibilityTable, NOpDets, fFastDecayTime,
                         fSlowDecayTime, op_id, cuda_deps, btr_results);
    std::cout << "Done " << btr_results.size() << std::endl;

    //std::cout << btr_results[0] << std::endl;
    for (auto & btrh : btr_results) {
      int nfast = btrh.nPhotFast;
      sim::OpDetBacktrackerRecord tmpbtr(op_id);
      double pos[3] = {btrh.posX, btrh.posY, btrh.posZ};
      for (int i = 0; i < nfast; ++i) {
        double dtime = btrh.time - fFastDecayTime*std::log(fRNG.Uniform());
        int time = static_cast<int>(std::round(dtime));
        ++dir_phlitcol[op_id].DetectedPhotons[time];
        tmpbtr.AddScintillationPhotons(btrh.trackID,
                                       time,
                                       1,
                                       pos,
                                       btrh.edep);
      }
      
      int nslow = btrh.nPhotSlow;
      for (int i = 0; i < nslow; ++i) {
        double dtime = btrh.time - fSlowDecayTime*std::log(fRNG.Uniform());
        int time = static_cast<int>(std::round(dtime));
        ++dir_phlitcol[op_id].DetectedPhotons[time];
        tmpbtr.AddScintillationPhotons(btrh.trackID,
                                       time,
                                       1,
                                       pos,
                                       btrh.edep);
      }
      AddOpDetBTR(*opbtr, PDChannelToSOCMapDirect, tmpbtr);
    }
  }

  e.put(move(phlit));
  e.put(move(opbtr));
}

void GPUProducer::AddOpDetBTR(std::vector<sim::OpDetBacktrackerRecord>& opbtr,
                              std::vector<int>& ChannelMap,
                              const sim::OpDetBacktrackerRecord& btr) const {
  int iChan = btr.OpDetNum();
  if (ChannelMap[iChan] < 0) {
    ChannelMap[iChan] = opbtr.size();
    opbtr.emplace_back(std::move(btr));
  }   
  else {
    size_t idtest = ChannelMap[iChan];
    auto const& timePDclockSDPsMap = btr.timePDclockSDPsMap();
    for (auto const& timePDclockSDP : timePDclockSDPsMap) {
      for (auto const& sdp : timePDclockSDP.second) {
        double xyz[3] = {sdp.x, sdp.y, sdp.z};
        opbtr.at(idtest).AddScintillationPhotons(
          sdp.trackID, timePDclockSDP.first, sdp.numPhotons, xyz, sdp.energy);
      }
    }   
  }   
}


DEFINE_ART_MODULE(GPUProducer)
