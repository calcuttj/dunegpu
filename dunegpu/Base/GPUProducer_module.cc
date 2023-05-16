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
#include <ROOT/TSeq.hxx>
//#include <ROOT/TTreeProcessorMT.hxx>
//#include "TTreeReader.h"

#include <cmath>
#include <thread>
#include <mutex>

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
  void DepLoop(std::vector<SimEnergyDepCuda> & deps,
               size_t workid,
               int n_workers);
  void SaveLoop(std::vector<gpu::BTRHelper> & btr_results,
                std::vector<sim::OpDetBacktrackerRecord> & opbtr,
                std::vector<sim::SimPhotonsLite> & dir_phlitcol,
                size_t workerid);

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
  std::mutex fSaveMutex;
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
  mf::LogInfo("GPUProducer") << "Loading library" << std::endl;
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
  mf::LogInfo("GPUProducer") << "Loaded library" << std::endl;

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

void GPUProducer::DepLoop(std::vector<SimEnergyDepCuda> & deps, size_t workid, int n_workers) {
  printf("Thread %zu, n_workers: %i\n", workid, n_workers);
  for (size_t i = workid; i < deps.size(); i += n_workers) {
    int voxel_id = fPVS->GetVoxelDef().GetVoxelID(
        fMapping->detectorToLibrary({deps[i].MidPointX(),
                                     deps[i].MidPointY(),
                                     deps[i].MidPointZ()}));
    deps[i].SetVoxelID(voxel_id);
  }
}

void GPUProducer::produce(art::Event & e) {

  //Tell art you're saving these
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


  //Get the energy deposits
  //auto allDeps = *(e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(fSimulationTag));
  auto cuda_deps = *(e.getValidHandle<std::vector<SimEnergyDepCuda>>("test"));
  //size_t ndeps = allDeps.size();

  //Turn them into objects that can be run on the GPUs
  //This is a limiting factor & inefficiency
  //mf::LogInfo("GPUProducer") << "Making Deps " << allDeps.size() << std::endl;
  //std::vector<SimEnergyDepCuda> cuda_deps(allDeps.begin(), allDeps.end());
  mf::LogInfo("GPUProducer") << "Done" << std::endl;

  std::vector<std::thread> workers;
  for (auto workerid : ROOT::TSeqI(fNThreads)) {
    std::thread worker(&GPUProducer::DepLoop, this,
        std::ref(cuda_deps), workerid, fNThreads);
    workers.emplace_back(std::move(worker));
    //workers.emplace_back(&AbsCexDriver::RefillSampleLoop, this,
       // events, samples, signal_sample_checks, beam_energy_bins,
       // signal_pars, flux_pars, syst_pars, fit_under_over,
       // tie_under_over, use_beam_inst_P, fill_incident, fix_factors,
       // workerid, events_split);
  }

  for (auto &&worker : workers) { worker.join();}

  //DepLoop(cuda_deps, 0, 1);
  //for (size_t i = 0; i < ndeps; ++i) {
  //  //cuda_deps.push_back(SimEnergyDepCuda((*allDeps)[i]));
  //  int voxel_id = fPVS->GetVoxelDef().GetVoxelID(
  //      fMapping->detectorToLibrary(allDeps[i].MidPoint()));
  //  cuda_deps[i].SetVoxelID(voxel_id);
  //}

  printf("%zu Op Dets\n", NOpDets);

  //Loop over the channel ids
  mf::LogInfo("GPUProducer") << "Starting processing" << std::endl;
  for (size_t op_id = 0; op_id < NOpDets; ++op_id) {
    auto chan_start = std::chrono::high_resolution_clock::now();

    //For each channel ID, call the wrapper function.
    //It will use thrust vectors and perform 'transformations'
    //which will get the visibility from the stored library
    //multiply that by the number of slow/fast photons 
    //created at each energy deposit and generate poisson-dist'd
    //number of photons that were detected by the given optical detector
    //
    //It only saves OpDet helper containers that had >0 photons detected
    std::vector<gpu::BTRHelper> btr_results;
    gpu::run_vis_functor(fVisibilityTable, NOpDets, op_id, cuda_deps,
                         btr_results);
    auto chan_end = std::chrono::high_resolution_clock::now();
    auto delta_proc = std::chrono::duration_cast<std::chrono::milliseconds>(
        chan_end - chan_start).count();
    std::cout << "Processing " << op_id << " took " << delta_proc << std::endl;

    //Go over the results and turn into the art-root objects that can be saved
    //For each slow/fast photon calculate a random time at which it reaches the
    //opdet and add it to the event record
    //
    //Could multi-thread (on CPU) this maybe
    auto save_start = std::chrono::high_resolution_clock::now();
    /*for (auto & btrh : btr_results) {
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
    }*/

    std::vector<std::thread> save_workers;
    for (auto workerid : ROOT::TSeqI(fNThreads)) {
      std::thread worker(&GPUProducer::SaveLoop, this,
          std::ref(btr_results), std::ref(*opbtr), std::ref(dir_phlitcol),
                   workerid);
      save_workers.emplace_back(std::move(worker));
    }
    for (auto &&worker : save_workers) { worker.join();}

    auto save_end = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
        save_end - save_start).count();
    std::cout << "Saving took " << delta << std::endl;
  }

  e.put(move(phlit));
  e.put(move(opbtr));
}

void GPUProducer::SaveLoop(std::vector<gpu::BTRHelper> & btr_results,
                           std::vector<sim::OpDetBacktrackerRecord> & opbtr,
                           std::vector<sim::SimPhotonsLite> & dir_phlitcol,
                           size_t workerid) {

  std::vector<int> PDChannelToSOCMapDirect(dir_phlitcol.size(), -1);
  for (size_t i = workerid; i < btr_results.size(); i += fNThreads) {
    auto & btrh = btr_results[i];
    int nfast = btrh.nPhotFast;
    sim::OpDetBacktrackerRecord tmpbtr(workerid);
    double pos[3] = {btrh.posX, btrh.posY, btrh.posZ};
    for (int i = 0; i < nfast; ++i) {
      double dtime = btrh.time - fFastDecayTime*std::log(fRNG.Uniform());
      int time = static_cast<int>(std::round(dtime));
      ++dir_phlitcol[workerid].DetectedPhotons[time];
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
      ++dir_phlitcol[workerid].DetectedPhotons[time];
      tmpbtr.AddScintillationPhotons(btrh.trackID,
                                     time,
                                     1,
                                     pos,
                                     btrh.edep);
    }
    std::lock_guard<std::mutex> guard(fSaveMutex);
    AddOpDetBTR(opbtr, PDChannelToSOCMapDirect, tmpbtr);
  }
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
