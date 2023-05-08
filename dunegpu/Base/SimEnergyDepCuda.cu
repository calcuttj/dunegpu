#include "SimEnergyDepCuda.cuh"

__host__
SimEnergyDepCuda::SimEnergyDepCuda(const sim::SimEnergyDeposit & dep)
  : numFPhotons(dep.NumFPhotons()),
    numSPhotons(dep.NumSPhotons()),
    numElectrons(dep.NumElectrons()),
    trackID(dep.TrackID()),
    edep(dep.Energy()),
    midPointX(dep.MidPointX()),
    midPointY(dep.MidPointY()),
    midPointZ(dep.MidPointZ()),
    startTime(dep.StartT()) {}

__host__ __device__
int SimEnergyDepCuda::NumFPhotons() {return numFPhotons;}

__host__ __device__
int SimEnergyDepCuda::NumSPhotons() {return numSPhotons;}

__host__ __device__
int SimEnergyDepCuda::NumElectrons() {return numElectrons;}

__host__ __device__
int SimEnergyDepCuda::TrackID() {return trackID;}


__host__ __device__
int SimEnergyDepCuda::VoxelID() {return voxelID;}

__host__ __device__
void SimEnergyDepCuda::SetVoxelID(int v) {voxelID = v;}

__host__ __device__
float SimEnergyDepCuda::Energy() {return edep;}

__host__ __device__
float SimEnergyDepCuda::MidPointX() {return midPointX;}

__host__ __device__
float SimEnergyDepCuda::MidPointY() {return midPointY;}

__host__ __device__
float SimEnergyDepCuda::MidPointZ() {return midPointZ;}

__host__ __device__
double SimEnergyDepCuda::StartT() {return startTime;}
