#ifndef SIMENERGYDEPCUDA_H
#define SIMENERGYDEPCUDA_H
#include "lardataobj/Simulation/SimEnergyDeposit.h"

#ifdef DOCUDA
#define HOSTDEVICE __host__ __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define HOST
#endif

class SimEnergyDepCuda {
  public:
    HOSTDEVICE SimEnergyDepCuda() {};
    HOST SimEnergyDepCuda(const sim::SimEnergyDeposit & dep);
    //__host__ SimEnergyDepCuda(const 
    HOSTDEVICE ~SimEnergyDepCuda() {};
    HOSTDEVICE int NumFPhotons();
    HOSTDEVICE int NumSPhotons();
    HOSTDEVICE int NumElectrons();
    HOSTDEVICE int TrackID();
    HOSTDEVICE int VoxelID();
    HOSTDEVICE void SetVoxelID(int v);
    HOSTDEVICE float Energy();
    HOSTDEVICE float MidPointX();
    HOSTDEVICE float MidPointY();
    HOSTDEVICE float MidPointZ();
    HOSTDEVICE double StartT();

  private:
    int numFPhotons;
    int numSPhotons;
    int numElectrons;
    int trackID;
    int voxelID;
    float edep;
    float midPointX, midPointY, midPointZ;
    double startTime;

};
#endif
