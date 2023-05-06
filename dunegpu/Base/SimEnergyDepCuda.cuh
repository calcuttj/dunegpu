#include "lardataobj/Simulation/SimEnergyDeposit.h"


class SimEnergyDepCuda {
  public:
    __host__ __device__ SimEnergyDepCuda() {};
    __host__ SimEnergyDepCuda(const sim::SimEnergyDeposit & dep);
    __host__ __device__ ~SimEnergyDepCuda() {};
    __host__ __device__ int NumFPhotons();
    __host__ __device__ int NumSPhotons();
    __host__ __device__ int NumElectrons();
    __host__ __device__ int TrackID();
    __host__ __device__ float Energy();
    __host__ __device__ float MidPointX();
    __host__ __device__ float MidPointY();
    __host__ __device__ float MidPointZ();
    __host__ __device__ double StartT();

  private:
    int numFPhotons;
    int numSPhotons;
    int numElectrons;
    int trackID;
    float edep;
    float midPointX, midPointY, midPointZ;
    double startTime;
};
