#ifndef BTRHELPER_H
#define BTRHELPER_H
namespace gpu {
struct BTRHelper {
  public:

    __host__ __device__
    BTRHelper() {};

    __host__ __device__
    ~BTRHelper() {};

    __host__ __device__
    BTRHelper(int nfast, int nslow, int tid, float px, float py, float pz,
              float tfast, float tslow, float ed)
      : nPhotFast(nfast),
        nPhotSlow(nslow),
        trackID(tid),
        posX(px),
        posY(py),
        posZ(pz),
        timeFast(tfast),
        timeSlow(tslow),
        edep(ed) {};

    int nPhotFast;
    int nPhotSlow;
    int trackID;
    float posX;
    float posY;
    float posZ;
    float timeFast;
    float timeSlow;
    float edep;
     
};
}
#endif
