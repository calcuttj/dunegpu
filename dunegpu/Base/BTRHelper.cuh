#ifndef BTRHELPER_H
#define BTRHELPER_H
#include <iostream>
namespace gpu {
struct BTRHelper {
  public:

    __host__ __device__
    BTRHelper() {};

    __host__ __device__
    ~BTRHelper() {};

    __host__ __device__
    BTRHelper(int nfast, int nslow, int tid, float px, float py, float pz,
              double t, float ed)
      : nPhotFast(nfast),
        nPhotSlow(nslow),
        trackID(tid),
        posX(px),
        posY(py),
        posZ(pz),
        time(t),
        edep(ed) {};

    __host__
    friend std::ostream & operator << (std::ostream & out, const BTRHelper & b) {
      out << b.trackID << " " << b.edep << "\n";
      out << b.nPhotFast << " " << b.nPhotSlow << " " << b.time << "\n";
      out << b.posX << " " << b.posY << " " << b.posZ << std::endl;
      return out;
    };

    int nPhotFast;
    int nPhotSlow;
    int trackID;
    float posX;
    float posY;
    float posZ;
    double time;
    float edep;
     
};
}
#endif
