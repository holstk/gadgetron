#include <iostream>
//#include "ga2d_trajectory.cpp"
#include <complex>
#include "hoNDArray.h"

using namespace Gadgetron;

int main(int argc, char** argv){
  
  hoNDArray<int> array(2,2);
  for (int i = 0 ; i < 4 ; i++)
  {
    //for (int j = 0 ; j < 2 ; j++)
    //{
      array[i] = i+1;
    //}
  }
  //int o = 1;
  std::cout <<  array[0] << "   " << array[1] << "   " << array[2] << "   " << array[3] << std::endl;
  /*
  int Nspokes = 2;
  int NsamplesPerSpoke = 4;
  std::complex<float>* trajectory = 0;
  trajectory = new std::complex<float>[2];
  if (trajectory == 0) {
    std::cerr << "Unable to allocate memory for array" << std::endl;
    return 1;
  }

  trajectory[0] = std::complex<float>(0.5, 0.5);
  trajectory[1] = std::complex<float>(1.5, 1.5);

  std::cout << "trajectory[0] = " << trajectory[0] << "\ntrajectory[1] = " << trajectory[1] << std::endl;
  //std::complex<float> trajTmp = trajectory;
  double const2 = 2.0;
  trajectory[0] = trajectory[0] * float(2.0);
  trajectory[1] = trajectory[1] * float(2.0);
std::cout << "trajTmp[0] = " << trajectory[0] << "\ntrajTmp[1] = " << trajectory[1] << std::endl;

  //std::cout << "Test of class" << std::endl;
  //delete [] trajectory;
  */
  return 0;
}
