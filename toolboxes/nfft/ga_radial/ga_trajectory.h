#ifndef GA_TRAJECTORY_H
#define GA_TRAJECTORY_H

#include <iostream>
#include <complex>

//#include "ga_trajectory.cpp"
#include "math.h"
#include "hoNDArray.h"
#include "vector_td_utilities.h"
//#include "hoNDArray_elemwise.h"
#include "hoNDFFT.h"

using namespace Gadgetron;

template <class T, unsigned int D> class GATrajectory
{
public:
    //GATrajectory();
    //~GATrajectory();
    hoNDArray<typename reald<T,D>::Type> calculateTrajectory(int Nspokes, int NsamplesPerSpoke, int Ndim, long int *spoke_idx = 0, int correction_type = 0);

    hoNDArray<T> calculateDCF(int Nspokes, int NsamplesPerSpoke, int Ndim, bool cut_off, int grid_size);

    void calculate_angles(int Nspokes, int correction_type);

    T get_azimuthal_angle(int spoke){
	return azi(spoke);
    }

    T get_polar_angle(int spoke){
	return pol(spoke);
    }

protected:
    hoNDArray<typename reald<T,D>::Type> trajectory;
    //hoNDArray<T> trajectory;
    hoNDArray<T> DCF_out;
    hoNDArray<T> azi;
    hoNDArray<T> pol;
};




#endif //GA_TRAJECTORY_H
