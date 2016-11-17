#ifndef ROTATE_KSPACE_H
#define ROTATE_KSPACE_H

#include <iostream>
#include <complex>

#include "math.h"
#include "hoNDArray.h"
#include "vector_td_utilities.h"
#include "hoNDFFT.h"

using namespace Gadgetron;

template <class T> class rotate_kspace
{
public:

    unsigned int tmp;

    void rotate_trajectory( hoNDArray<typename reald<T,3>::Type > *trajectory, hoNDArray<T> rotation);

};




#endif //ROTATE_KSPACE_H
