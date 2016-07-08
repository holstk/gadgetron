#ifndef GRIDDER_H
#define GRIDDER_H

#include "hoNDArray.h"
#include "hoNDFFT.h"
#include "hoNDArray_fileio.h"

using namespace Gadgetron;

template <class T, unsigned int D> class Gridder
{
public:
    Gridder(hoNDArray<typename reald<T,D>::Type>& coordinates, int output_dimensions[], T over_sampling = 2.0);
    ~Gridder();
    
    hoNDArray< std::complex<T> > convolution_2d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction);
    hoNDArray< std::complex<T> > convolution_3d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction);
    
    hoNDArray< std::complex<T> > calculate_deapodization_filter();
    
    hoNDArray< T > return_kernel_tables();
    
private:
    T over_sampling;
    T kernel_width;
    int kernel_table_steps;
    T kernel_beta;
    int kernel_samples;
    int transformed_kernel_samples;
    
    T* kernel_table;
    T* transformed_kernel_table;
    
    hoNDArray<int> dims;
    hoNDArray<int> oversampled_dims;
    hoNDArray<int> grid_positions;
    hoNDArray<int> kernel_positions;
    
    void calculate_kernel_tables();
    void calculate_point_vectors(hoNDArray<typename reald<T,D>::Type>& kt_pos);
    double bessi0(double x);
};


#endif //GRIDDER_H
