#ifndef GRIDDER_HPP
#define GRIDDER_HPP

//#include "types.hpp"
//#include "types.hcu"

//#include "export.h"

#include "hoNDArray.h"

template <class T> class Gridder
{
public:
        Gridder(hoNDArray<T>& coordinates, int output_dimensions[], T over_sampling_factor = 2.0);
	~Gridder();

	hoNDArray< std::complex<T> > convolution_2d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction);
	hoNDArray< std::complex<T> > convolution_3d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction);

	hoNDArray< std::complex<T> > calculate_deapodization_filter();

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
        hoNDArray<int> oversampled_dimensions;
        //hoNDArray<int> fixed_dims;
        hoNDArray<int> fixed_dims;
        hoNDArray<int> grid_positions;
        hoNDArray<int> kernel_positions;
  
        void calculate_kernel_tables();
	void calculate_point_vectors(hoNDArray<T>& kt_pos);
	double bessi0(double x);
};


#endif //GRIDDER_HPP
