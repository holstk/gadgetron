#ifndef GRIDDER_H
#define GRIDDER_H

#include "types.hpp"
#include "types.hcu"

#include "export.h"

template <class T> class DLLEXPORT Gridder
{
public:
	Gridder(mr_recon::NDArray<T>& coordinates, mr_recon::RealIntArray& output_dimensions, T over_sampling_factor = 2.0);
	~Gridder();

	mr_recon::NDArray< std::complex<T> > convolution_2d(mr_recon::NDArray< std::complex<T> >& data_in, mr_recon::NDArray<T>* weight, int direction);
	mr_recon::NDArray< std::complex<T> > convolution_3d(mr_recon::NDArray< std::complex<T> >& data_in, mr_recon::NDArray<T>* weight, int direction);

	mr_recon::NDArray< std::complex<T> > calculate_deapodization_filter();

private:
	T over_sampling;
	T kernel_width;
	int kernel_table_steps;
	T kernel_beta;
	int kernel_samples;
	int transformed_kernel_samples;

	T* kernel_table;
	T* transformed_kernel_table;

	mr_recon::RealIntArray dims;
	mr_recon::RealIntArray oversampled_dimensions;
	mr_recon::RealIntArray fixed_dims;
	mr_recon::RealIntArray kernel_positions;
	mr_recon::RealIntArray grid_positions;

	void calculate_kernel_tables();
	void calculate_point_vectors(mr_recon::NDArray<T>& kt_pos);
	double bessi0(double x);
};


#endif //GRIDDER_H
