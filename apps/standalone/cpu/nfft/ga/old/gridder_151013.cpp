#include "gridder.hpp"
#include "hoNDArray.h"
#include <iostream>

using namespace Gadgetron;

#ifndef SQR
#define SQR(x)		((x)*(x))
#endif

#define GRID_DEFAULT_OVERSAMPLE_FACTOR 2.0
#define GRID_DEFAULT_KERNEL_WIDTH 4.0
#define GRID_DEFAULT_KERNEL_TABLE_STEPS 100
#define GRID_DEFAULT_KERNEL_BETA 18.5547



template <class T> Gridder<T>::Gridder(hoNDArray<T>& coordinates, int output_dimensions[], T over_sampling_factor)
{
  over_sampling = over_sampling_factor;
  kernel_width = GRID_DEFAULT_KERNEL_WIDTH;
  kernel_table_steps = GRID_DEFAULT_KERNEL_TABLE_STEPS;
  kernel_beta = GRID_DEFAULT_KERNEL_BETA;
  kernel_table = 0;
  transformed_kernel_table = 0;

  calculate_kernel_tables();

  if (coordinates.get_number_of_dimensions() != 2)
  {
    std::cout << "Invalid coordinate format for gridding." << std::endl;
    return;
  }

  if (coordinates.get_size(1) < 2 || coordinates.get_size(1) > 3 )
  {
    std::cout << "Gridding only implemented for 2 or 3 dimensions at the moment." << std::endl;
    return;
  }

  dims.create(coordinates.get_size(1));
  //int dims = new int[2];
  //int dims = new int[coordinates.get_size(1)];
  dims[0] = output_dimensions[0];  dims[1] = output_dimensions[1];
  //dims = output_dimensions;
  //oversampled_dimensions = new int[2];
  oversampled_dimensions.create(coordinates.get_size(1));
  oversampled_dimensions[0] = dims[0];  oversampled_dimensions[1] = dims[1];
  std::cout << "output_dimensions: " << output_dimensions[0] << ", " << output_dimensions[1] << std::endl;
  std::cout << "dims: " << dims[0] << ", " << dims[1] << std::endl;
  std::cout << "oversampled_dimensions: " << oversampled_dimensions[0] << ", " << oversampled_dimensions[1] << std::endl;
  
  
  std::cout << "coordinates.get_size(1): " << coordinates.get_size(1) << std::endl;
  fixed_dims.create(coordinates.get_size(1));
  //fixed_dims = new int[coordinates.get_size(1)];
  //fixed_dims[coordinates.get_size(1)];
  

  
  for (int i = 0; i < coordinates.get_size(1); i++)
  {
    fixed_dims[i] = 1;
    for (int j = 0; j < coordinates.get_size(0); j++)
    {
      if (floor(coordinates[coordinates.get_size(0)*i + j]) != coordinates[coordinates.get_size(0)*i + j])
      {
	fixed_dims[i] = 0;  
	std::cout << "coordinates.get_size(0)" << coordinates.get_size(0) << std::endl;
	std::cout << "coordinates.get_size(1)" << coordinates.get_size(1) << std::endl;
	std::cout << "coordinates[" << j << "," << i << "] = " << coordinates[coordinates.get_size(0)*i + j] << std::endl;
	break;
      }
    }
  }
  
  for (unsigned int i = 0; i < 2; i++)
  {
    if (fixed_dims[i] == 0)
    {
      oversampled_dimensions[i] = static_cast<int>(dims[i]*over_sampling);
    }
    else
    {
      oversampled_dimensions[i] = dims[i];
    }
  }
  
  for (int i = 0; i < coordinates.get_size(1); i++) std::cout << "fixed_dims[" << i << "] = " << fixed_dims[i] << std::endl;

  std::cout << "coordinates.get_number_of_dimensions(): " << coordinates.get_number_of_dimensions() << std::endl;
  std::cout << "coordinates.get_number_of_elements(): " << coordinates.get_number_of_elements() << std::endl;
  std::cout << "coordinates.get_size(0): " << coordinates.get_size(0) << std::endl;
  std::cout << "coordinates.get_size(1): " << coordinates.get_size(1) << std::endl;
  

  calculate_point_vectors(coordinates);

}

template <class T> Gridder<T>::~Gridder()
{
  if (kernel_table != 0) delete [] kernel_table;
  if (transformed_kernel_table != 0) delete [] transformed_kernel_table;

}



template <class T> hoNDArray< std::complex<T> > Gridder<T>::convolution_2d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction)
{
	//fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0

	int n,x,y;
	T kx,ky;
	int o1,o;
	int kernel_limits[2];
	int kernel_step[2];
	int oversampled_dims[2];
	int ndim = 2;

	int npoints = kernel_positions.get_size(0);
	hoNDArray< std::complex<T> > weighted_data_in;

	/*-----------------*/
	/* Apply weights   */
	/*-----------------*/    
	if (weight != 0 && direction) /* Weights only make sense going onto the cartesian grid */  
	{
	  weighted_data_in = hoNDArray< std::complex<T> >(data_in);

	  for (n=0;n<npoints;n++)
	  {
		  weighted_data_in[n] = data_in[n]*(*weight)[n];
	  }
  }
  else
  {
	  weighted_data_in = data_in;
  }

  for (n = 0 ; n < ndim ; n++)
  {
	  if (fixed_dims[n])
	  {
		  kernel_limits[n] = 1;
		  oversampled_dims[n] = dims[n];
		  kernel_step[n] = kernel_table_steps;
		  std::cout << "for n = " << n << ":\nkernel_limits: " << kernel_limits[n] << "\noversampled_dims: " << oversampled_dims[n] << "\nkernel_step: " << kernel_step[n] << std::endl;
	  }
	  else
	  {
		  kernel_limits[n] = static_cast<int>(kernel_width*over_sampling);
		  oversampled_dims[n] = static_cast<int>(over_sampling*dims[n]);
		  kernel_step[n] = static_cast<int>(kernel_table_steps/over_sampling);
	  }   
  }

  hoNDArray< std::complex<T> > data_out;
  if (direction)
  {
		std::cout << "ndim: " << ndim << "\noversampled_dims[0]: " << oversampled_dims[0] << "\noversampled_dims[1]: " << oversampled_dims[1] << std::endl;
          	data_out = hoNDArray< std::complex<T> >(ndim, oversampled_dims[0], oversampled_dims[1]);
  }
  else
  {
                data_out = hoNDArray< std::complex<T> >(npoints);
  }

  for (n = 0 ; n < npoints ; n++)
  {
	  for (y=0;y<kernel_limits[1];y++)
	  {
		  if (fixed_dims[1])
		  {
			  ky = 1;
		  }
		  else
		  {
			  ky = kernel_table[kernel_positions[npoints + n]+y*kernel_step[1]];
		  }
		  o1 = oversampled_dims[0]*((grid_positions[npoints + n]+y+oversampled_dims[1])%oversampled_dims[1]);
		  for (x = 0 ; x < kernel_limits[0] ; x++)
		  {
			  if (fixed_dims[0])
			  {
				  kx = 1;
			  }
			  else
			  {
				  kx = ky*(kernel_table[kernel_positions[n]+x*kernel_step[0]]);
			  }
			  o = (o1 + (grid_positions[n]+x+oversampled_dims[0])%oversampled_dims[0]);

			  if (direction)
			  {
				  data_out[o] += kx*weighted_data_in[n];
			  }
			  else
			  {
				  data_out[n] += kx*data_in[o];
			  }
		  }
	  }
  }
  std::cout << "Data: " << data_out.get_size(0) << ", " << data_out.get_size(1) << std::endl;
  return data_out;
}
												   
												   


template <class T> void Gridder<T>::calculate_kernel_tables()
{
  T k, k2;
  T kernel_norm;
    
  kernel_samples = (int)kernel_width*kernel_table_steps+1;
  try
  {
    kernel_table = new T[kernel_samples];
  }
  catch (bad_alloc&)
  {
    std::cout << "Error allocating kernel table" << std::endl;
  }

  k = static_cast<T>(-kernel_samples>>1);
  kernel_norm = 0;
  for (int i = 0 ; i < kernel_samples; i++)
  {
    k2 = static_cast<T>(1.0)-SQR(static_cast<T>(2.0)*k/kernel_samples);
    if (k2<0) k2=0; else k2=static_cast<T>(sqrt(k2));    /* Prevent round off error below 0 */
    k2 = static_cast<T>(bessi0(kernel_beta * k2));
    kernel_table[i] = k2;
    kernel_norm += k2;
    k++;
  }
  for (int i = 0; i < kernel_samples; i++)
  {
    kernel_table[i] /= (kernel_norm/kernel_table_steps);
  }
}


template<class T> void Gridder<T>::calculate_point_vectors(hoNDArray<T>& kt_pos)
{
  
  int npoints = kt_pos.get_size(0);
  int ndim    = kt_pos.get_size(1);

  std::cout << "npoints: " << npoints << "\nndim: " << ndim << std::endl;
  //hoNDArray<int> tmp_(npoints,ndim);
  kernel_positions.create(npoints,ndim);
  std::cout << "Debug: after kernel_pos init." << std::endl;
  //hoNDArray<int> tmp2_(npoints,ndim);
  grid_positions.create(npoints,ndim);
  std::cout << "Debug: after grid_pos init." << std::endl;
  
  T tmp;
  return;

  for (int i = 0 ; i < npoints ; i++)
  {
    for(int j = 0; j < ndim; j++)
    {
      if (fixed_dims[j])
      {
	grid_positions[npoints*j+i] = static_cast<int>(kt_pos[npoints*j+i])+(dims[j]>>1);
	kernel_positions[npoints*j+i] = static_cast<int>(abs((kt_pos[npoints*j+i]+(dims[j]>>1) - kernel_width/2.0)*kernel_table_steps - grid_positions[npoints*j+i]*kernel_table_steps));
      }
      else
      {
	tmp = kt_pos[npoints*j+i]+(dims[j]>>1) - kernel_width/2.0;
	grid_positions[npoints*j+i] = static_cast<int>(ceil(tmp*over_sampling));
	kernel_positions[npoints*j+i] = static_cast<int>(abs((tmp*over_sampling - grid_positions[npoints*j+i])*(kernel_table_steps/over_sampling)));
      }
    }
  }
}


template <class T> double Gridder<T>::bessi0(double x)
{
  double ax,ans;
  double y;
  if ((ax=fabs(x)) < 3.75) 
  {
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } 
  else 
  {
    y=3.75/ax;
    ans=(-0.02057706+y*(0.02635537+y*(-0.01647633+(y*0.00392377))));
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.01328592+y*(0.00225319+y*(-0.00157565+y*(0.00916281+y*ans)))));
  }
  return ans;
}

template class Gridder<float>;
template class Gridder<double>;

#undef SQR
