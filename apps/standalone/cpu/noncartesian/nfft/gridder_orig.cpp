#include "types.hpp"

#include <iostream>

using namespace std;
using namespace mr_recon;

#ifndef SQR
#define SQR(x)		((x)*(x))
#endif

#define GRID_DEFAULT_OVERSAMPLE_FACTOR 2.0
#define GRID_DEFAULT_KERNEL_WIDTH 4.0
#define GRID_DEFAULT_KERNEL_TABLE_STEPS 100
#define GRID_DEFAULT_KERNEL_BETA 18.5547

template <class T> Gridder<T>::Gridder(NDArray<T>& coordinates, RealIntArray& output_dimensions, T over_sampling_factor)
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
    cout << "Invalid coordinate format for gridding." << endl;
    return;
  }

  if (coordinates.get_size(1) < 2 || coordinates.get_size(1) > 3 )
  {
    cout << "Gridding only implemented for 2 or 3 dimensions at the moment." << endl;
    return;
  }

  dims = output_dimensions;
  oversampled_dimensions = dims;

  fixed_dims = RealIntArray(coordinates.get_size(1));
  /* Determine which dimensions need gridding */

  for (int i = 0; i < coordinates.get_size(1); i++)
  {
    fixed_dims[i] = 1;
    for (int j = 0; j < coordinates.get_size(0); j++)
    {
      if (floor(coordinates[coordinates.get_size(0)*i + j]) != coordinates[coordinates.get_size(0)*i + j])
      {
	fixed_dims[i] = 0;
	break;
      }
    }
  }


  for (unsigned int i = 0; i < dims.get_number_of_elements(); i++)
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
  
  //for (int i = 0; i < coordinates.get_size(1); i++) cout << "fixed_dims[" << i << "] = " << fixed_dims[i] << endl;

  calculate_point_vectors(coordinates);

}

template <class T> Gridder<T>::~Gridder()
{
  if (kernel_table != 0) delete [] kernel_table;
  if (transformed_kernel_table != 0) delete [] transformed_kernel_table;

}

template <class T> NDArray< complex<T> > Gridder<T>::convolution_2d(NDArray< complex<T> >& data_in, NDArray<T>* weight, int direction)
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
	NDArray< complex<T> > weighted_data_in;

	/*-----------------*/
	/* Apply weights   */
	/*-----------------*/
	if (weight != 0 && direction) /* Weights only make sense going onto the cartesian grid */
	{
		weighted_data_in = NDArray< complex<T> >(data_in);

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
	  }
	  else
	  {
		  kernel_limits[n] = static_cast<int>(kernel_width*over_sampling);
		  oversampled_dims[n] = static_cast<int>(over_sampling*dims[n]);
		  kernel_step[n] = static_cast<int>(kernel_table_steps/over_sampling);
	  }   
  }

  NDArray< complex<T> > data_out;
  if (direction)
  {
	  data_out = NDArray< complex<T> >(ndim, oversampled_dims);
  }
  else
  {
	  data_out = NDArray< complex<T> >(npoints);
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
  return data_out;
}


template <class T> NDArray< complex<T> > Gridder<T>::convolution_3d(NDArray< complex<T> >& data_in, NDArray<T>* weight, int direction)
{
  //fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0
    
  int n,x,y,z;
  T kx,ky,kz;
  int o2,o1,o;
  int kernel_limits[3];
  int kernel_step[3];
  int oversampled_dims[3];
  int ndim = 3;

  int npoints = kernel_positions.get_size(0);
  NDArray< complex<T> > weighted_data_in;
  
  /*-----------------*/
  /* Apply weights   */
  /*-----------------*/
  if (weight != 0 && direction) /* Weights only make sense going onto the cartesian grid */
  {
    weighted_data_in = NDArray< complex<T> >(data_in);
    
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
    }
    else
    {
      kernel_limits[n] = static_cast<int>(kernel_width*over_sampling);
      oversampled_dims[n] = static_cast<int>(over_sampling*dims[n]);
      kernel_step[n] = static_cast<int>(kernel_table_steps/over_sampling);
    }   
  }

  NDArray< complex<T> > data_out;
  if (direction)
  {
    
    data_out = NDArray< complex<T> >(ndim, oversampled_dims);
  }
  else
  {
    data_out = NDArray< complex<T> >(npoints);
  }

  for (n = 0 ; n < npoints ; n++)
  {
    for (z=0;z<kernel_limits[2];z++)
    {
      if (fixed_dims[2])
      {
	kz = 1;
      }
      else
      {
	kz = kernel_table[kernel_positions[npoints*2 + n]+z*kernel_step[2]];
      }
      o2 = oversampled_dims[1]*oversampled_dims[0]*((grid_positions[npoints*2 + n]+z+oversampled_dims[2])%oversampled_dims[2]);
 
      for (y=0;y<kernel_limits[1];y++)
      {
	if (fixed_dims[1])
	{
	  ky = 1;
	}
	else
	{
	  ky = kz * kernel_table[kernel_positions[npoints + n]+y*kernel_step[1]];
	}
	o1 = o2 + oversampled_dims[0]*((grid_positions[npoints + n]+y+oversampled_dims[1])%oversampled_dims[1]);
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
  }
  return data_out;
}


template <class T> NDArray< complex<T> > Gridder<T>::calculate_deapodization_filter()
{
  int ndim = kernel_positions.get_size(1);

  NDArray< complex<T> > a(ndim, oversampled_dimensions.get_data_ptr());
  int kernel_table_center = ((static_cast<int>(kernel_table_steps*kernel_width))>>1);
  
  int zmin, zmax, ymin, ymax, xmin, xmax, zoffset, yoffset, xoffset;

  if (ndim == 3 && fixed_dims[2] == 0)
  {
    zmin = static_cast<int>(-floor(kernel_width/2)*over_sampling);
    zmax = static_cast<int>(floor(kernel_width/2)*over_sampling);
  }
  else
  {
    zmin = 0; zmax = 0;
  }
  if (ndim == 3)
  {
    zoffset = oversampled_dimensions[2]>>1;
  }
  else
  {
    zoffset = 0;
  }
  
  if (fixed_dims[1] == 0)
  {
    ymin = static_cast<int>(-floor(kernel_width/2)*over_sampling);
    ymax = static_cast<int>(floor(kernel_width/2)*over_sampling);
  }
  else
  {
    ymin = 0; ymax = 0;
  }
  yoffset = oversampled_dimensions[1]>>1;

  if (fixed_dims[0] == 0)
  {
    xmin = static_cast<int>(-floor(kernel_width/2)*over_sampling);
    xmax = static_cast<int>(floor(kernel_width/2)*over_sampling);
  }
  else
  {
    xmin = 0; xmax = 0;
  }
  xoffset = oversampled_dimensions[0]>>1;


  T kz, ky, kx;
  int points = 0;
  int ix,iy,iz;
  for (int z = zmin; z <= zmax; z++)
  {
    iz = static_cast<int>(kernel_table_center+(z*(kernel_table_steps/over_sampling)));
    if (ndim == 3 && !fixed_dims[2])
    {
      kz = kernel_table[iz];
    }
    else
    {
      kz = 1;
    }
    for (int y = ymin; y <= ymax; y++)
    {
      iy = static_cast<int>(kernel_table_center+(y*(kernel_table_steps/over_sampling)));
      ky = kernel_table[iy];
      for (int x = xmin; x <= xmax; x++)
      {
	ix = static_cast<int>(kernel_table_center+(x*(kernel_table_steps/over_sampling)));
	kx = kernel_table[ix];
	a[(z+zoffset)*oversampled_dimensions[1]*oversampled_dimensions[0]+(y+yoffset)*oversampled_dimensions[0]+(x+xoffset)] = kx*ky*kz;
	points++;
      }
    }
  }

  Transformer<T> t;
  t.K2I(a);

  return a;
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
    cout << "Error allocating kernel table" << endl;
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


template<class T> void Gridder<T>::calculate_point_vectors(NDArray<T>& kt_pos)
{
  
  int npoints = kt_pos.get_size(0);
  int ndim    = kt_pos.get_size(1);

  grid_positions = RealIntArray(npoints,ndim);
  kernel_positions = RealIntArray(npoints,ndim);

  T tmp;
  
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
