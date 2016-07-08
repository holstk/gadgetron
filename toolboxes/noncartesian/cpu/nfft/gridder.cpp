#include "gridder.h"
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



template <class T, unsigned int D> Gridder<T,D>::Gridder(hoNDArray<typename reald<T,D>::Type>& coordinates, int output_dimensions[], T over_sampling_in)
{
    over_sampling = over_sampling_in;
    kernel_width = GRID_DEFAULT_KERNEL_WIDTH;
    kernel_table_steps = GRID_DEFAULT_KERNEL_TABLE_STEPS;
    kernel_beta = GRID_DEFAULT_KERNEL_BETA;
    kernel_table = 0;
    transformed_kernel_table = 0;
    
    calculate_kernel_tables();

    std::cout << "coordinates.get_number_of_dimensions() = " << coordinates.get_number_of_dimensions() << std::endl;
    
    /**
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
    */
    
    dims.create(D); 
    std::cout << "coordinates.size(1): " << coordinates.get_size(1) << std::endl;
    oversampled_dims.create(D);
    for (int i = 0 ; i < D ; i++)
    {
	dims[i] = output_dimensions[i];
	oversampled_dims[i] = std::ceil( output_dimensions[i] * over_sampling );
	std::cout << "oversampled_dims[" << i << "]: " << oversampled_dims[i] << std::endl;
    }
    calculate_point_vectors(coordinates);

    std::cout << "Oversampled_dims in Gridder " << std::endl;
    std::cout << "noversampled_dims[0]>>1: " << (oversampled_dims[0]>>1) << std::endl;
    std::cout << "noversampled_dims[1]>>1: " << (oversampled_dims[1]>>1) << std::endl;
    std::cout << "noversampled_dims[2]>>1: " << (oversampled_dims[2]>>1) << std::endl;

    
}

template <class T, unsigned int D> Gridder<T,D>::~Gridder()
{
    if (kernel_table != 0) delete [] kernel_table;
    if (transformed_kernel_table != 0) delete [] transformed_kernel_table;
}



template <class T, unsigned int D> hoNDArray< std::complex<T> > Gridder<T,D>::convolution_2d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction)
{
    int n,x,y;
    T kx,ky;
    int o1,o;
    int kernel_limits[2];
    int kernel_step[2];
    int oversampled_dims[2];
    int ndim = 2;
    
    float tmp_var=0;
    
    int npoints = kernel_positions.get_size(0);
    hoNDArray< std::complex<T> > weighted_data_in;
    
    //Apply weights
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
	kernel_limits[n] = static_cast<int>( std::ceil( kernel_width*over_sampling ) );
	oversampled_dims[n] = static_cast<int>( std::ceil( over_sampling*dims[n] ) );
	kernel_step[n] = static_cast<int>( std::floor( kernel_table_steps/over_sampling ) );	   
    }
    
    hoNDArray< std::complex<T> > data_out;
    if (direction)
    {               
	data_out.create(oversampled_dims[0], oversampled_dims[1]);
    }
    else
    {
	data_out.create(npoints);
    }
    
    for (n = 0 ; n < npoints ; n++)
    {
	for (y=0;y<kernel_limits[1];y++)
	{	   
	    ky = kernel_table[kernel_positions[npoints + n]+y*kernel_step[1]];
	    o1 = oversampled_dims[0] * ( (grid_positions[npoints + n] + y + oversampled_dims[1]) % oversampled_dims[1] );
	    
	    for (x = 0 ; x < kernel_limits[0] ; x++)
	    {
		kx = ky*(kernel_table[kernel_positions[n]+x*kernel_step[0]]);
		o = (o1 + (grid_positions[n]+x+oversampled_dims[0])%oversampled_dims[0]);
		if (o > tmp_var){
		    tmp_var = o;
		}
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
    //std::cout << "tmp_var: " << tmp_var << std::endl;
    return data_out;
}



template <class T, unsigned int D> hoNDArray< std::complex<T> > Gridder<T,D>::convolution_3d(hoNDArray< std::complex<T> >& data_in, hoNDArray<T>* weight, int direction)
{
    //fixed_dims holds '1' if dimension should be skipped during gridding and zero if 0
    
    int n,x,y,z;
    T kx,ky,kz;
    int o2,o1,o;
    int kernel_limits[3];
    int kernel_step[3];
    int oversampled_dims[3];
    int ndim = 3;
    
    //std::cout << "Oversampled_dims in convolution" << std::endl;
    //std::cout << "noversampled_dims[0]>>1: " << (oversampled_dims[0]>>1) << std::endl;
    //std::cout << "noversampled_dims[1]>>1: " << (oversampled_dims[1]>>1) << std::endl;
    //std::cout << "noversampled_dims[2]>>1: " << (oversampled_dims[2]>>1) << std::endl;

    int npoints = kernel_positions.get_size(0);
    hoNDArray< std::complex<T> > weighted_data_in;
    
    //Apply weights
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
	kernel_limits[n] = static_cast<int>( std::ceil( kernel_width*over_sampling ) );
	oversampled_dims[n] = static_cast<int>( std::ceil( over_sampling*dims[n] ) );
	kernel_step[n] = static_cast<int>( std::floor( kernel_table_steps/over_sampling ) );   
    }

    //std::cout << "Oversampled_dims in convolution" << std::endl;
    //std::cout << "noversampled_dims[0]>>1: " << (oversampled_dims[0]>>1) << std::endl;
    //std::cout << "noversampled_dims[1]>>1: " << (oversampled_dims[1]>>1) << std::endl;
    //std::cout << "noversampled_dims[2]>>1: " << (oversampled_dims[2]>>1) << std::endl;

    
    hoNDArray< std::complex<T> > data_out;
    if (direction)
    {
	data_out.create(oversampled_dims[0], oversampled_dims[1], oversampled_dims[2]);
    }
    else
    {
	data_out.create(npoints);
    }
    
    for (n = 0 ; n < npoints ; n++)
    {
	for (z=0;z<kernel_limits[2];z++)
	{
	    kz = kernel_table[kernel_positions[npoints*2 + n]+z*kernel_step[2]];
	    o2 = oversampled_dims[1]*oversampled_dims[0]*((grid_positions[npoints*2 + n]+z+oversampled_dims[2])%oversampled_dims[2]);
	    
	    for (y=0;y<kernel_limits[1];y++)
	    {
		ky = kz * kernel_table[kernel_positions[npoints + n]+y*kernel_step[1]];
		o1 = o2 + oversampled_dims[0]*((grid_positions[npoints + n]+y+oversampled_dims[1])%oversampled_dims[1]);
		for (x = 0 ; x < kernel_limits[0] ; x++)
		{
		    kx = ky*(kernel_table[kernel_positions[n]+x*kernel_step[0]]);
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

	

template <class T, unsigned int D> hoNDArray<T> Gridder<T,D>::return_kernel_tables()
{
    calculate_kernel_tables();
    hoNDArray< T > kernel_table_return;
    kernel_table_return.create((int)kernel_width*kernel_table_steps+1);
    for (int i = 0 ; i < (int)kernel_width*kernel_table_steps+1 ; i++)
    {
	kernel_table_return[i] = kernel_table[i];
    }
    return kernel_table_return;
}



template <class T, unsigned int D> void Gridder<T,D>::calculate_kernel_tables()
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



template<class T, unsigned int D> void Gridder<T,D>::calculate_point_vectors(hoNDArray<typename reald<T,D>::Type>& kt_pos) //kt_pos are k-space coordinates
{
    int npoints = kt_pos.get_size(0);
    int ndim    = D;
    //std::cout << "npoints: " << npoints << "\nndim: " << ndim << std::endl;
 
    kernel_positions.create(npoints,ndim);
    grid_positions.create(npoints,ndim);
  
    T tmp;
    typename reald<T,D>::Type tmp_nd;

    for (int i = 0 ; i < npoints ; i++)
    {
	tmp_nd = kt_pos[i];
	for(int j = 0; j < ndim; j++)
	{
	    //tmp = kt_pos[npoints*j+i]+(dims[j]>>1) - kernel_width/2.0;
	    tmp = tmp_nd[j] + (dims[j]>>1) - kernel_width/2.0;
	    grid_positions[npoints*j+i] = static_cast<int>(std::round(tmp*over_sampling));
	    kernel_positions[npoints*j+i] = static_cast<int>(std::round(abs((tmp*over_sampling - grid_positions[npoints*j+i])*(kernel_table_steps/over_sampling))));
	}
    }
}



template <class T, unsigned int D> double Gridder<T,D>::bessi0(double x)
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



template <class T, unsigned int D> hoNDArray< std::complex<T> > Gridder<T,D>::calculate_deapodization_filter()
{
    std::cout << "Oversampled_dims in deapodization" << std::endl;
    std::cout << "Ooversampled_dims[0]: " << (oversampled_dims[0]) << std::endl;
    std::cout << "Ooversampled_dims[1]: " << (oversampled_dims[1]) << std::endl;
    std::cout << "Ooversampled_dims[2]: " << (oversampled_dims[2]) << std::endl;

    //std::cout << "Hi :)" << std::endl;
    int ndim = kernel_positions.get_size(1);

    hoNDArray< std::complex<T> > a;
    if (ndim==2)
    {
	a.create(oversampled_dims[0], oversampled_dims[1]);
    }
    else
    {
	a.create(oversampled_dims[0], oversampled_dims[1], oversampled_dims[2]);
    }

    for(int nn = 0; nn < a.get_number_of_elements(); nn++)
    {
	a(nn) = 0;
    }

    //write_nd_array< std::complex<T> >(&a, "/home/kh/data/images/gadget_output_a_empty.cplx");

  
    int kernel_table_center = ((static_cast<int>(kernel_table_steps*kernel_width))>>1);
  
    int zmin, zmax, ymin, ymax, xmin, xmax, zoffset, yoffset, xoffset;

    if (ndim == 3)
    {
	zmin = static_cast<int>(std::round(-std::floor(kernel_width/2)*over_sampling));
	zmax = static_cast<int>(std::round(std::floor(kernel_width/2)*over_sampling));
    }
    else
    {
	zmin = 0; zmax = 0;
    }
    if (ndim == 3)
    {
	zoffset = oversampled_dims[2]>>1;
    }
    else
    {
	zoffset = 0;
    }

    ymin = static_cast<int>(std::round(-std::floor(kernel_width/2)*over_sampling));
    ymax = static_cast<int>(std::round(std::floor(kernel_width/2)*over_sampling));
  
    yoffset = oversampled_dims[1]>>1;

    xmin = static_cast<int>(std::round(-std::floor(kernel_width/2)*over_sampling));
    xmax = static_cast<int>(std::round(std::floor(kernel_width/2)*over_sampling));

    xoffset = oversampled_dims[0]>>1;

    std::cout << "xmin: " << xmin << "\nxmax: " << xmax << "\nymin: " << ymin << "\nymax: " << ymax << "\nzmin: " << zmin << "\nzmax: " << zmax << "\nxoffset: " << xoffset << "\nyoffset: " << yoffset << "\nzoffset: " << zoffset << std::endl;
    std::cout << "ndim: " << ndim << "\noversampled_dims[0]>>1: " << (oversampled_dims[0]>>1) << std::endl;
    //std::cout << "Oversampled_dims in deapodization" << std::endl;
    //std::cout << "noversampled_dims[0]>>1: " << (oversampled_dims[0]>>1) << std::endl;
    //std::cout << "noversampled_dims[1]>>1: " << (oversampled_dims[1]>>1) << std::endl;
    //std::cout << "noversampled_dims[2]>>1: " << (oversampled_dims[2]>>1) << std::endl;
    T kz, ky, kx;
    int points = 0;
    int ix,iy,iz;
    for (int z = zmin; z <= zmax; z++)
    {
	iz = static_cast<int>(kernel_table_center+(z*kernel_table_steps/over_sampling));
	if (ndim == 3)
	{
	    kz = kernel_table[iz];
	}
	else
	{
	    kz = 1;
	}
	for (int y = ymin; y <= ymax; y++)
	{
	    iy = static_cast<int>(kernel_table_center+(y*kernel_table_steps/over_sampling));
	    ky = kernel_table[iy];
	    for (int x = xmin; x <= xmax; x++)
	    {	
		ix = static_cast<int>(kernel_table_center+(x*kernel_table_steps/over_sampling));
		kx = kernel_table[ix];
		a[(z+zoffset)*oversampled_dims[1]*oversampled_dims[0]+(y+yoffset)*oversampled_dims[0]+(x+xoffset)] = kx*ky*kz;
		
		//std::cout << "kx: " << kx << ", ky: " << ky << ", kz: " << kz << ",   pos: " << (z+zoffset)*oversampled_dims[1]*oversampled_dims[0]+(y+yoffset)*oversampled_dims[0]+(x+xoffset) << std::endl;
		
		points++;
	    }
	}
    }
    //Transformer<T> t;
    //t.K2I(a);

    std::cout << "ndim: " << ndim << std::endl;

    std::cout << "Test A" << std::endl;
    write_nd_array< std::complex<T> >(&a, "/home/kh/data/images/gadget_output_a_no_ifft.cplx");
    std::cout << "Test B" << std::endl;

    std::cout << "ndim: " << ndim << std::endl;

    if (ndim == 2)
	{
	    hoNDFFT<T>::instance()->ifft2c(a);
	}
    else
	{
	    std::cout << "Running line inside if for 3D deapodization before ifft" << std::endl;
	    hoNDFFT<T>::instance()->ifft(&a,0); 
	    std::cout << "After first ifft of deapodization" << std::endl;
	    hoNDFFT<T>::instance()->ifft(&a,1);
	    hoNDFFT<T>::instance()->ifft(&a,2);
	    write_nd_array< std::complex<T> >(&a, "/home/kh/data/images/gadget_output_a_in_fun.cplx");
	    std::cout << "Running line inside if for 3D deaodization ifft" << std::endl;
	}
	    std::cout << "Running line after deaodization ifft" << std::endl;
    


    //hoNDFFT<T>::instance()->ifft(&a,1);
    //if (ndim==3)   hoNDFFT<T>::instance()->ifft(&a,2);
  
    //write_nd_array< std::complex<T> >(&a, "/home/kh/data/images/gadget_output_a_in_fun.cplx");

    return a;
}


//template class Gridder<float,1>;
template class Gridder<float,2>;
template class Gridder<float,3>;
//template class Gridder<float,4>;

//template class Gridder<double,1>;
template class Gridder<double,2>;
template class Gridder<double,3>;
//template class Gridder<double,4>;

#undef SQR
