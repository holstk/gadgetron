#include "cpu_b1_map.h"
//#include "hoNDArray_operators.h"
#include "hoNDArray_elemwise.h"
#include "vector_td_utilities.h"
#include "real_utilities.h"
//#include "real_utilities_device.h"
#include "complext.h"
//#include "check_CUDA.h"
//#include "cudaDeviceManager.h"
#include "setup_grid.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace Gadgetron{

    const int kernel_width = 7;

    template<class REAL, unsigned int D> static void smooth_correlation_matrices( hoNDArray<complext<REAL> >*, hoNDArray<complext<REAL> >*);
    template<class REAL> static void extract_csm( hoNDArray<complext<REAL> >*, hoNDArray<complext<REAL> >*, unsigned int, unsigned int);
    //template<class REAL> static void set_phase_reference( hoNDArray<complext<REAL> >*, unsigned int, unsigned int);
    template<class T> static void find_stride( hoNDArray<T> *in, unsigned int dim, unsigned int *stride, std::vector<size_t> *dims );
    template<class T> static void correlation( hoNDArray<T> *in, hoNDArray<T> *out );
    template<class T> static void rss_normalize( hoNDArray<T> *in_out, unsigned int dim );

    template<class REAL> static void extract_csm_kernel(hoNDArray<complext<REAL> >*, hoNDArray<complext<REAL> >*, unsigned int, unsigned int, hoNDArray<complext<REAL> >* );

    //
    // Main method
    //

    template<class REAL, unsigned int D> void
    estimate_b1_map( hoNDArray<complext<REAL> > *data_in, hoNDArray<complext<REAL> > *data_out, int target_coils)
    {
	//std::cout << "csm: beg of calc b1 map" << std::endl;
	//std::cout << "Dims data_in: " << data_in->get_size(0) << ", " << data_in->get_size(1) << ", " << data_in->get_size(2)  << std::endl;
	//std::cout << "csm: target_coils: " << target_coils << std::endl;

	if( data_in->get_number_of_dimensions() < 2 ){
	    throw std::runtime_error("estimate_b1_map:: dimensionality mismatch.");
	}

	if( data_in->get_number_of_dimensions()-1 != D ){
	    throw std::runtime_error("estimate_b1_map:: dimensionality mismatch.");
	}

	int target_coils_int = 0;
	if ((target_coils <= 0) || (target_coils > data_in->get_size(D))) {
	    target_coils_int = data_in->get_size(D);
	} else {
	    target_coils_int = target_coils;
	}
	//std::cout << "csm: set target_coils" << std::endl;
	
	vector<unsigned int> image_dims, dims_to_xform;
	unsigned int pixels_per_coil = 1;
  
	for( unsigned int i=0; i<D; i++ ){
	    image_dims.push_back(data_in->get_size(i));
	    dims_to_xform.push_back(i);
	    pixels_per_coil *= data_in->get_size(i);
	}
	//std::cout << "csm: pix_per_coil: " << pixels_per_coil << std::endl;
  
	unsigned int ncoils = data_in->get_size(D);
	
	// Make a copy of input data, but only the target coils
	
	//std::cout << "ncoils: " << ncoils << ", target_coils_int: " << target_coils_int << std::endl;

	if (target_coils_int == ncoils) {
	    //hoNDArray<complext<REAL> > *data_out = new hoNDArray<complext<REAL> >(*data_in);
	    data_out->create(data_in->get_dimensions(), data_in->get_data_ptr());
	    //data_out = hoNDArray<complext<REAL> >(*_data_out);
	    //std::cout << "csm: in if where coils is all coils" << std::endl;
	} else {
	    //std::cout << "csm: in if where coils is not all coils" << std::endl;
	    std::vector<size_t> odims = *(data_in->get_dimensions());
	    odims[D] = target_coils_int;
	    hoNDArray<complext<REAL> > *data_out = new hoNDArray<complext<REAL> >(&odims);
	    //data_out = hoNDArray<complext<REAL> >(&odims);

	    //Now copy one coil at a time
	    unsigned int elements_per_coil = data_in->get_number_of_elements()/ncoils;
	    for (unsigned int c = 0; c < target_coils_int; c++) {
		for (unsigned int i = 0; i < elements_per_coil; i++){
		    data_out->get_data_ptr()[i + c*elements_per_coil] = data_in->get_data_ptr()[i + c*elements_per_coil];
		}
	    }
	    ncoils = target_coils_int;
	}
	//std::cout << "csm: make data copy" << std::endl;
	//std::cout << "Dims data_out: " << data_out->get_size(0) << ", " << data_out->get_size(1) << ", " << data_out->get_size(2)  << std::endl;
	
	// Normalize by the RSS of the coils
	rss_normalize( data_out, D );
	//std::cout << "csm: after rss_normalize" << std::endl;
	//std::cout << "Dims data_out: " << data_out->get_size(0) << ", " << data_out->get_size(1) << ", " << data_out->get_size(2)  << std::endl;
	
	std::vector<size_t> dims_cor = *data_out->get_dimensions(); 
	dims_cor.push_back(ncoils); //Adding extra dimension so has D x num_coils x num_coils
	//std::cout << "Dims: " << dims_cor[0] << ", " << dims_cor[1] << ", " << dims_cor[2] << ", " << dims_cor[3] << std::endl;
	//std::cout << "csm: correlation: after dims pushback" << std::endl;
        //out->create(&dims_cor);
	//std::cout << "csm: correlation: after creating out" << std::endl;

	// Now calculate the correlation matrices
	hoNDArray<complext<REAL> > *corrm = new hoNDArray<complext<REAL> >(&dims_cor);
	//corrm->create(&dims_cor);
	correlation( data_out, corrm );
	//data_out.reset();
	//std::cout << "csm: after correlation" << std::endl;
	
	// Smooth (onto copy of corrm)
	hoNDArray<complext<REAL> > *_corrm_smooth = new hoNDArray<complext<REAL> >();
	_corrm_smooth->create(corrm->get_dimensions());
	hoNDArray<complext<REAL> > *corrm_smooth(_corrm_smooth);
	//std::cout << "csm: after making corr smooth arrays" << std::endl;
	
	smooth_correlation_matrices<REAL,D>( corrm, corrm_smooth );
	//corrm.reset();
	//std::cout << "csm: corr smooth" << std::endl;

	
	vector<size_t> csm_dims;
	
	for( unsigned int i=0; i<corrm_smooth->get_number_of_dimensions()-1; i++ ){
	    csm_dims.push_back(corrm_smooth->get_size(i)); //So this also is D x num_coils x num_coils??
	}
	
	// Get the dominant eigenvector for each correlation matrix.
	hoNDArray<complext<REAL> > *csm = new hoNDArray<complext<REAL> >(&csm_dims);
	// Allocate output
	//csm->create(&image_dims);
	//std::cout << "After creating csm" << std::endl;
		
	// Temporary buffer. TODO: use shared memory
	hoNDArray<complext<REAL> > *tmp_v = new hoNDArray<complext<REAL> >(&csm_dims); 
	//tmp_v->create(&image_dims);
	//std::cout << "After creating tmp_v" << std::endl;

	extract_csm_kernel<REAL>( corrm_smooth, csm, ncoils, pixels_per_coil, tmp_v );
	//extract_csm<REAL>( corrm_smooth, csm, ncoils, pixels_per_coil );
	//corrm_smooth.reset();
	//std::cout << "csm: after extract csm" << std::endl;
	
	// Set phase according to reference (coil 0)
	set_phase_reference( csm, ncoils, pixels_per_coil );
	//std::cout << "csm: after set phase ref" << std::endl;
    }

    
    template<class T> static void find_stride( hoNDArray<T > *in, unsigned int dim,
					       unsigned int *stride, std::vector<size_t> *dims )
    {
	*stride = 1;
	for( unsigned int i=0; i<in->get_number_of_dimensions(); i++ ){
	    if( i != dim )
		dims->push_back(in->get_size(i));
	    if( i < dim )
		*stride *= in->get_size(i);
	}
    }
    
    template<class REAL, class T> static REAL
    _rss( unsigned int idx, hoNDArray<T> *in, unsigned int stride, unsigned int number_of_batches )
    {
	unsigned int in_idx = (idx/stride)*stride*number_of_batches+(idx%stride);
	REAL rss = REAL(0);
    
	for( unsigned int i=0; i<number_of_batches; i++ ) 
	    rss += norm(in->get_data_ptr()[i*stride+in_idx]);
    
	rss = std::sqrt(rss); 
    
	return rss;
    }
  
    // Normalized RSS
    template<class T> static
    void rss_normalize( hoNDArray<T> *in_out, unsigned int dim )
    {
	unsigned int number_of_batches = in_out->get_size(dim);
	unsigned int number_of_elements = in_out->get_number_of_elements()/number_of_batches;
    

	// Find element stride
	unsigned int stride; std::vector<size_t> dims;
	find_stride<T>( in_out, dim, &stride, &dims );
	
	typedef typename realType<T>::Type REAL;

	for (unsigned int idx = 0; idx < number_of_elements; idx++){
	    REAL reciprocal_rss = 1/(_rss<REAL,T>(idx, in_out, stride, number_of_batches));
      
	    unsigned int in_idx = (idx/stride)*stride*number_of_batches+(idx%stride);
      
	    for( unsigned int i=0; i<number_of_batches; i++ ) {
		T out = in_out->get_data_ptr()[i*stride+in_idx];
		out *= reciprocal_rss; // complex-scalar multiplication (element-wise operator)
		in_out->get_data_ptr()[i*stride+in_idx] = out; 
	    } 
	}
    }

  
    // Build correlation matrix
    template<class T> static void correlation( hoNDArray<T> *in, hoNDArray<T> *out )
    {
	//std::cout << "csm: correlation: beg" << std::endl;

	//typedef typename realType<T>::Type REAL;

	unsigned int number_of_coils = in->get_size(in->get_number_of_dimensions()-1);
	unsigned int number_of_elements = in->get_number_of_elements()/number_of_coils;
	//std::cout << "csm: correlation: after n coils and n elem" << std::endl;
	//std::cout << "out n elem: " << out->get_number_of_elements() << std::endl;

	//std::vector<size_t> dims = *in->get_dimensions(); 
	//std::cout << "csm: correlation: after dims" << std::endl;
	//dims.push_back(number_of_coils); //Adding extra dimension so has D x num_coils x num_coils
	//std::cout << "Dims: " << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << std::endl;
	//std::cout << "csm: correlation: after dims pushback" << std::endl;
        //out->create(*dims);
	//std::cout << "csm: correlation: after creating out" << std::endl;

	//std::cout << "n elem, n coil: " << number_of_elements << ", " << number_of_coils << std::endl;
	for (unsigned int i = 0; i < number_of_elements; i++){
	    for (unsigned int c = 0; c < number_of_coils; c++){
		for( unsigned int cc=0; cc<c; cc++){
		    T tmp = in->get_data_ptr()[c*number_of_elements+i]*conj(in->get_data_ptr()[cc*number_of_elements+i]);
		    out->get_data_ptr()[(cc*number_of_coils+c)*number_of_elements+i] = tmp;
		    out->get_data_ptr()[(c*number_of_coils+cc)*number_of_elements+i] = conj(tmp);
		    //std::cout << "csm: correlation: in second loop. i, c, cc: " << i << ", " << c << ", " << cc << std::endl;
		}
		T tmp = in->get_data_ptr()[c*number_of_elements+i];
		out->get_data_ptr()[(c*number_of_coils+c)*number_of_elements+i] = tmp*conj(tmp);
		//std::cout << "csm: correlation: in second loop. i, c: " << i << ", " << c << std::endl;
	    }
	}
    }

    /**
     // Smooth correlation matrices by box filter (1D)
     template<class REAL> __global__ static void
     smooth_correlation_matrices_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<1>::Type image_dims )
     {
     const int idx = blockIdx.x*blockDim.x + threadIdx.x;
     const int batch = blockIdx.y;

     const int num_image_elements = prod(image_dims);

     if( idx < num_image_elements ){
    
     const int co = idx;    
     const int x = co;
    
     const int size_x = image_dims.vec[0];
    
     const REAL scale = REAL(1)/((REAL)kernel_width);
    
     complext<REAL> result = complext<REAL>(0);
    
     for (int kx = 0; kx < kernel_width; kx++) {
      
     if ((x-(kernel_width>>1)+kx) >= 0 &&
     (x-(kernel_width>>1)+kx) < size_x)
     {	    
     int source_offset = 
     batch*num_image_elements +
     (x-(kernel_width>>1)+kx);
	  
     result += corrm[source_offset];
     }
     }
     corrm_smooth[batch*num_image_elements+idx] = scale*result;
     }
     }
    */

    // Smooth correlation matrices by box filter (2D)
    template<class REAL> static  void
    smooth_correlation_matrices_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<2>::Type image_dims )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=2; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}
	
	//std::cout << "csm: smooth_cor_kern: im elem, batches: " << num_image_elements << ", " << number_of_batches << std::endl;
	

	for (unsigned int batch = 0; batch < number_of_batches; batch++){
	    //std::cout << "csm: smooth_cor_kern: in for loop" << std::endl;
	    for (unsigned int idx = 0; idx < num_image_elements; idx++){  
		const intd2 co = idx_to_co<2>(idx, image_dims);
    
		const int x = co.vec[0];
		const int y = co.vec[1];
    
		const int size_x = image_dims.vec[0];
		const int size_y = image_dims.vec[1];
    
		const int half_width = kernel_width>>1;

		const int yminus = y-half_width;
		const int xminus = x-half_width;
		const int yplus = y+half_width;
		const int xplus = x+half_width;
	    
		const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));
    
		complext<REAL> result = complext<REAL>(0);
   
		if( (yminus >=0) ){
		    if( yplus < size_y ){
			if( xminus >= 0 ){
			    if( xplus < size_x ){

#pragma unroll
				for (int ky = 0; ky < kernel_width; ky++){
#pragma unroll
				    for (int kx = 0; kx < kernel_width; kx++) {
		
					int cy = yminus+ky;
					int cx = xminus+kx;
		
					int source_offset = batch*num_image_elements + cy*size_x + cx;
					result += corrm->get_data_ptr()[source_offset];
				    }
				}
			    }
			}
		    }
		}
		//if (idx = )
		//std::cout << "idx, batch: " << idx << ", " << batch << std::endl;
		corrm_smooth->get_data_ptr()[batch*num_image_elements+idx] = scale*result;
	    }
	}
    }

    /**
     // Smooth correlation matrices by box filter (3D)
     template<class REAL> __global__ static  void
     smooth_correlation_matrices_kernel( const  complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<3>::Type image_dims )
     {
     const int idx = blockIdx.x*blockDim.x + threadIdx.x;
     const int batch = blockIdx.y;

     const int num_image_elements = prod(image_dims);

     if( idx < num_image_elements ){
    
     const intd3 co = idx_to_co<3>(idx, image_dims);
    
     const int x = co.vec[0];
     const int y = co.vec[1];
     const int z = co.vec[2];
    
     const int size_x = image_dims.vec[0];
     const int size_y = image_dims.vec[1];
     const int size_z = image_dims.vec[2];
    
     const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width*kernel_width));
    
     complext<REAL> result = complext<REAL>(0);
    
     for (int kz = 0; kz < kernel_width; kz++) {
     for (int ky = 0; ky < kernel_width; ky++) {
     for (int kx = 0; kx < kernel_width; kx++) {
	
     if ((z-(kernel_width>>1)+kz) >= 0 &&
     (z-(kernel_width>>1)+kz) < size_z &&
     (y-(kernel_width>>1)+ky) >= 0 &&
     (y-(kernel_width>>1)+ky) < size_y &&
     (x-(kernel_width>>1)+kx) >= 0 &&
     (x-(kernel_width>>1)+kx) < size_x) 
     {	    
     int source_offset = 
     batch*num_image_elements +
     (z-(kernel_width>>1)+kz)*size_x*size_y +
     (y-(kernel_width>>1)+ky)*size_x +
     (x-(kernel_width>>1)+kx);
	    
     result += corrm[source_offset];
     }
     }
     }
     }
     corrm_smooth[batch*num_image_elements+idx] = scale*result;
     }
     }

     // Smooth correlation matrices by box filter (3D)
     template<class REAL> __global__ static void
     smooth_correlation_matrices_kernel( const complext<REAL> * __restrict__ corrm, complext<REAL> * __restrict__ corrm_smooth, intd<4>::Type image_dims )
     {
     const int idx = blockIdx.x*blockDim.x + threadIdx.x;
     const int batch = blockIdx.y;

     const int num_image_elements = prod(image_dims);

     if( idx < num_image_elements ){
    
     const intd4 co = idx_to_co<4>(idx, image_dims);
    
     const int x = co.vec[0];
     const int y = co.vec[1];
     const int z = co.vec[2];
     const int w = co.vec[3];
    
     const int size_x = image_dims.vec[0];
     const int size_y = image_dims.vec[1];
     const int size_z = image_dims.vec[2];    
     const int size_w = image_dims.vec[3];
    
     const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width*kernel_width*kernel_width));
    
     complext<REAL> result = complext<REAL>(0);
    
     for (int kw = 0; kw < kernel_width; kw++) {
     for (int kz = 0; kz < kernel_width; kz++) {
     for (int ky = 0; ky < kernel_width; ky++) {
     for (int kx = 0; kx < kernel_width; kx++) {
	
     if ((w-(kernel_width>>1)+kw) >= 0 &&
     (w-(kernel_width>>1)+kw) < size_w &&
     (z-(kernel_width>>1)+kz) >= 0 &&
     (z-(kernel_width>>1)+kz) < size_z &&
     (y-(kernel_width>>1)+ky) >= 0 &&
     (y-(kernel_width>>1)+ky) < size_y &&
     (x-(kernel_width>>1)+kx) >= 0 &&
     (x-(kernel_width>>1)+kx) < size_x) 
     {	    
     int source_offset = 
     batch*num_image_elements +
     (w-(kernel_width>>1)+kw)*size_x*size_y*size_z +
     (z-(kernel_width>>1)+kz)*size_x*size_y +
     (y-(kernel_width>>1)+ky)*size_x +
     (x-(kernel_width>>1)+kx);
	    
     result += corrm[source_offset];
     }
     }
     }
     }
     }
     corrm_smooth[batch*num_image_elements+idx] = scale*result;
     }
     }
    */

    int _min( int A, int B ){
	return (A<B) ? A : B;
    }

    // Smooth correlation matrices border by box filter (2D)
    template<class REAL> static void
    smooth_correlation_matrices_border_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<2>::Type image_dims, unsigned int number_of_border_threads )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=2; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}

	for (unsigned int batch = 0; batch < number_of_batches; batch ++){
	    for (unsigned int idx = 0; idx < num_image_elements; idx++){
	    
		intd2 co;
		const int half_width = kernel_width>>1;

		co.vec[1] = idx/image_dims.vec[0];
		co.vec[1] = _min(co.vec[1], half_width );
	    
		if( co.vec[1] == half_width ){
		    int new_idx = idx-half_width*image_dims.vec[0];
		    int num_skips = new_idx/half_width;
		    int rows_offset = _min(num_skips>>1, image_dims.vec[1]-(half_width<<1) );
		    co.vec[1] += rows_offset;

		    if( co.vec[1] == (half_width + image_dims.vec[1]-(half_width<<1)) ){
			new_idx -= ((image_dims.vec[1]-(half_width<<1))*(half_width<<1));
			co.vec[1] += (new_idx / image_dims.vec[0]);
			co.vec[0] = (new_idx % image_dims.vec[0]);
		    }
		    else{
			co.vec[0] = (num_skips%2)*(image_dims.vec[0]-half_width) + (new_idx%half_width);
		    }
		}
		else{
		    co.vec[0] = idx%image_dims.vec[0];
		}
    
		const int x = co.vec[0];
		const int y = co.vec[1];
    
		const int size_x = image_dims.vec[0];
		const int size_y = image_dims.vec[1];
    
		const int yminus = y-half_width;
		const int xminus = x-half_width;

		const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));
    
		complext<REAL> result = complext<REAL>(0);
#pragma unroll
		for (int ky = 0; ky < kernel_width; ky++) {
#pragma unroll
		    for (int kx = 0; kx < kernel_width; kx++) {
	
			if( (yminus+ky >=0) ){
			    if( yminus+ky < size_y ){
				if( xminus+kx >= 0 ){
				    if( xminus+kx < size_x ){
		
					int source_offset = batch*num_image_elements + (xminus+kx);
		
					result += corrm->get_data_ptr()[source_offset];
				    }
				}
			    }
			}
		    }
		}
		corrm_smooth->get_data_ptr()[batch*num_image_elements+co_to_idx<2>(co,image_dims)] = scale*result;  
	    }
	}
    }
    

    template<class REAL, unsigned int D> static void
    smooth_correlation_matrices( hoNDArray<complext<REAL> > * corrm, hoNDArray<complext<REAL> > * corrm_smooth )
    {
	//std::cout << "csm: smooth_correlation_matrices beg" << std::endl;
	typename intd<D>::Type image_dims;

	for( unsigned int i=0; i<D; i++ ){
	    image_dims.vec[i] = corrm->get_size(i);
	}
	//std::cout << "csm: smooth_correlation_matrices: after image_dims" << std::endl;
  
	unsigned int number_of_batches = 1;
  
	for( unsigned int i=D; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i); //number_of_batches must be number_of_elements (same in image_dims) x num_coils x num_coils
	}
	//std::cout << "csm: smooth_correlation_matrices: after num_batches" << std::endl;
  
	smooth_correlation_matrices_kernel<REAL> ( corrm, corrm_smooth, image_dims );
	//std::cout << "After smooth_correlation_matrices_kernel" << std::endl;
  
	unsigned int number_of_border_threads = ((kernel_width>>1)<<1)*(sum(image_dims)-((kernel_width>>1)<<1));
	
	smooth_correlation_matrices_border_kernel<REAL> ( corrm, corrm_smooth, image_dims, number_of_border_threads );
	//std::cout << "After smooth_correlation_matrices_boarder_kernel" << std::endl;
    }


    // Extract CSM
    template<class REAL> static void
    extract_csm_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *csm, unsigned int num_batches, unsigned int num_elements, hoNDArray<complext<REAL> > *tmp_v )
    {	
	for (unsigned int idx = 0; idx < num_elements; idx++){  
	    
	    const unsigned int iterations = 2;
	    
	    for( unsigned int c=0; c<num_batches; c++){
		csm->get_data_ptr()[c*num_elements+idx] = complext<REAL>(1);
	    }
	    
	    for( unsigned int it=0; it<iterations; it++ ){
		
		for( unsigned int c=0; c<num_batches; c++){
		    tmp_v->get_data_ptr()[c*num_elements+idx] = complext<REAL>(0);
		}
		
		for( unsigned j=0; j<num_batches; j++){
		    for( unsigned int k=0; k<num_batches; k++){
			typedef complext<REAL> T;
			tmp_v->get_data_ptr()[j*num_elements+idx] += corrm->get_data_ptr()[(k*num_batches+j)*num_elements+idx]*csm->get_data_ptr()[k*num_elements+idx];
		    }
		}
		
		REAL tmp = REAL(0);
		
		for (unsigned int c=0; c<num_batches; c++){
		    tmp += norm(tmp_v->get_data_ptr()[c*num_elements+idx]);
		}
		
		tmp = 1/std::sqrt(tmp);
		
		for (unsigned int c=0; c<num_batches; c++){
		    complext<REAL> res = tmp*tmp_v->get_data_ptr()[c*num_elements+idx];
		    csm->get_data_ptr()[c*num_elements+idx] = res;
		}
	    }
	}
    }
    
    // Extract CSM
    template<class REAL> static void
    extract_csm(hoNDArray<complext<REAL> > *corrm_in, hoNDArray<complext<REAL> > *out, unsigned int number_of_batches, unsigned int number_of_elements )
    {
	
	
	//if( out != 0x0 && tmp_v != 0x0 )
	//  extract_csm_kernel<REAL>( corrm_in, out, number_of_batches, number_of_elements, tmp_v );
	
	//delete tmp_v;
    }

    
    // Set refence phase
    void set_phase_reference( hoNDArray<complext<float> > *csm, unsigned int num_coils, unsigned int num_elements )
    {
	//const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	//if( idx < num_elements ){
	for (unsigned int idx = 0; idx < num_elements; idx ++){
	    float angle = arg<float>(csm->get_data_ptr()[idx]); //Phase of the first coil
	    float sin_a, cos_a; 
	    sincosf( angle, &sin_a, &cos_a );
	    
	    complext<float> tmp;
	    tmp.vec[0] = cos_a; tmp.vec[1] = sin_a;
	    tmp = conj(tmp);
	    
	    for( unsigned int c=0; c<num_coils; c++ ){
		complext<float> val = csm->get_data_ptr()[c*num_elements+idx];
		typedef complext<float> T;
		val = val*tmp;
		csm->get_data_ptr()[c*num_elements+idx] = val;
	    }
	} 
    }
    
    // Set refence phase
    void set_phase_reference( hoNDArray<complext<double> > *csm, unsigned int num_coils, unsigned int num_elements )
    {
	//const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	//if( idx < num_elements ){
	for (unsigned int idx = 0; idx < num_elements; idx ++){
	    double angle = arg<double>(csm->get_data_ptr()[idx]); //Phase of the first coil
	    double sin_a, cos_a; 
	    sincos( angle, &sin_a, &cos_a );
	    
	    complext<double> tmp;
	    tmp.vec[0] = cos_a; tmp.vec[1] = sin_a;
	    tmp = conj(tmp);
	    
	    for( unsigned int c=0; c<num_coils; c++ ){
		complext<double> val = csm->get_data_ptr()[c*num_elements+idx];
		typedef complext<double> T;
		val = val*tmp;
		csm->get_data_ptr()[c*num_elements+idx] = val;
	    }
	} 
    }
    


    //
    // Template instantiation
    //

    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,1>(cuNDArray<complext<float> >*, int);
    template void estimate_b1_map<float,2>(hoNDArray<complext<float> >*, hoNDArray<complext<float> >*, int);
    //template boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,3>(cuNDArray<complext<float> >*, int);
    //template boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,4>(cuNDArray<complext<float> >*, int);

    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,1>(cuNDArray<complext<double> >*, int);
    template void estimate_b1_map<double,2>(hoNDArray<complext<double> >*, hoNDArray<complext<double> >*, int);
    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,3>(cuNDArray<complext<double> >*, int);
    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,4>(cuNDArray<complext<double> >*, int);
}
