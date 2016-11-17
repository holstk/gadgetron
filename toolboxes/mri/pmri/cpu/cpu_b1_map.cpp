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

#include "hoNDArray_fileio.h"

#include <iostream>
#include <cmath>

using namespace std;

namespace Gadgetron{

    const int kernel_width = 7;

    static std::string save_path_b1 = "/mnt/scratch/karen/go/";
    static bool do_save_b1 = true;

    template<class REAL, unsigned int D> static void smooth_correlation_matrices( hoNDArray<complext<REAL> >*, hoNDArray<complext<REAL> >*);
    template<class T> static void correlation( hoNDArray<T> *in, hoNDArray<T> *out );
    template<class T> static void rss_normalize( hoNDArray<T> *in_out, unsigned int dim );

    template<class REAL> static void extract_csm(hoNDArray<complext<REAL> >*, hoNDArray<complext<REAL> >*, unsigned int, unsigned int, hoNDArray<complext<REAL> >* );


    		    
// ======================================================== //
// ====================== Main method ===================== //
// ======================================================== //


    template<class REAL, unsigned int D> void
    estimate_b1_map( hoNDArray<complext<REAL> > *data_in, hoNDArray<complext<REAL> > *data_out, int target_coils)
    {
	//** Initial verifications and parameter extraction**//

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
	
	vector<unsigned int> image_dims, dims_to_xform;
	unsigned int pixels_per_coil = 1;
  
	for( unsigned int i=0; i<D; i++ ){
	    image_dims.push_back(data_in->get_size(i)); //e.g. [176 176 176]
	    dims_to_xform.push_back(i); //e.g. [0 1 2]
	    pixels_per_coil *= data_in->get_size(i);
	}
  
	unsigned int ncoils = data_in->get_size(D);

	
	// ======================================================== //
	// ================== Copy of input data ================== //
	// ======================================================== //

	data_out->create(data_in->get_dimensions());
	for (unsigned int ii = 0; ii < data_out->get_number_of_elements(); ii++)
	    data_out->get_data_ptr()[ii] = data_in->get_data_ptr()[ii];
	/**
	if (target_coils_int == ncoils) {
	    data_out->create(data_in->get_dimensions(), data_in->begin());
	} else {
	    std::vector<size_t> odims = *(data_in->get_dimensions());
	    odims[D] = target_coils_int;
	    hoNDArray<complext<REAL> > *data_out = new hoNDArray<complext<REAL> >(&odims);
	    unsigned int elements_per_coil = data_in->get_number_of_elements()/ncoils;
	    for (unsigned int c = 0; c < target_coils_int; c++) {
		for (unsigned int i = 0; i < elements_per_coil; i++){
		    data_out->get_data_ptr()[i + c*elements_per_coil] = data_in->get_data_ptr()[i + c*elements_per_coil];
		}
	    }
	    ncoils = target_coils_int;
	}
	*/
	
	// ======================================================== //
	// =========== Normalize by the RSS of the coils ========== //
	// ======================================================== //

	rss_normalize( data_out, D );
	//* WRITE  
	/*if (do_save_b1){
	    hoNDArray<std::complex<REAL> > data_out_save(data_out->get_dimensions(), (std::complex<REAL>*) data_out->get_data_ptr());
	    std::string filename = save_path_b1 + "rss_norm_output.cplx";
            write_nd_array<std::complex<REAL> >(&data_out_save, filename.c_str());
	    }*/
	
	
	// ======================================================== //
	// ==== Correlation of all coil images with each other ==== //
	// ======================================================== //
	
	std::vector<size_t> dims_cor = *data_out->get_dimensions(); 
	dims_cor.push_back(ncoils); //Adding extra dimension so has D x num_coils x num_coils
	//std::cout << "Dims: " << dims_cor[0] << ", " << dims_cor[1] << ", " << dims_cor[2] << ", " << dims_cor[3] << std::endl;

	hoNDArray<complext<REAL> > *corrm = new hoNDArray<complext<REAL> >(&dims_cor);
	correlation( data_out, corrm );
	/* WRITE */ 
	/*if (do_save_b1){
	    hoNDArray<std::complex<REAL> > corrm_save(corrm->get_dimensions(), (std::complex<REAL>*) corrm->get_data_ptr());
	    std::string filename = save_path_b1 + "corrm.cplx";
            write_nd_array<std::complex<REAL> >(&corrm_save, filename.c_str());
	}*/

	

	// ======================================================== //
	// ============== Smooth correlation matrices ============= //
	// ======================================================== //

	hoNDArray<complext<REAL> > *_corrm_smooth = new hoNDArray<complext<REAL> >();
	_corrm_smooth->create(corrm->get_dimensions());
	hoNDArray<complext<REAL> > *corrm_smooth(_corrm_smooth);
	
	smooth_correlation_matrices<REAL,D>( corrm, corrm_smooth );
	/* WRITE */
	/*if (do_save_b1){
	    hoNDArray<std::complex<REAL> > corrm_smooth_save(corrm_smooth->get_dimensions(), (std::complex<REAL>*) corrm_smooth->get_data_ptr());
	    std::string filename = save_path_b1 + "corrm_smooth.cplx";
            write_nd_array<std::complex<REAL> >(&corrm_smooth_save, filename.c_str());
	}*/




	// ======================================================== //
	// ============ Calculate coil sensitivity maps =========== //
	// ======================================================== //

	vector<size_t> csm_dims;
	
	for( unsigned int i=0; i<corrm_smooth->get_number_of_dimensions()-1; i++ ){
	    csm_dims.push_back(corrm_smooth->get_size(i)); //So this also is D x num_coils x num_coils??
	}
	
	// Get the dominant eigenvector for each correlation matrix.
	hoNDArray<complext<REAL> > *csm = new hoNDArray<complext<REAL> >(&csm_dims);
		
	// Temporary buffer. TODO: use shared memory
	hoNDArray<complext<REAL> > *tmp_v = new hoNDArray<complext<REAL> >(&csm_dims); 

	std::cout << "csm dims = " << csm->get_number_of_dimensions() << std::endl;
	std::cout << "tmp_v dims = " << tmp_v->get_number_of_dimensions() << std::endl;

	extract_csm<REAL>( corrm_smooth, csm, ncoils, pixels_per_coil, tmp_v );

	/* WRITE */
	if (do_save_b1){
	    hoNDArray<std::complex<REAL> > csm_save(csm->get_dimensions(), (std::complex<REAL>*) csm->get_data_ptr());
	    std::string filename = save_path_b1 + "csm_pre.cplx";
            write_nd_array<std::complex<REAL> >(&csm_save, filename.c_str());
	}

	


	// ======================================================== //
	// ======= Set phase according to reference (coil 0) ====== //
	// ======================================================== //
	
	set_phase_reference( csm, ncoils, pixels_per_coil );



	// ======================================================== //
	// ======== Write coil sensitivity maps to output  ======== //
	// ======================================================== //

	for (unsigned int j = 0; j < csm->get_number_of_elements(); j++){
	    data_out->get_data_ptr()[j] = csm->get_data_ptr()[j];
	}

    }




// ======================================================== //
// =============== RSS in RSS norm function =============== //
// ======================================================== //

    template<class REAL, class T> static REAL
    _rss( unsigned int idx, hoNDArray<T> *in, unsigned int number_of_elements, unsigned int number_of_batches )
    {
	REAL rss = REAL(0);
    
	for( unsigned int i=0; i<number_of_batches; i++ ) 
	    rss += norm(in->get_data_ptr()[i*number_of_elements+idx]);
    
	rss = std::sqrt(rss); 
    
	return rss;
    }
  



// ======================================================== //
// ==================== Normalized RSS ==================== //
// ======================================================== //

    template<class T> static
    void rss_normalize( hoNDArray<T> *in_out, unsigned int dim )
    {
	unsigned int number_of_batches = in_out->get_size(dim); //ncoils
	unsigned int number_of_elements = in_out->get_number_of_elements()/number_of_batches; //samples per coil
    
	typedef typename realType<T>::Type REAL;

	for (unsigned int idx = 0; idx < number_of_elements; idx++){
	    REAL reciprocal_rss = 1/(_rss<REAL,T>(idx, in_out, number_of_elements, number_of_batches));
	    //rss is the root of SS of one complex element for all coils
      
	    for( unsigned int i=0; i<number_of_batches; i++ ) {
		T out = in_out->get_data_ptr()[i*number_of_elements+idx];
		out *= reciprocal_rss; // complex-scalar multiplication (element-wise operator)
		in_out->get_data_ptr()[i*number_of_elements+idx] = out; 
	    } 
	}
    }

  


// ======================================================== //
// ================ Bild correlation matrix =============== //
// ======================================================== //

    template<class T> static void correlation( hoNDArray<T> *in, hoNDArray<T> *out )
    {
	unsigned int number_of_coils = in->get_size(in->get_number_of_dimensions()-1);
	unsigned int number_of_elements = in->get_number_of_elements()/number_of_coils;

	for (unsigned int i = 0; i < number_of_elements; i++){
	    for (unsigned int c = 0; c < number_of_coils; c++){
		for( unsigned int cc=0; cc<c; cc++){
		    T tmp = in->get_data_ptr()[c*number_of_elements+i]*conj(in->get_data_ptr()[cc*number_of_elements+i]);
		    out->get_data_ptr()[(cc*number_of_coils+c)*number_of_elements+i] = tmp;
		    out->get_data_ptr()[(c*number_of_coils+cc)*number_of_elements+i] = conj(tmp);
		}
		T tmp = in->get_data_ptr()[c*number_of_elements+i];
		out->get_data_ptr()[(c*number_of_coils+c)*number_of_elements+i] = tmp*conj(tmp);
	    }
	}
    }
  



// ======================================================== //
// ==== Smooth correlation matrices by box filter (2D) ==== //
// ======================================================== //

    template<class REAL> static  void
    smooth_correlation_matrices_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<2>::Type image_dims )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=2; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}
	
	for (unsigned int batch = 0; batch < number_of_batches; batch++){
	    for (unsigned int idx = 0; idx < num_image_elements; idx++){  
		const intd2 co = idx_to_co<2>(idx, image_dims);
    
		const int x = co.vec[0];
		const int y = co.vec[1];
    
		const int size_x = image_dims.vec[0];
		const int size_y = image_dims.vec[1];
    
		const int half_width = kernel_width>>1; //3 for kernel = 7

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

				for (int ky = 0; ky < kernel_width; ky++){
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
		corrm_smooth->get_data_ptr()[batch*num_image_elements+idx] = scale*result;
	    }
	}
    }




// ======================================================== //
// ==== Smooth correlation matrices by box filter (3D) ==== //
// ======================================================== //

    template<class REAL> static  void
    smooth_correlation_matrices_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<3>::Type image_dims )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=3; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}
	
	for (unsigned int batch = 0; batch < number_of_batches; batch++){
	    for (unsigned int idx = 0; idx < num_image_elements; idx++){  
	        const intd<3>::Type co = idx_to_co<3>(idx, image_dims);

		const int x = co.vec[0];
		const int y = co.vec[1];
		const int z = co.vec[2];
    
		const int size_x = image_dims.vec[0];
		const int size_y = image_dims.vec[1];
		const int size_z = image_dims.vec[2];
    
		const int half_width = kernel_width>>1;

		const int yminus = y-half_width;
		const int xminus = x-half_width;
		const int zminus = z-half_width;
		const int yplus = y+half_width;
		const int xplus = x+half_width;
		const int zplus = z+half_width;
	    
		const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));
    
		complext<REAL> result = complext<REAL>(0);
   
		if( (zminus >=0) ){
		    if( zplus < size_z ){
			if( (yminus >=0) ){
			    if( yplus < size_y ){
				if( xminus >= 0 ){
				    if( xplus < size_x ){

					for (int kz = 0; kz < kernel_width; kz++){
					    for (int ky = 0; ky < kernel_width; ky++){
						for (int kx = 0; kx < kernel_width; kx++) {
						    
						    int cz = zminus+kz;
						    int cy = yminus+ky;
						    int cx = xminus+kx;
		
						    int source_offset = batch*num_image_elements + cz*size_y*size_x + cy*size_x + cx;
						    result += corrm->get_data_ptr()[source_offset];
						}
					    }
					}
				    }
				}
			    }
			}
		    }
		}
		corrm_smooth->get_data_ptr()[batch*num_image_elements+idx] = scale*result;
	    }
	}
    }




// ======================================================== //
// ============== Find minimum of two numbers ============= //
// ======================================================== //

    int _min( int A, int B ){
	return (A<B) ? A : B;
    }



// ======================================================== //
//  Smooth correlation matrices border by box filter (2D)   //
// ======================================================== //

    template<class REAL> static void
    smooth_correlation_matrices_border_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<2>::Type image_dims, unsigned int number_of_border_threads )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=2; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}

	for (unsigned int batch = 0; batch < number_of_batches; batch ++){
	    for (unsigned int idx = 0; idx < number_of_border_threads; idx++){

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
		for (int ky = 0; ky < kernel_width; ky++) {
		    for (int kx = 0; kx < kernel_width; kx++) {
	
			if( (yminus+ky >=0) ){
			    if( yminus+ky < size_y ){
				if( xminus+kx >= 0 ){
				    if( xminus+kx < size_x ){
		
					int source_offset = batch*num_image_elements + (yminus+ky)*size_x + (xminus+kx);
		
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


    

// ======================================================== //
//  Smooth correlation matrices border by box filter (3D)   //
// ======================================================== //

    template<class REAL> static void
    smooth_correlation_matrices_border_kernel( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *corrm_smooth, intd<3>::Type image_dims, unsigned int number_of_border_threads )
    {
	const int num_image_elements = prod(image_dims);

	unsigned int number_of_batches = 1;
	for( unsigned int i=3; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i);
	}

	for (unsigned int batch = 0; batch < number_of_batches; batch ++){
	    for (unsigned int idx = 0; idx < num_image_elements; idx++){

	    
		const int half_width = kernel_width>>1;

		const intd<3>::Type co = idx_to_co<3>(idx, image_dims);		
    
		const int x = co.vec[0];
		const int y = co.vec[1];
		const int z = co.vec[2];
    
		const int size_x = image_dims.vec[0];
		const int size_y = image_dims.vec[1];
		const int size_z = image_dims.vec[2];
    
		const int zminus = z-half_width;
		const int yminus = y-half_width;
		const int xminus = x-half_width;

		const REAL scale = REAL(1)/((REAL)(kernel_width*kernel_width));
    
		complext<REAL> result = complext<REAL>(0);

		bool do_edge_smoothing = false;

		if (z < half_width || z > size_z - half_width - 1)
		    do_edge_smoothing = true;
		else if (y < half_width || y > size_y - half_width - 1)
		    do_edge_smoothing = true;
		else if (x < half_width || x > size_x - half_width - 1)
		    do_edge_smoothing = true;

		    
		if (do_edge_smoothing){

		for (int kz = 0; kz < kernel_width; kz++) {
		    for (int ky = 0; ky < kernel_width; ky++) {
			for (int kx = 0; kx < kernel_width; kx++) {
			    
			    if( (zminus+kz >=0) ){
				if( zminus+kz < size_z ){
				    if( (yminus+ky >=0) ){
					if( yminus+ky < size_y ){
					    if( xminus+kx >= 0 ){
						if( xminus+kx < size_x ){
						    
						    int source_offset = batch*num_image_elements + (zminus+kz)*size_y*size_x + (yminus+ky)*size_x + (xminus+kx);						    
						    result += corrm->get_data_ptr()[source_offset];
						}
					    }
					}
				    }
				}
			    }
			}
		    }
		}
		corrm_smooth->get_data_ptr()[batch*num_image_elements+co_to_idx<3>(co,image_dims)] = scale*result;  

		}
	    }
	}
     }
    
    


// ======================================================== //
// ======= Main of smoothing the correlation matrix ======= //
// ======================================================== //

    template<class REAL, unsigned int D> static void 
    smooth_correlation_matrices( hoNDArray<complext<REAL> > * corrm, hoNDArray<complext<REAL> > * corrm_smooth )
    {
	typename intd<D>::Type image_dims;

	for( unsigned int i=0; i<D; i++ ){
	    image_dims.vec[i] = corrm->get_size(i);
	}
  
	unsigned int number_of_batches = 1;
  
	for( unsigned int i=D; i<corrm->get_number_of_dimensions(); i++ ){
	    number_of_batches *= corrm->get_size(i); //number_of_batches must be num_coils x num_coils
	}
  
	smooth_correlation_matrices_kernel<REAL> ( corrm, corrm_smooth, image_dims );

	/* WRITE */ 
	/*if (do_save_b1){
	    hoNDArray<std::complex<REAL> > corrm_smooth_save(corrm_smooth->get_dimensions(), (std::complex<REAL>*) corrm_smooth->get_data_ptr());
	    std::string filename = save_path_b1 + "corrm_smooth_middle.cplx";
            write_nd_array<std::complex<REAL> >(&corrm_smooth_save, filename.c_str());
	    }*/
  
	unsigned int number_of_border_threads = ((kernel_width>>1)<<1)*(sum(image_dims)-((kernel_width>>1)<<1));
	
	//** Smoothing the boarders **//
	
	smooth_correlation_matrices_border_kernel<REAL> ( corrm, corrm_smooth, image_dims, number_of_border_threads );
    }




// ======================================================== //
// ====================== Extract CSM ===================== //
// ======================================================== //

    template<class REAL> static void
    extract_csm( hoNDArray<complext<REAL> > *corrm, hoNDArray<complext<REAL> > *csm, unsigned int num_batches, unsigned int num_elements, hoNDArray<complext<REAL> > *tmp_v )
    {	
	for (unsigned int idx = 0; idx < num_elements; idx++){  

	    std::cout << "num_batches = " << num_batches << std::endl;
	    std::cout << "num_elements = " << num_elements << std::endl;
	    
	    const unsigned int iterations = 2;
	    
	    for( unsigned int c=0; c<num_batches; c++){
		csm->get_data_ptr()[c*num_elements+idx] = complext<REAL>(1);
	    }
	    
	    for( unsigned int it=0; it<iterations; it++ ){
		
		for( unsigned int c=0; c<num_batches; c++){
		    tmp_v->get_data_ptr()[c*num_elements+idx] = complext<REAL>(0);
		}
		
		for( unsigned int j=0; j<num_batches; j++){
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
    
 


// ======================================================== //
// ================== Set reference phase ================= //
// ======================================================== //

    template<class REAL>
    void set_phase_reference( hoNDArray<complext<REAL> > *csm, unsigned int num_coils, unsigned int num_elements )
    {
	for (unsigned int idx = 0; idx < num_elements; idx ++){
	    REAL angle = arg<REAL>(csm->get_data_ptr()[idx]); //Phase of the first coil
	    REAL sin_a = sin(angle);
	    REAL cos_a = cos(angle);
   
	    complext<REAL> tmp;
	    tmp.vec[0] = cos_a; tmp.vec[1] = sin_a;
	    tmp = conj(tmp);
	    
	    for( unsigned int c=0; c<num_coils; c++ ){
		complext<REAL> val = csm->get_data_ptr()[c*num_elements+idx];
		typedef complext<REAL> T;
		val = val*tmp;
		csm->get_data_ptr()[c*num_elements+idx] = val;
	    }
	} 
    }
   


    
    //** Template instantiation **//

    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,1>(cuNDArray<complext<float> >*, int);
    template void estimate_b1_map<float,2>(hoNDArray<complext<float> >*, hoNDArray<complext<float> >*, int);
    template void estimate_b1_map<float,3>(hoNDArray<complext<float> >*, hoNDArray<complext<float> >*, int);
    //template boost::shared_ptr< cuNDArray<complext<float> > > estimate_b1_map<float,4>(cuNDArray<complext<float> >*, int);

    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,1>(cuNDArray<complext<double> >*, int);
    template void estimate_b1_map<double,2>(hoNDArray<complext<double> >*, hoNDArray<complext<double> >*, int);
    template void estimate_b1_map<double,3>(hoNDArray<complext<double> >*, hoNDArray<complext<double> >*, int);
    //template EXPORTGPUPMRI boost::shared_ptr< cuNDArray<complext<double> > > estimate_b1_map<double,4>(cuNDArray<complext<double> >*, int);
}