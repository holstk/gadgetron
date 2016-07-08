#pragma once

#include "hoNDArray.h"
#include "vector_td.h"

#include <boost/shared_ptr.hpp>
#include "KaiserBessel_operators.h"
#include "complext.h"


namespace Gadgetron{
    
    template< class T, unsigned int D > class NFFT_plan
    {
	
    public:

	NFFT_plan(){};

	NFFT_plan( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W ){};
	
	//virtual ~NFFT_plan() = 0;

	virtual void test_fun(int i) = 0;

	virtual void wipe( int NFFT_wipe_mode ) = 0;
	
	virtual	void setup( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W, int device = -1 ) = 0;
	
	virtual void preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory, int NFFT_prep_mode ) = 0;

	virtual void compute( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, hoNDArray<T> *dcw, int NFFT_comp_mode ) = 0;
	
	virtual void mult_MH_M( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, hoNDArray<T> *dcw, std::vector<size_t> halfway_dims ) = 0;
		
	virtual void convolve( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, hoNDArray<T> *dcw, int NFFT_conv_mode, bool accumulate = false ) = 0;

	virtual void fft( hoNDArray<complext<T> > *data, int NFFT_fft_mode, bool do_scale = true ) = 0;
  
	virtual void deapodize( hoNDArray<complext<T> > *image, bool fourier_domain = false) = 0;
	
	virtual inline typename uint64d<D>::Type get_matrix_size() = 0;

	virtual inline typename uint64d<D>::Type get_matrix_size_os() = 0;

	virtual inline T get_W() = 0;
	
	virtual inline bool is_setup() = 0;
	
    };

    // Pure virtual class to cause compile errors if you try to use NFFT with double and atomics
    // - since this is not supported on the device
    //template< unsigned int D> class EXPORTGPUNFFT cuNFFT_plan<double,D,true>{ 
    //	virtual void atomics_not_supported_for_type_double() = 0; };

    
}
