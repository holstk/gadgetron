#pragma once

#include "hoNDArray.h"
#include "vector_td.h"
#include "complext.h"
//#include "gpunfft_export.h"

//#include <thrust/device_vector.h>
#include <boost/shared_ptr.hpp>
#include "KaiserBessel_operators.h"
#include "../NFFT.h"
//#include "hoNDArray_reductions.h"


//template<class T, unsigned int D> struct _convolve_NFFT_NC2C;

namespace Gadgetron{
    
    template< class T, unsigned int D > class hoNFFT_plan : public NFFT_plan<T, D>
    {
	
    public: // Main interface
	
	/** 
	    Default constructor
	*/
	
	hoNFFT_plan();
	
	/**
	   Constructor defining the required NFFT parameters.
	   \param matrix_size the matrix size to use for the NFFT. Define as a multiple of 32.
	   \param matrix_size_os intermediate oversampled matrix size. Define as a multiple of 32.
	   The ratio between matrix_size_os and matrix_size define the oversampling ratio for the NFFT implementation.
	   Use an oversampling ratio between 1 and 2. The higher ratio the better quality results, 
	   however at the cost of increased execution times. 
	   \param W the concolution window size used in the NFFT implementation. 
	   The larger W the better quality at the cost of increased runtime.
	   \param device the device (GPU id) to use for the NFFT computation. 
	   The default value of -1 indicates that the currently active device is used.
	*/
	hoNFFT_plan( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W ); 
	
	/**
	   Destructor
	*/
	virtual ~hoNFFT_plan();
	
	
	virtual void test_fun(int i){
	    std::cout << "i = " << i << std::endl;
	};
	
	
	//Enum to specify the desired mode for cleaning up when using the wipe() method.
	
	/**
	enum NFFT_wipe_mode { 
	    NFFT_WIPE_ALL, //**< delete all internal memory. 
	    NFFT_WIPE_PREPROCESSING ///**< delete internal memory holding the preprocessing data structures. 
	};
	*/
	
	
        //Clear internal storage
        //\param mode enum defining the wipe mode
	
	virtual void wipe( int NFFT_wipe_mode ); //NFFT_WIPE_ALL = 1, NFFT_WIPE_PREPROCESSING = 2
	
	
        //Setup the plan. Please see the constructor taking similar arguments for a parameter description.
	
	
	virtual void setup( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W, int device = -1);
	
	
	
	//Enum to specify the preprocessing mode.
	
	/**
	enum NFFT_prep_mode { 
	    NFFT_PREP_C2NC, //**< preprocess to perform a Cartesian to non-Cartesian NFFT. 
	    NFFT_PREP_NC2C, //**< preprocess to perform a non-Cartesian to Cartesian NFFT. 
	    NFFT_PREP_ALL //**< preprocess to perform NFFTs in both directions. 
	};
	*/

	
	//Perform NFFT preprocessing for a given trajectory.
	//\param trajectory the NFFT non-Cartesian trajectory normalized to the range [-1/2;1/2]. 
	//\param mode enum specifying the preprocessing mode
	
	virtual void preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory, int NFFT_prep_mode ); //NFFT_PREP_C2NC = 1, NFFT_PREP_NC2C = 2, NFFT_PREP_ALL = 4
	//virtual void preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory, NFFT_prep_mode mode );

	

	//Enum defining the desired NFFT operation
    
	/**
	enum NFFT_comp_mode { 
	    NFFT_FORWARDS_C2NC, //**< forwards NFFT Cartesian to non-Cartesian. 
	    NFFT_FORWARDS_NC2C, //**< forwards NFFT non-Cartesian to Cartesian. 
	    NFFT_BACKWARDS_C2NC, //**< backwards NFFT Cartesian to non-Cartesian.
	    NFFT_BACKWARDS_NC2C //**< backwards NFFT non-Cartesian to Cartesian. 
	};
	*/
	
	
	//Execute the NFFT.
	//\param[in] in the input array.
	//\param[out] out the output array.
	//\param[in] dcw optional density compensation weights weighing the input samples according to the sampling density. 
	//If an 0x0-pointer is provided no density compensation is used.
	//\param mode enum specifying the mode of operation.
	
	virtual void compute( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out,
			      hoNDArray<T> *dcw, int NFFT_comp_mode ); //NFFT_FORWARDS_CN2C = 1, NFFT_FORWARDS_NC2C = 2, NFFT_BACKWARDS_C2NC = 4, NFFT_BACKWARDS_NC2C = 8
	
        
	//Execute an NFFT iteraion (from Cartesian image space to non-Cartesian Fourier space and back to Cartesian image space).
	//\param[in] in the input array.
	//\param[out] out the output array.
	//\param[in] dcw optional density compensation weights weighing the input samples according to the sampling density. 
	//If an 0x0-pointer is provided no density compensation is used.
	//\param[in] halfway_dims specifies the dimensions of the intermediate Fourier space (codomain).
	
	virtual void mult_MH_M( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out,
			hoNDArray<T> *dcw, std::vector<size_t> halfway_dims );
	
    public: // Utilities
        
	   //Enum specifying the direction of the NFFT standalone convolution
	
	/**
	enum NFFT_conv_mode { 
	    NFFT_CONV_C2NC, //**< convolution: Cartesian to non-Cartesian. 
	    NFFT_CONV_NC2C //**< convolution: non-Cartesian to Cartesian. 
	};
	*/
	
	
	//Perform "standalone" convolution
	//\param[in] in the input array.
	//\param[out] out the output array.
	//\param[in] dcw optional density compensation weights.
	//\param[in] mode enum specifying the mode of the convolution
	//\param[in] accumulate specifies whether the result is added to the output (accumulation) or if the output is overwritten.
	
	virtual void convolve( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, hoNDArray<T> *dcw,
			       int NFFT_conv_mode, bool accumulate = false ); //NFFT_CONV_C2NC = 1, NFFT_CONV_NC2C = 2
	
	
	//Enum specifying the direction of the NFFT standalone FFT.
	
	/**
	enum NFFT_fft_mode { 
	    NFFT_FORWARDS, //**< forwards FFT. 
	    NFFT_BACKWARDS //**< backwards FFT. 
	};
	*/
	
	
	//Cartesian FFT. For completeness, just invokes the cuNDFFT class.
	//\param[in,out] data the data for the inplace FFT.
	//\param mode enum specifying the direction of the FFT.
	//\param do_scale boolean specifying whether FFT normalization is desired.
	
	virtual void fft( hoNDArray<complext<T> > *data, int NFFT_fft_mode, bool do_scale = true ); //NFFT_FORWARDS = 1, NFFT_BACKWARDS = 2
  
	
	//NFFT deapodization.
	//\param[in,out] image the image to be deapodized (inplace).
	
	virtual void deapodize( hoNDArray<complext<T> > *image, bool fourier_domain=false );
	
    public: // Setup queries
	
	
	//Get the matrix size.
	
	inline typename uint64d<D>::Type get_matrix_size(){
	    return matrix_size;
	}
	
	
	//Get the oversampled matrix size.
	
	inline typename uint64d<D>::Type get_matrix_size_os(){
	    return matrix_size_os;
	}
	
	
	//Get the convolution kernel size
	
	inline T get_W(){
	    return W;
	}
	
	
	//Query of the plan has been setup
	
	inline bool is_setup(){
	    return initialized;
	}
	
	//friend struct _convolve_NFFT_NC2C<T,D>;
	
	
	
    private: // Internal to the implementation
	
	// Validate setup / arguments
	enum NFFT_components { _NFFT_CONV_C2NC = 1, _NFFT_CONV_NC2C = 2, _NFFT_FFT = 4, _NFFT_DEAPODIZATION = 8 };
	void check_consistency( hoNDArray<complext<T> > *samples, hoNDArray<complext<T> > *image,
				hoNDArray<T> *dcw, unsigned char components );

	

	// Shared barebones constructor
	void barebones();
	
	
	// Compute beta control parameter for Kaiser-Bessel kernel
	void compute_beta();
	
	
	// Compute deapodization filter
	boost::shared_ptr<hoNDArray<complext<T> > > compute_deapodization_filter(bool FFTed = false);
	
	// Dedicated computes
	void compute_NFFT_C2NC(  hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out );
	void compute_NFFT_NC2C(  hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out );
	void compute_NFFTH_NC2C( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out );
	void compute_NFFTH_C2NC( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out );
	
	
	// Dedicated convolutions
	void convolve_NFFT_C2NC( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate );
	void convolve_NFFT_NC2C( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate );
	
	// Internal utility
	void image_wrap( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate );
	
	
    private:
	
	typename uint64d<D>::Type matrix_size;          // Matrix size
	typename uint64d<D>::Type matrix_size_os;       // Oversampled matrix size
	typename uint64d<D>::Type matrix_size_wrap;     // Wrap size at border
	
	typename reald<T,D>::Type alpha;           // Oversampling factor (for each dimension)
	typename reald<T,D>::Type beta;            // Kaiser-Bessel convolution kernel control parameter
	
	T W;                                       // Kernel width in oversampled grid
	
	unsigned int number_of_samples;               // Number of samples per frame per coil
	unsigned int number_of_frames;                // Number of frames per reconstruction
	
	//int device;                                   // Associated device id

	 //
	 // Internal data structures for convolution and deapodization
	 //

	boost::shared_ptr< hoNDArray<complext<T> > > deapodization_filter; //Inverse fourier transformed deapodization filter

	boost::shared_ptr< hoNDArray<complext<T> > > deapodization_filterFFT; //Fourier transformed deapodization filter
   

	//    thrust::device_vector< typename reald<REAL,D>::Type > *trajectory_positions;
	//vector_td<T,D> *trajectory_positions;
	hoNDArray<typename reald<T,D>::Type> *trajectory_positions;
	//vector_td<T>  *trajectory_positions;
	
	hoNDArray<typename uintd<D>::Type> grid_cell_pos;
	hoNDArray<unsigned int> traj_idx;
	unsigned int num_cells;

	unsigned int *tuples_last;
	unsigned int *bucket_begin, *bucket_end;

	//
	// State variables
	//

 
	bool preprocessed_C2NC, preprocessed_NC2C;
	bool initialized;
    
    };

    // Pure virtual class to cause compile errors if you try to use NFFT with double and atomics
    // - since this is not supported on the device
    //template< unsigned int D> class EXPORTGPUNFFT cuNFFT_plan<double,D,true>{ 
    //	virtual void atomics_not_supported_for_type_double() = 0; };

    
}
