// Includes - Gadgetron
#include "hoNFFT.h"
#include "hoNDFFT.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_utils.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"
#include "NFFT.h"
#include "string.h"

// Includes - stdlibs
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
//#include "hoNDArray_reductions.h"

//using namespace std;
using std::vector;
using namespace Gadgetron;

// Includes containing the NFFT convolution implementation
#include "KaiserBessel_operators.h"
#include "NFFT_C2NC_conv_kernel.cpp"
//#include "NFFT_NC2C_conv_kernel.cu"
#include "NFFT_preprocess.cpp"
#include "hoNDArray_reductions.h"



//
// Public class methods
//	   

template<class T, unsigned int D> Gadgetron::hoNFFT_plan<T,D>::hoNFFT_plan() : NFFT_plan<T, D>()
{
    // Minimal initialization
    //std::cout << "hoNFFT_plan(): call to barebones()" << std::endl;
    barebones();
}

template<class T, unsigned int D> Gadgetron::hoNFFT_plan<T,D>::hoNFFT_plan( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W ) 
: NFFT_plan<T, D>( matrix_size,  matrix_size_os, W)
{
    // Minimal initialization
    //std::cout << "hoNFFT_plan" << std::endl;
    //std::cout << "matrix_size: " << matrix_size << ", matrix_size_os: " << matrix_size_os << ", W: " << W << std::endl;
    //std::cout << "hoNFFT_plan(input): call to barebones()" << std::endl;
    barebones();

    //std::cout << "hoNFFT_plan: barebones done" << std::endl;

    // Setup plan
    //std::cout << "hoNFFT_plan(input): call to setup()" << std::endl;
    setup( matrix_size, matrix_size_os, W );

    //std::cout << "hoNFFT_plan: setup done" << std::endl;
}

template<class T, unsigned int D> Gadgetron::hoNFFT_plan<T,D>::~hoNFFT_plan()
{
    //std::cout << "~hoNFFT_plan(): call to wipe()" << std::endl;
    wipe(1); //NFFT_WIPE_ALL
}


template<class T, unsigned int D>  void Gadgetron::hoNFFT_plan<T,D>::setup( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W, int device )
{
    // Free memory
    //std::cout << "setup(): call to wipe()" << std::endl;
    wipe(1); //NFFT_WIPE_ALL

    // The convolution does not work properly for very small convolution kernel widths
    // (experimentally observed limit)

    if( W < T(1.8) ) {
	throw std::runtime_error("Error: the convolution kernel width for the cuNFFT plan is too small.");
    }

    //
    // Check input against certain requirements
    //
    
    //
    // Setup private variables
    //

    this->matrix_size = matrix_size;
    this->matrix_size_os = matrix_size_os;
    
    T W_half = T(0.5)*W; //Half kernel size
    vector_td<T,D> W_vec(W_half); //Vector of half kernel size repeated D times

    matrix_size_wrap = vector_td<size_t,D>( ceil(W_vec) );
    //This is used for increasing the matrix_size_os on all sides by the kernel size when convolving with the kernel
    matrix_size_wrap<<=1; //matrix_wrap_size = matrix_wrap_size << 1 = matrix_wrap_size * 2 
    alpha = vector_td<T,D>(matrix_size_os) / vector_td<T,D>(matrix_size);
  
    typename reald<T,D>::Type ones(T(1));
    if( weak_less( alpha, ones ) ){
	throw std::runtime_error("Error: hoNFFT : Illegal oversampling ratio suggested");
    }

    this->W = W;
  
    // Compute Kaiser-Bessel beta
    //std::cout << "setup(): call to compute_beta()" << std::endl;
    compute_beta();
  
    initialized = true;
}


template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory, int NFFT_prep_mode )
{

    if ( !(NFFT_prep_mode == 1) && !(NFFT_prep_mode == 2) && !(NFFT_prep_mode == 4) ){
    	throw std::runtime_error("Error: hoNFFT_plan: preprocess: mode must be NFFT_PREP_C2NC (1), NFFT_PREP_NC2C (2) or NFFT_PREP_ALL (4)");
    }

    if (NFFT_prep_mode == 1) std::cout << "NFFT prep mode: NFFT_PREP_C2NC" << std::endl;
    if (NFFT_prep_mode == 2) std::cout << "NFFT prep mode: NFFT_PREP_NC2C" << std::endl;
    if (NFFT_prep_mode == 4) std::cout << "NFFT prep mode: NFFT_PREP_ALL" << std::endl;

    if( !trajectory || trajectory->get_number_of_elements()==0 ){
	throw std::runtime_error("Error: hoNFFT_plan::preprocess: invalid trajectory");
    }
    
    if( !initialized ){
	throw std::runtime_error("Error: hoNFFT_plan::preprocess: hoNFFT_plan::setup must be invoked prior to preprocessing.");
    }
    
    //std::cout << "preprocess(): call to wipe()" << std::endl;
    wipe(2);//NFFT_WIPE_PREPROCESSING

    //std::cout << "Preprocess: wipe done" << std::endl;
    
    //hoNDArray<T> *trajectory;    
    
    number_of_samples = trajectory->get_size(0);
    number_of_frames = trajectory->get_number_of_elements()/number_of_samples;
    //vector_td<T,1> n_dims_tmp = trajectory->get_number_of_dimensions();
    //std::vector<long unsigned int> traj_dims_tmp = *trajectory->get_dimensions();
    //typename uint64d<1>::Type image_dims = from_std_vector<size_t,1> *trajectory->get_dimensions();
    //T* traj_ptr_tmp = &trajectory->get_data_ptr()[0];//&(trajectory->get_data_ptr()[0]);
    //std::cout << "pointer: " << &traj_ptr_tmp << std::endl;
    //std::cout << "traj dim 0: " << traj_dims_tmp[0] << ", traj dim 1: " << traj_dims_tmp[1] <<  std::endl;

    //std::cout << "sam and frame: " << number_of_samples << " and " << number_of_frames << std::endl;
    //std::cout << n_dims_tmp << std::endl;

    T* first_val_tmp = (T*)trajectory->get_data_ptr();
    T* last_val_tmp = (T*)(trajectory->get_data_ptr()+trajectory->get_number_of_elements()-1);

    //std::cout << "traj first / last: " << .first_val_tmp[0] << " / " << last_val_tmp[0] << std::endl;
    
    //std::cout << "Preprocess: number of samples and frames done" << std::endl;

    // Make sure that the trajectory values are within range [-1/2;1/2]
    //std::pair<T*,T*> mm_pair
    std::pair<T*,T*> mm_pair = std::minmax_element(first_val_tmp, last_val_tmp);
    //std::pair<T*,T*> mm_pair = std::minmax_element(traj_ptr_tmp,  (traj_ptr_tmp + &trajectory_int->get_number_of_elements()*D));

    //std::cout << "mmpair: " << *mm_pair[0] << std::endl;
    //std::cout << "Preprocess: mmpair done" << std::endl;
  
    
    if( *mm_pair.first < T(-0.5) || *mm_pair.second > T(0.5) ){
    	std::stringstream ss;
    	ss << "Error: hoNFFT::preprocess : trajectory [" << *mm_pair.first << "; " << *mm_pair.second << "] out of range [-1/2;1/2]";
    	throw std::runtime_error(ss.str());
    }
    

    //std::cout << "Preprocess: [-0.5 0.5] check done" << std::endl;

    //vector_td<unsigned int> traj_dims_tmp = trajectory->get_dimensions();
    //int traj_ptr_tmp = trajectory->get_data_ptr();

    //std::vector<size_t> traj_dims_tmp = *trajectory->get_dimensions();
    //unsigned int traj_ndims_tmp = trajectory->get_number_of_dimensions();
    //std::cout << "num dims in trajectory: " << traj_ndims_tmp << std::endl;
    //std::cout << "traj_dims: " << traj_dims_tmp[0]; for (int d = 1; d < D; d++) std::cout << ", " << traj_dims_tmp[d]; std::cout << std::endl;
    trajectory_positions = new hoNDArray<typename reald<T,D>::Type>(*trajectory->get_dimensions());

    //std::cout << "Preprocess: creting trajectory positions done" << std::endl;

    // ####### CALC TRAJECTORY POSITIONS #######
    
    vector_td<T,D> matrix_size_os_real = vector_td<T,D>( matrix_size_os ); //Making vector sized D with all elements having the value of matrix_size_os
    vector_td<T,D> matrix_size_os_plus_wrap_real = vector_td<T,D>( (matrix_size_os+matrix_size_wrap)>>1 ); //Added kernel width and taken hal
   
    // Make trajectory postions in range [matrix_size_wrap/2   matrix_size_os+matrix_size_wrap/2] instead of [-1/2 1/2]
    for(int i = 0; i < trajectory_positions->get_number_of_elements()-1; i++){
	trajectory_positions->get_data_ptr()[i] = trajectory->get_data_ptr()[i] * matrix_size_os_real + matrix_size_os_plus_wrap_real;
    }
    //* WRITE */ write_nd_array<typename reald<T,D>::Type>(trajectory_positions, "/home/karen/data/images/trajectory_positions.real");
    
    //T* first_val_tmp_tmp = (T*)trajectory_positions->get_data_ptr();
    //T* last_val_tmp_tmp = (T*)(trajectory_positions->get_data_ptr()+trajectory_positions->get_number_of_elements()-1);
    //std::pair<T*,T*> mm_pair_tmp = std::minmax_element(first_val_tmp_tmp, last_val_tmp_tmp);
    //printf("\nmin/max traj pos: %f, %f \n", *mm_pair_tmp.first, *mm_pair_tmp.second );
    
    if ( !(NFFT_prep_mode == 1) ){ //not C2NC

	T half_W = T(0.5)*W;
	
	//Calculating number of cells for the convolution kernel to cover
	unsigned int upper_limit = (unsigned int) std::floor( half_W ); //For W = 5.5, upper_lim = floor(2.75) = 2
	unsigned int lower_limit = (unsigned int) std::ceil( -half_W ); //For W = 5.5, lower_lim = ceil(-2.75) = -2
	num_cells = (upper_limit-lower_limit+1);
	//std::cout << "num_cells before multiplying: " << num_cells << std::endl;
	//std::cout << "D: " << D << std::endl;
	unsigned int num_cells_tmp = num_cells;
	for (int d = 0; d < (D-1); d++){
	    //std::cout << "Loop run: " << d << std::endl;
	    //std::cout << "Num_cells: " << num_cells << std::endl; 
	    num_cells *= num_cells_tmp;
	}
	//std::cout << "Number of cells to include in convolution: " << num_cells << std::endl;

        grid_cell_pos.create(trajectory_positions->get_number_of_elements()*num_cells);
        traj_idx.create(trajectory_positions->get_number_of_elements()*num_cells);
	//std::cout << "Number of elements to loop over during convolution: " << traj_idx.get_number_of_elements() << std::endl;
	//std::cout << "Trajectory lenghth: " << trajectory->get_number_of_elements() << std::endl;
	//std::cout << "num_cells: " << num_cells << std::endl;

	int traj_length = trajectory_positions->get_number_of_elements();
	
	//std::cout << "preprocess(): call to output_pairs()" << std::endl;
	output_pairs<T>( trajectory_positions->get_data_ptr(), traj_length, vector_td<unsigned int,D> (matrix_size_os), vector_td<unsigned int,D> (matrix_size_wrap), 
			 grid_cell_pos.get_data_ptr(), traj_idx.get_data_ptr(), half_W, num_cells );

	//std::cout << "grid_cell_pos lenghth: " << grid_cell_pos.get_number_of_elements() << std::endl;
	//std::cout << "traj_idx lenghth: " << traj_idx.get_number_of_elements() << std::endl;
	

	///* WRITE */ write_nd_array<unsigned int>(&traj_idx, "/home/karen/data/images/traj_idx.uint");
	///* WRITE */ write_nd_array<typename uintd<D>::Type>(&grid_cell_pos, "/home/karen/data/images/grid_cell_pos.uint");
	}

    preprocessed_C2NC = true;

    if( !(NFFT_prep_mode == 1) )
	preprocessed_NC2C = true;
}



template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out,
									     hoNDArray<T> *dcw, int NFFT_comp_mode )
{  

    //* WRITE */ write_nd_array<complext<T> >(in, "/home/karen/data/images/compute_input.cplx");
    //std::cout << "hoNFFT: compute: in dims: " << in->get_size(0) << ", " << in->get_size(1) << std::endl;

    // Validity checks
    
    unsigned char components;

    if( NFFT_comp_mode == 1 ){ //NFFT_FORWARDS_C2NC 
	components = _NFFT_CONV_C2NC + _NFFT_FFT + _NFFT_DEAPODIZATION; //1+4+8
	std::cout << "NFFT comp mode: NFFT_FORWARDS_C2NC" << std::endl;
    }
    else if( NFFT_comp_mode == 2 ){ //NFFT_FORWARDS_NC2C
	components = _NFFT_CONV_NC2C + _NFFT_FFT + _NFFT_DEAPODIZATION; //2+4+8
	std::cout << "NFFT comp mode: NFFT_FORWARDS_NC2C" << std::endl;
    }
    else if( NFFT_comp_mode == 4 ){ //NFFT_BACKWARDS_C2NC 
	components = _NFFT_CONV_NC2C + _NFFT_FFT + _NFFT_DEAPODIZATION; //2+4+8
	std::cout << "NFFT comp mode: NFFT_BACKWARDS_NC2C" << std::endl;
    }
    else if( NFFT_comp_mode == 8 ){ //NFFT_BACKWARDS_NC2C
	components = _NFFT_CONV_C2NC + _NFFT_FFT + _NFFT_DEAPODIZATION; //1+4+8
	std::cout << "NFFT comp mode: NFFT_BACKWARDS_NC2C" << std::endl;
    }
    else{
	throw std::runtime_error("Error: hoNFFT_plan::compute: unknown mode");
    }
 
  
    
    {
	hoNDArray<complext<T> > *samples, *image;
	
	if( NFFT_comp_mode == 1 || NFFT_comp_mode == 4 ){ //NFFT C2NC FORWARDS and BACKWARDS
	    image = in; samples = out;
	} else{
	    image = out; samples = in;
	}
	
	//std::cout << "compute(): call to check_consistency()" << std::endl;
	check_consistency( samples, image, dcw, components ); //First consistency check
	//* WRITE */ write_nd_array<complext<T> >(samples, "/home/karen/data/images/compute_samples.cplx");
    }
    //std::cout << "check consistency successful!" << std::endl;
    
    hoNDArray<complext<T> > *in_int = 0x0, *out_int = 0x0;
    hoNDArray<T> *dcw_int = 0x0;
    
    typename uint64d<D>::Type image_dims = from_std_vector<size_t,D>
	( (NFFT_comp_mode == 1 || NFFT_comp_mode == 4 ) ? *in->get_dimensions() : *out->get_dimensions() ); //Must be D-dimensional integer (integer representing D sizes), C2NC
    bool oversampled_image = (image_dims==matrix_size_os);
    
    vector<size_t> vec_dims = to_std_vector(matrix_size_os); //Must be 1-d standard vector with D number of elements
    {
	hoNDArray<complext<T> > *image = ((NFFT_comp_mode == 1 || NFFT_comp_mode == 4 ) ? in : out ); //C2NC
	for( unsigned int d=D; d<image->get_number_of_dimensions(); d++ )
	    vec_dims.push_back(image->get_size(d));
    }
    
    hoNDArray<complext<T> > *working_image = 0x0;
    hoNDArray<complext<T> > *working_samples = 0x0;

    //typename uint64d<D>::Type matrix_size_crop = (matrix_size_os-matrix_size)>>1;
    
    switch(NFFT_comp_mode){
	
    case 1: //NFFT_FORWARDS_C2NC
	
	if( !oversampled_image ){
	    working_image = new hoNDArray<complext<T> >(&vec_dims);
	    //std::cout << "hoNFFT: compute: working image dims: " << working_image->get_size(0) << ", " << working_image->get_size(1) << std::endl;
	    pad<complext<T>, D >(matrix_size_os, in, working_image );
	    //pad(matrix_size_os[0], in, working_image );
	    //std::cout << "hoNFFT: compute: working image dims: " << working_image->get_size(0) << ", " << working_image->get_size(1) << std::endl;
	}
	else{
	    working_image = in;
	}
	
	//std::cout << "compute(): call to compute_NFFT_C2NC()" << std::endl;
	compute_NFFT_C2NC( working_image, out );
	
	
	if( dcw )
	    multiply(*out, *dcw, *out);
	
	
	if( !oversampled_image ){
	    delete working_image; working_image = 0;
	}    
	break;
	
	
    case 2: //NFFT_FORWARDS_NC2C:
	
	// Density compensation
	if( dcw ){
	    working_samples = new hoNDArray<complext<T> >(*in);
	    multiply(*working_samples, *dcw, *working_samples);
	}
	else{
	    working_samples = in;
	}
	
	if( !oversampled_image ){
	    working_image = new hoNDArray<complext<T> >(&vec_dims);
	}
	else{
	    working_image = out;
	}
    
	//std::cout << "compute(): call to compute_NFFT_NC2C()" << std::endl;
	compute_NFFT_NC2C( working_samples, working_image );
	
	if( !oversampled_image ){
	    crop<complext<T>, D>( matrix_size, working_image, out );
	}
	
	if( !oversampled_image ){
	    delete working_image; working_image = 0x0;
	}
	
	if( dcw ){
	    delete working_samples; working_samples = 0x0;
	}    
	break;
	
    case 8: //NFFT_BACKWARDS_NC2C:
	
	// Density compensation
	//if (dcw)	
	    //std::cout << "Compute: NFFT_BACKWARDS_NC2C if dcw: " << dcw << std::endl;
	//std::cout << "Compute: NFFT_BACKWARDS_NC2C dcw size: " << dcw->get_number_of_elements() << std::endl;
	if( dcw ){
	    working_samples = new hoNDArray<complext<T> >(*in);
	    //* WRITE */ write_nd_array<complext<T> >(working_samples, "/home/kh/data/images/compute_working_samples.cplx");
	    //* WRITE */ write_nd_array<T>(dcw, "/home/kh/data/images/compute_dcf.real");
	    //std::cout << "Compute: myltiplying dcw" << std::endl;
	    //*working_samples *= *dcw;
	    multiply(*working_samples, *dcw, *working_samples);
	}
	else{
	    working_samples = in;
	}
	//std::cout << "Compute: after myltiplying dcw" << std::endl;

	//* WRITE */ write_nd_array<complext<T> >(working_samples, "/home/karen/data/images/compute_working_samples_dcw.cplx");
	
	if( !oversampled_image ){
	    working_image = new hoNDArray<complext<T> >(&vec_dims);
	}
	else{
	    working_image = out;
	}
	
	//std::cout << "compute(): call to compute_NFFTH_NC2C()" << std::endl;
	compute_NFFTH_NC2C( working_samples, working_image );
	
	if( !oversampled_image ){
	    crop<complext<T> ,D>( matrix_size, working_image, out );
	}
	
	if( !oversampled_image ){
	    delete working_image; working_image = 0x0;
	}
	
	if( dcw ){
	    delete working_samples; working_samples = 0x0;
	}    
	break;
	
    case 4: //NFFT_BACKWARDS_C2NC:
	
	if( !oversampled_image ){
	    working_image = new hoNDArray<complext<T> >(&vec_dims);
	    
	    pad<complext<T>, D >( matrix_size_os, in, working_image );
	    //pad( matrix_size_os[1], in, working_image );
	}
	else{
	    working_image = in;
	}
	
	//std::cout << "compute(): call to compute_NFFTH_C2NC()" << std::endl;
	compute_NFFTH_C2NC( working_image, out );
	
	if( dcw )
	    multiply(*out, *dcw, *out);
        
	if( !oversampled_image ){
	    delete working_image; working_image = 0x0;
	}
	
	break;
    };
    //* WRITE */ write_nd_array<complext<T> >(out, "/home/karen/data/images/compute_output.cplx");
}



template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::mult_MH_M( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out,
									       hoNDArray<T> *dcw, std::vector<size_t> halfway_dims )
{
    //std::cout << "hoNFFT: mult_MH_M: beginning" << std::endl;
    // Validity checks
    
    unsigned char components = _NFFT_CONV_C2NC + _NFFT_CONV_NC2C + _NFFT_FFT + _NFFT_DEAPODIZATION;
    
    if( in->get_number_of_elements() != out->get_number_of_elements() ){
	throw std::runtime_error("Error: hoNFFT_plan::mult_MH_M: in/out image sizes mismatch");
    }
    
    hoNDArray<complext<T> > *working_samples = new hoNDArray<complext<T> >(&halfway_dims);
    
    //std::cout << "mult_MH_M(): call to check_consistency()" << std::endl;
    check_consistency( working_samples, in, dcw, components );
    //std::cout << "hoNFFT: mult_MH_M: check consistency done" << std::endl;
    
    hoNDArray<complext<T> > *in_int = 0x0;
    hoNDArray<complext<T> > *out_int = 0x0;
    hoNDArray<T> *dcw_int = 0x0;
    
    hoNDArray<complext<T> > *working_image = 0x0;
    
    typename uint64d<D>::Type image_dims = from_std_vector<size_t,D>(*in->get_dimensions()); 
    bool oversampled_image = (image_dims==matrix_size_os); 
    
    vector<size_t> vec_dims = to_std_vector(matrix_size_os); 
    for( unsigned int d=D; d<in->get_number_of_dimensions(); d++ )
	vec_dims.push_back(in->get_size(d));
    
    if( !oversampled_image ){
	working_image = new hoNDArray<complext<T> >(&vec_dims);
	pad<complext<T>, D >( matrix_size_os, in, working_image );
	//pad( matrix_size_os[1], in, working_image );
    }
    else{
	working_image = in;
    }
    
    //std::cout << "mult_MH_M(): call to compute_NFFT_C2NC()" << std::endl;
    compute_NFFT_C2NC( working_image, working_samples );
    
    // Density compensation
    if( dcw ){
	multiply(*working_samples, *dcw, *working_samples);
	multiply(*working_samples, *dcw, *working_samples);
    }
    
    //std::cout << "mult_MH_M(): call to compute_NFFTH_NC2C()" << std::endl;
    compute_NFFTH_NC2C( working_samples, working_image );
    
    delete working_samples;
    working_samples = 0x0;
    
    if( !oversampled_image ){
	crop<complext<T>, D>( matrix_size, working_image, out );
	delete working_image; working_image = 0x0;
    }
   
}


template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::convolve( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out,
									      hoNDArray<T> *dcw, int NFFT_conv_mode, bool accumulate )
{
    //* WRITE */ write_nd_array<complext<T> >(in, "/home/karen/data/images/convolve_input.cplx");

    //std::cout << "Convolve: mode: " << NFFT_conv_mode << std::endl;
    unsigned char components;
    
    if( NFFT_conv_mode == 1 ){ //NFFT_CONV_C2NC 
	components = _NFFT_CONV_C2NC;
	std::cout << "NFFT conv mode: NFFT_CONV_C2NC" << std::endl;
    }
    else if (NFFT_conv_mode == 2){ //NFFT_CONV_NC2C
	components = _NFFT_CONV_NC2C;
	std::cout << "NFFT conv mode: NFFT_CONV_NC2C" << std::endl;
    }
    else
	throw std::runtime_error("Error: hoNFFT_plan::convolve: NFFT_conv_mode must be either NFFT_CONV_C2NC (1) or NFFT_CONV_NC2C (2)");
	
    
    {
	hoNDArray<complext<T> > *samples, *image;
	
	if( NFFT_conv_mode == 1 ){ //NFFT_CONV_C2NC
	    image = in; samples = out;
	} else{ //NFFT_CONV_NC2C
	    image = out; samples = in;
	}
	
	//std::cout << "convolve(): call to check_consistency()" << std::endl;
	check_consistency( samples, image, dcw, components );
    }
    
    hoNDArray<complext<T> > *in_int = 0x0, *out_int = 0x0;
    hoNDArray<T> *dcw_int = 0x0;
    
    hoNDArray<complext<T> > *working_samples = 0x0;
    
    typename uint64d<D>::Type image_dims = from_std_vector<size_t, D>
	(*(((NFFT_conv_mode == 1) ? in : out )->get_dimensions())); 
    bool oversampled_image = (image_dims==matrix_size_os); 

    
    if( !oversampled_image ){
	throw std::runtime_error("Error: hoNFFT_plan::convolve: ERROR: oversampled image not provided as input.");
    }
    
    vector<size_t> vec_dims = to_std_vector(matrix_size_os); 
    {
	hoNDArray<complext<T> > *image = ((NFFT_conv_mode == 1) ? in : out );
	for( unsigned int d=D; d<image->get_number_of_dimensions(); d++ )
	    vec_dims.push_back(image->get_size(d));
    }
    
    //std::cout << "Convolve: after vec_dims push" << std::endl;

    switch(NFFT_conv_mode){
	
    case 1: //NFFT_CONV_C2NC:
	//std::cout << "convolve(): call to convolve_NFFT_C2NC()" << std::endl;
  	convolve_NFFT_C2NC( in, out, accumulate );
  	if( dcw ) multiply(*out, *dcw, *out);
	break;
	
    case 2: //NFFT_CONV_NC2C:
	
	//std::cout << "Convolve: in case NFFT_CONV_NC2C " << std::endl;
	    
	// Density compensation
	if( dcw ){
	    working_samples = new hoNDArray<complext<T> >(*in);
	    multiply(*working_samples, *dcw, *working_samples);
	    //std::cout << "Convolve: dcw exist!" << std::endl;
	}
	else{
	    working_samples = in;
	    //std::cout << "Convolve: dcw is 0x0!" << std::endl;
	}
	
	//* WRITE */ write_nd_array<complext<T> >(working_samples, "/home/karen/data/images/convolve_working_samples.cplx");

	//std::cout << "convolve(): call to convolve_NFFT_NC2C()" << std::endl;
	convolve_NFFT_NC2C( working_samples, out, accumulate );
	//_convolve_NFFT_NC2C<REAL,D,ATOMICS>::apply( this, working_samples, out_int, accumulate );
	
	//std::cout << "Convolve: in case NFFT_CONV_NC2C: after specific convolution " << std::endl;

	if( dcw ){
	    delete working_samples; working_samples = 0x0;
	}    
	break;
	
    default:
	throw std::runtime_error( "Error: hoNFFT_plan::convolve: unknown mode.");
    }
}


template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::fft(hoNDArray<complext<T> > *data, int NFFT_fft_mode, bool do_scale )
{
    hoNDArray<complext<T> > *data_int = 0x0;
    
    if( NFFT_fft_mode == 1 ){ //NFFT_FORWARDS
	std::cout << "NFFT fft mode: NFFT_FORWARDS" << std::endl;
	for(int i=0; i<D; i++){
	    hoNDFFT<T>::instance()->fft( data, i );
	}
    }
    else if(NFFT_fft_mode == 2){ //NFFT_BACKWARDS
	std::cout << "NFFT fft mode: NFFT_BACKWARDS" << std::endl;
	for(int i=0; i<D; i++){
	    hoNDFFT<T>::instance()->ifft( data, i );
	}
    }
    else
	throw std::runtime_error("Error: hoNFFT_plan: fft: NFFT_fft_mode must be either NFFT_FORWARDS (1) or NFFT_BACKWARDS (2)");
}

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::deapodize( hoNDArray<complext<T> > *image, bool fourier_domain)
{

    //* WRITE */ write_nd_array<complext<T> >(image, "/home/karen/data/images/deapodize_input.cplx");
    //std::cout << "Deapodization: fourier_domain = " << fourier_domain << std::endl;
    unsigned char components;
    components = _NFFT_FFT;
    //std::cout << "deapodize(): call to check_consistency()" << std::endl;
    check_consistency( 0x0, image, 0x0, components );
    
    hoNDArray<complext<T> > *image_int = 0x0;
    
    typename uint64d<D>::Type image_dims = from_std_vector<size_t, D>(*image->get_dimensions()); 
    bool oversampled_image = (image_dims==matrix_size_os); 
    
    if( !oversampled_image ){
	throw std::runtime_error( "Error: hoNFFT_plan::deapodize: ERROR: oversampled image not provided as input.");
    }
    
    if (fourier_domain){
  	if (!deapodization_filterFFT){
	    //std::cout << "deapodize(): call to compute_deapodization_filter()" << std::endl;
	    deapodization_filterFFT = compute_deapodization_filter(true);
	}
	multiply(*image, *deapodization_filterFFT, *image);
    } 
    else {
  	if (!deapodization_filter){
	    //std::cout << "deapodize(): call to compute_deapodization_filter()" << std::endl;
	    deapodization_filter = compute_deapodization_filter(false);
	    //* WRITE */ write_nd_array<complext<T> >(deapodization_filter.get(), "/home/karen/data/images/deapodization_filter_1p1.cplx");
	}
	multiply(*image, *deapodization_filter, *image);
    }
 }

//
// Private class methods
//

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::check_consistency( hoNDArray<complext<T> > *samples, hoNDArray<complext<T> > *image,
										       hoNDArray<T> *weights, unsigned char components )
{

    if( !initialized ){
	throw std::runtime_error( "Error: hoNFFT_plan: Unable to proceed without setup.");
    }

    //std::cout << "preprocessed_C2NC: " << preprocessed_C2NC << "\npreprocessed_NC2C: " << preprocessed_NC2C << std::endl;    
    if( (components & _NFFT_CONV_C2NC ) && !preprocessed_C2NC ){
	throw std::runtime_error("Error: hoNFFT_plan: Unable to compute NFFT before preprocessing.");
    }
    
    if( (components & _NFFT_CONV_NC2C ) && !preprocessed_NC2C ){
	throw std::runtime_error("Error: hoNFFT_plan: Unable to compute NFFT before preprocessing.");
    }
    
    if( ((components & _NFFT_CONV_C2NC ) || (components & _NFFT_CONV_NC2C )) && !(image && samples) ){
	throw std::runtime_error("Error: hoNFFT_plan: Unable to process 0x0 input/output.");
    }
    
    if( ((components & _NFFT_FFT) || (components & _NFFT_DEAPODIZATION )) && !image ){
	throw std::runtime_error("Error: hoNFFT_plan: Unable to process 0x0 input.");
    }
    
    if( image->get_number_of_dimensions() < D ){
	throw std::runtime_error("Error: hoNFFT_plan: Number of image dimensions mismatch the plan.");
    }    
    
    typename uint64d<D>::Type image_dims = from_std_vector<size_t,D>( *image->get_dimensions() );
    bool oversampled_image = (image_dims==matrix_size_os);
    
    //std::cout << "Check consistency: oversampled image: " << oversampled_image << std::endl;
    //std::cout << "Check consistency: image_dims: " << image_dims << std::endl;
    if( !((oversampled_image) ? (image_dims == matrix_size_os) : (image_dims == matrix_size) )){
	throw std::runtime_error("Error: hoNFFT_plan: Image dimensions mismatch.");
    }
    
    //std::cout << "Check consistency: after initial checks" << std::endl;

    if( (components & _NFFT_CONV_C2NC ) || (components & _NFFT_CONV_NC2C )){
	//std::cout << "Consistency check: beginning of if" << std::endl;
	//std::cout << "Consistency check: samples number of elements: " << samples->get_number_of_elements() << std::endl;
	if( (samples->get_number_of_elements() == 0) || (samples->get_number_of_elements() % (number_of_frames*number_of_samples)) ){
	    printf("\nhoNFFT::check_consistency() failed:\n#elements in the samples array: %ld.\n#samples from preprocessing: %d.\n#frames from preprocessing: %d.\n",samples->get_number_of_elements(), number_of_samples, number_of_frames ); fflush(stdout);
	    throw std::runtime_error("Error: hoNFFT_plan: The number of samples is not a multiple of #samples/frame x #frames as requested through preprocessing");
	}

	//std::cout << "Check consistency: after element check if components" << std::endl;
    
	unsigned int num_batches_in_samples_array = samples->get_number_of_elements()/(number_of_frames*number_of_samples);
	unsigned int num_batches_in_image_array = 1;
	
	for( unsigned int d=D; d<image->get_number_of_dimensions(); d++ ){
	    num_batches_in_image_array *= image->get_size(d);
	}
	num_batches_in_image_array /= number_of_frames;
	
	if( num_batches_in_samples_array != num_batches_in_image_array ){
	    printf("\nhoNFFT::check_consistency() failed:\n#elements in the samples array: %ld.\n#samples from preprocessing: %d.\n#frames from preprocessing: %d.\nLeading to %d batches in the samples array.\nThe number of batches in the image array is %d.\n",samples->get_number_of_elements(), number_of_samples, number_of_frames, num_batches_in_samples_array, num_batches_in_image_array ); fflush(stdout);
	    throw std::runtime_error("Error: hoNFFT_plan: Number of batches mismatch between samples and image arrays");
	}
    }

    //std::cout << "Consistency check: after if" << std::endl;
    
    if( components & _NFFT_CONV_NC2C ){
	if( weights ){ 
	    if( weights->get_number_of_elements() == 0 ||
		!( weights->get_number_of_elements() == number_of_samples || 
		   weights->get_number_of_elements() == number_of_frames*number_of_samples) ){
		printf("\ncuNFFT::check_consistency() failed:\n#elements in the samples array: %ld.\n#samples from preprocessing: %d.\n#frames from preprocessing: %d.\n#weights: %ld.\n",samples->get_number_of_elements(), number_of_samples, number_of_frames, weights->get_number_of_elements() ); fflush(stdout);
		throw std::runtime_error("Error: hoNFFT_plan: The number of weights should match #samples/frame x #frames as requested through preprocessing");
	    }
	}
    }  
}



template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::barebones()
{	
    // These are the fundamental booleans checked before accessing the various member pointers
    initialized = preprocessed_C2NC = preprocessed_NC2C = false;
    
    // Clear matrix sizes
    clear(matrix_size);
    clear(matrix_size_os);
    
    // Clear pointers
    trajectory_positions = 0x0;
}


template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::wipe( int NFFT_wipe_mode ){
	    
    if( NFFT_wipe_mode==1 && initialized ){ //NFFT_WIPE_ALL
	deapodization_filter.reset();
	initialized = false;
    }
	    
    if( preprocessed_C2NC || preprocessed_NC2C ){
	delete trajectory_positions;
	preprocessed_C2NC = preprocessed_NC2C = false;
    }
};




template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute_beta()
{	
    // Compute Kaiser-Bessel beta paramter according to the formula provided in 
    // Beatty et. al. IEEE TMI 2005;24(6):799-808.
    // Alpha is the oversampling factor and beta is the convolution kernel control parameter. Both in all three dimensions. 
    for( unsigned int d=0; d<D; d++ )
	beta[d] = (M_PI*std::sqrt((W*W)/(alpha[d]*alpha[d])*(alpha[d]-T(0.5))*(alpha[d]-T(0.5))-T(0.8))); 
}



//
// Function to calculate the deapodization filter
//

template<class T, unsigned int D> boost::shared_ptr<hoNDArray<complext<T> > > Gadgetron::hoNFFT_plan<T,D>::compute_deapodization_filter( bool FFTed)
{
    //std::cout << "compute_deapodiation_filter(): call to KaiserBessel_operators()" << std::endl;
    KaiserBessel_operators<T> kb;

    std::vector<size_t> tmp_vec_os = to_std_vector(matrix_size_os);
    
    boost::shared_ptr< hoNDArray<complext<T> > > filter( new hoNDArray<complext<T> >(tmp_vec_os));
    vector_td<T,D> matrix_size_os_real = vector_td<T,D>(matrix_size_os);
    
    const unsigned int num_elements = prod(matrix_size_os);
    
    for(int i = 0; i < num_elements; i++){

	typename uintd<D>::Type cell_pos;
	int i_tmp = i;
	for(int d = 0; d < D; d++){
	    cell_pos[d] = i_tmp % matrix_size_os[d];
	    i_tmp -= cell_pos[d];
	    i_tmp /= matrix_size_os[d];
	}
	
	// Sample position ("origin")
	const vector_td<T,D> sample_pos = T(0.5)*matrix_size_os_real;
	
	// Calculate the distance between the cell and the sample
	vector_td<T,D> cell_pos_real = vector_td<T,D>(cell_pos);
	const typename reald<T,D>::Type delta = abs(sample_pos-cell_pos_real);
	
	// Compute convolution weight. 
	T weight; 
	T zero = T(0);
	T half_W = T(0.5)*W; //half kernel width
	T one_over_W = T(1)/W;
	vector_td<T,D> half_W_vec( half_W );
	
	
	if( weak_greater( delta, half_W_vec ) )
	    weight = zero;
	else{ 
	    //std::cout << "compute_deapodiation_filter(): call to KaiserBessel()" << std::endl;
	    weight = kb.KaiserBessel( delta, matrix_size_os_real, one_over_W, beta );
	    //std::cout << "weight = " << weight << std::endl;
	}

	filter->get_data_ptr()[i] = weight;

	//if( !weak_greater( delta, half_W_vec ) ){
	    //std::cout << "weight: " << weight << std::endl;
	    //std::cout << "filter value: " << filter->get_data_ptr()[i] << std::endl;
	//}
    }
    //* WRITE */ write_nd_array<complext<T> >(filter.get(), "/home/karen/data/images/filter_1p1.cplx");
    
    // FFT
    if (FFTed){
	//std::cout << "compute_deapodiation_filter(): call to fft()" << std::endl;
  	fft( filter.get(), 1, false ); //NFFT_FORWARDS
    }
    else{
	//std::cout << "compute_deapodiation_filter(): call to fft()" << std::endl;
  	fft( filter.get(), 2, false ); //NFFT_BACKWARDS
    }
    // Reciprocal
    reciprocal_inplace(filter.get());
    return filter;
 }



template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute_NFFT_C2NC( hoNDArray<complext<T> > *image, hoNDArray<complext<T> > *samples )
{
    // private method - no consistency check. We trust in ourselves.
    
    // Deapodization
    //std::cout << "compute_NFFT_C2NC(): call to deapodize()" << std::endl;
    deapodize( image );
    
    // FFT
    //std::cout << "compute_NFFT_C2NC(): call to fft()" << std::endl;
    fft( image, 1 ); //NFFT_FORWARDS
    
    // Convolution
    //std::cout << "compute_NFFT_C2NC(): call to convolve()" << std::endl;
    convolve( image, samples, 0x0, 1 ); //NFFT_CONV_C2NC
 }

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute_NFFTH_NC2C( hoNDArray<complext<T> > *samples, hoNDArray<complext<T> > *image )
{
    //* WRITE */ write_nd_array<complext<T> >(samples, "/home/karen/data/images/compute_NC2C_samples.cplx");
    // private method - no consistency check. We trust in ourselves.
    
    // Convolution
    //std::cout << "Compute_NFFTH_NC2C: size of image, samples: " << image->get_number_of_elements() << " \n";// << samples->get_number_of_elements() << std::endl;
    //std::cout << "compute_NFFTH_NC2C(): call to convolve()" << std::endl;
    convolve( samples, image, 0x0, 2 ); //NFFT_CONV_NC2C
    //* WRITE */ write_nd_array<complext<T> >(samples, "/home/karen/data/images/compute_NC2C_samples_convolved.cplx");
    //* WRITE */ write_nd_array<complext<T> >(image, "/home/karen/data/images/compute_NC2C_image_convolved.cplx");
    
    // FFT
    //std::cout << "compute_NFFTH_NC2C(): call to fft()" << std::endl;
    fft( image, 2 ); //NFFT_BACKWARDS
    //* WRITE */ write_nd_array<complext<T> >(image, "/home/karen/data/images/compute_NC2C_image_ffted.cplx");
    
    // Deapodization  
    //std::cout << "compute_NFFTH_NC2C(): call to deapodize()" << std::endl;
    deapodize( image );
    //* WRITE */ write_nd_array<complext<T> >(image, "/home/karen/data/images/compute_NC2C_image_deapoed.cplx");
}

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute_NFFTH_C2NC( hoNDArray<complext<T> > *image, hoNDArray<complext<T> > *samples )
{
    // private method - no consistency check. We trust in ourselves.
    
    // Deapodization
    //std::cout << "compute_NFFTH_C2NC(): call to deapodize()" << std::endl;
    deapodize( image, true );
    
    // FFT
    //std::cout << "compute_NFFTH_C2NC(): call to fft()" << std::endl;
    fft( image, 2 ); //NFFT_BACKWARDS
    
    // Convolution
    //std::cout << "compute_NFFTH_C2NC(): call to convolve()" << std::endl;
    convolve( image, samples, 0x0, 1 ); //NFFT_CONV_C2NC
}

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::compute_NFFT_NC2C( hoNDArray<complext<T> > *samples, hoNDArray<complext<T> > *image )
{
    // private method - no consistency check. We trust in ourselves.
    
    // Convolution
    //std::cout << "compute_NFFT_NC2C(): call to convolve()" << std::endl;
    convolve( samples, image, 0x0, 2 ); //NFFT_CONV_NC2C
    
    // FFT
    //std::cout << "compute_NFFT_NC2C(): call to fft()" << std::endl;
    fft( image, 1 ); //NFFT_FORWARDS
    
    // Deapodization
    //std::cout << "compute_NFFT_NC2C(): call to deapodize()" << std::endl;
    deapodize( image, true );
}



template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::convolve_NFFT_C2NC( hoNDArray<complext<T> > *image, hoNDArray<complext<T> > *samples, bool accumulate )
{
    // private method - no consistency check. We trust in ourselves.
    
    unsigned int num_coils = 1;
    for( unsigned int d=D; d<image->get_number_of_dimensions(); d++ )
	num_coils *= image->get_size(d);
    num_coils /= number_of_frames; //E.g. for receive coils. Then batches becomes number of coils. 

    
    //Setup grid and threads
    
    
    //size_t threads_per_block;
    //unsigned int max_coils;
    
    //threads_per_block = NFFT_THREADS_PER_KERNEL;
    
    /**if( cudaDeviceManager::Instance()->major_version(device) == 1 ){
	max_coils = NFFT_MAX_COILS_COMPUTE_1x;
    }
    else{
	max_coils = NFFT_MAX_COILS_COMPUTE_2x;
    }*/
  
    // We can (only) convolve max_coils batches per run due to shared memory issues. 
    //unsigned int domain_size_coils_desired = num_coils;
    //unsigned int num_repetitions = domain_size_coils_desired/max_coils + 
    //( ((domain_size_coils_desired%max_coils)==0) ? 0 : 1 );
    //unsigned int domain_size_coils = (num_repetitions==1) ? domain_size_coils_desired : max_coils;
    //unsigned int domain_size_coils_tail = (num_repetitions==1) ? domain_size_coils_desired : domain_size_coils_desired - (num_repetitions-1)*domain_size_coils;
    
    // Block and Grid dimensions
    //dim3 dimBlock( (unsigned int)threads_per_block );
    //dim3 dimGrid( (number_of_samples+dimBlock.x-1)/dimBlock.x, number_of_frames );
    
    // Calculate how much shared memory to use per thread
    //size_t bytes_per_thread = domain_size_coils * sizeof( vector_td<REAL,D> );
    //size_t bytes_per_thread_tail = domain_size_coils_tail * sizeof( vector_td<REAL,D> );
    
    //unsigned int double_warp_size_power=0;
    //unsigned int __tmp = cudaDeviceManager::Instance()->warp_size(device)<<1;
    //while(__tmp!=1){
    //	__tmp>>=1;
    //	double_warp_size_power++;
    //}
    
    vector_td<T,D> matrix_size_os_real = vector_td<T,D>( matrix_size_os );
    
    	
    // Number of reals to compute/output per thread
    const unsigned int num_reals = num_coils<<1; //number of receive coils times 2

    // Sample position to convolve onto
    // Computed in preprocessing, which included a wrap zone. Remove this wrapping.
    const vector_td<T,D> half_wrap_real = vector_td<T,D>(matrix_size_wrap>>1);
    
    for (int i = 0; i < number_of_samples; i++){ 
	
	const vector_td<T,D> sample_position = trajectory_positions->get_data_ptr()[i]-half_wrap_real;
    
	// Half the kernel width
	const vector_td<T,D> half_W_vec( W*T(0.5) );
  
	// Limits of the subgrid to consider
	const vector_td<int,D> lower_limit = vector_td<int,D>( ceil(sample_position-half_W_vec));
	const vector_td<int,D> upper_limit = vector_td<int,D>( floor(sample_position+half_W_vec));

	// Accumulate contributions from the grid
        //vector_td<int,D> grid_position = 
	//std::cout << "convolve_NFFT_C2NC(): call to NFFT_iterate()" << std::endl;
	NFFT_iterate<T>(alpha, beta, W, vector_td<unsigned int,D>(matrix_size_os), num_coils, image->get_data_ptr(), 
			samples->get_data_ptr(), W*T(0.5), T(1)/W,  matrix_size_os_real, sample_position, 
			lower_limit, upper_limit
			, number_of_frames, number_of_samples, i, accumulate 
			);
	/**
	// Calculate the distance between current sample and the grid cell
	vector_td<T,D> grid_position_real = vector_td<T,D>(grid_position);
	const vector_td<T,D> delta = abs(sample_position-grid_position_real);
	//const vector_td<T,D> half_W_vec( W*T(0.5) );
	
	// If cell too distant from sample then move on to the next cell
	if( weak_greater( delta, half_W_vec ))
	    return;
	
	// Compute convolution weight.
	KaiserBessel_operators<T> kb;
	const T weight = kb.KaiserBessel( delta, matrix_size_os_real, T(1)/W, beta );
	
	// Resolve wrapping of grid position
	vector_td<int,D> zero(0);
	grid_position += vector_less(grid_position, zero) * vector_td<int,D>(matrix_size_os);
	grid_position -= vector_greater_equal(grid_position, matrix_size_os) * vector_td<int,D>(matrix_size_os);
		
	for( unsigned int coil=0; coil<num_coils; coil++ ){
	    
	    // Read the grid cell value from global memory
	    const complext<T> grid_value = 
		image->get_data_ptr()[ (coil*number_of_frames) * prod(matrix_size_os)];// + co_to_idx<D>( vector_td<unsigned int, D>(grid_position), matrix_size_os ) ];
	    
	    
	    unsigned int out_idx = (coil*number_of_frames)*number_of_samples + i;
	    
	    complext<T> sample_value;
	    sample_value = weight*grid_value;//shared_mem[sharedMemFirstSampleIdx+(batch<<double_warp_size_power)];
	    
	    if( accumulate ) sample_value += samples->get_data_ptr()[out_idx];
	    samples->get_data_ptr()[out_idx] = sample_value;
	    }*/
	
    }

    /**
    for( unsigned int batch=0; batch<number_of_batches; batch++ ){
	complext<REAL>sample_value;
	sample_value.vec[0] = shared_mem[sharedMemFirstSampleIdx+(batch<<double_warp_size_power)];
	sample_value.vec[1] = shared_mem[sharedMemFirstSampleIdx+(batch<<double_warp_size_power)+warpSize];

	unsigned int out_idx = (batch*gridDim.y+blockIdx.y)*number_of_samples + globalThreadId;

	if( accumulate ) sample_value += samples[out_idx];
	samples[out_idx] = sample_value;
    }
    */

    //Invoke kernel
    
    /**
    for( unsigned int repetition = 0; repetition<num_repetitions; repetition++ ){
	NFFT_convolve_kernel<REAL,D>
	    <<<dimGrid, dimBlock, ((repetition==num_repetitions-1) ? dimBlock.x*bytes_per_thread_tail : dimBlock.x*bytes_per_thread)>>>
	    ( alpha, 
	      beta, 
	      W, 
	      vector_td<unsigned int,D>(matrix_size_os), 
	      vector_td<unsigned int,D>(matrix_size_wrap), 
	      number_of_samples,
	      (repetition==num_repetitions-1) ? domain_size_coils_tail : domain_size_coils, //num_coils
	      raw_pointer_cast(&(*trajectory_positions)[0]), 
	      image->get_data_ptr()+repetition*prod(matrix_size_os)*number_of_frames*domain_size_coils,
	      samples->get_data_ptr()+repetition*number_of_samples*number_of_frames*domain_size_coils, 
	      double_warp_size_power, 
	      REAL(0.5)*W, 
	      REAL(1)/(W), 
	      accumulate, 
	      matrix_size_os_real );
	
	CHECK_FOR_CUDA_ERROR();    
    } */
}

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::convolve_NFFT_NC2C( hoNDArray<complext<T> > *samples, hoNDArray<complext<T> > *image, bool accumulate )
{
    //* WRITE */ write_nd_array<complext<T> >(samples, "/home/karen/data/images/convolve_NC2C_input_samples.cplx");
    //_convolve_NFFT_NC2C<T,D>::apply( this, image, samples, accumulate );
    
    //std::cout << "Convolve_NFFT_NC2C: beginning" << std::endl;

    //std::cout << "convolve_NFFT_NC2C(): call to hoNFFT_plan()" << std::endl;
    hoNFFT_plan<T,D> *plan;
    
    //std::cout << "Convolve_NFFT_NC2C: after plan instantiation" << std::endl;

    //std::cout << "convolve_NFFT_NC2C(): call to KaiserBessel_operators()" << std::endl;
    KaiserBessel_operators<T> kb;

    //std::cout << "Convolve_NFFT_NC2C: after Kaiser Bessel" << std::endl;
    
    
    // Bring in some variables from the plan
    
    //unsigned int device = plan->device;
    //unsigned int number_of_frames = plan->number_of_frames;     
    //unsigned int number_of_samples = plan->number_of_samples;     
    //typename uint64d<D>::Type matrix_size_os = plan->matrix_size_os;     
    //typename uint64d<D>::Type matrix_size_wrap = plan->matrix_size_wrap;     
    //typename reald<T,D>::Type alpha = plan->alpha;     
    //typename reald<T,D>::Type beta = plan->beta;     
    //T W = plan->W;
    //thrust::device_vector< typename reald<T,D>::Type > *trajectory_positions = plan->trajectory_positions;    
    //std::cout << "Convolve_NFFT_NC2C: n_frames: " << number_of_frames << std::endl;
    //std::cout << "Convolve_NFFT_NC2C: n_samples: " << number_of_samples << std::endl;
    //std::cout << "Convolve_NFFT_NC2C: mat_s_os: " << matrix_size_os[0] << std::endl;
    //std::cout << "Convolve_NFFT_NC2C: mat_s_wrap: " << matrix_size_wrap[0] << std::endl;
    //std::cout << "Convolve_NFFT_NC2C: alpha: " << alpha[0] << std::endl;
    //std::cout << "Convolve_NFFT_NC2C: beta: " << beta[0] << std::endl;

    //std::cout << "Convolve_NFFT_NC2C: after extracting from plan" << std::endl;

    unsigned int num_coils = 1;
    for( unsigned int d=D; d<image->get_number_of_dimensions(); d++ )
	num_coils *= image->get_size(d);
    num_coils /= number_of_frames; //E.g. for receive coils. Then batches becomes number of coils. 
    
    vector_td<T,D> matrix_size_os_real = vector_td<T,D>( matrix_size_os );
    
    // Define temporary image that includes a wrapping zone
    hoNDArray<complext<T> > _tmp;
    
    vector<size_t> vec_dims = to_std_vector(matrix_size_os+matrix_size_wrap); //Size of the image, including the expansion due to the size of convolution kernel 
    if( number_of_frames > 1 )
	vec_dims.push_back(number_of_frames);
    if( num_coils > 1 ) //E.g. if more receive coils
	vec_dims.push_back(num_coils); //Adds an element to vec_dim which has the value of Num_batches. E.e. [128 128 128] with 30 receive coils gives [128 128 128 30]
    
    _tmp.create(&vec_dims); //Image dimensions

    //std::cout << "Convolve_NFFT_NC2C: after creating image dims" << std::endl;
    
    
    vector_td<unsigned int,D> grid_dims = vector_td<unsigned int,D>(matrix_size_os + matrix_size_wrap);
    
    const unsigned int number_of_grid_cells = prod(grid_dims); //Sum of all elements, im dims os and wrap and for all receive coils
    
    //std::cout << "Convolve_NFFT_NC2C: after number of grid cells" << std::endl;

    
    vector_td<T,D> half_W_vec( T(0.5)*W );
    int count_var = 0;
    int count_var2 = 0;
    //std::cout << "half_W_vec: " << half_W_vec[0];  for (int ii = 1; ii<D; ii++)  std::cout << ", " << half_W_vec[ii];  std::cout << std::endl;
    //std::cout << "traj_idx lenghth: " << traj_idx.get_number_of_elements() << std::endl;
    
    //for (int i = 0; i < traj_idx.get_number_of_elements(); i++){
    for (int i = 0; i < trajectory_positions->get_number_of_elements(); i++){
	count_var2 ++;
	//int idx = traj_idx.get_data_ptr()[i];
	//std::cout << "idx: " << idx << std::endl;
	vector_td<T,D> sample_pos = trajectory_positions->get_data_ptr()[i];
	
	for (int j = 0; j < num_cells; j++){
	    vector_td<unsigned int,D> cell_pos = grid_cell_pos.get_data_ptr()[i*num_cells+j];
	    //std::cout << "sample_pos: " << sample_pos << std::endl;
	    //std::cout << "cell_pos: " << cell_pos << std::endl;
	    
	    // Calculate the distance between the cell and the sample
	    vector_td<T,D> delta = abs(sample_pos-cell_pos);
	    //std::cout << "sample_pos: " << sample_pos[0];  for (int ii = 1; ii<D; ii++)  std::cout << ", " << sample_pos[ii];  std::cout << std::endl;
	    //std::cout << "cell_pos: " << cell_pos[0];  for (int ii = 1; ii<D; ii++)  std::cout << ", " << cell_pos[ii];  std::cout << std::endl;
	    
	    // Check if sample will contribute
	    if( weak_greater(delta, half_W_vec ))
		continue;
	    
	    count_var ++;
	    
	    //std::cout << "delta: " << delta[0];  for (int ii = 1; ii<D; ii++)  std::cout << ", " << delta[ii];  std::cout << std::endl;
	    //std::cout << "delta: " << delta  << std::endl;
	    //std::cout << "convolve_NFFT_NC2C: after continue" << std::endl;
	    
	    // Compute convolution weights
	    //std::cout << "convolve_NFFT_NC2C(): call to KaiserBessel()" << std::endl;
	    T weight = kb.KaiserBessel( delta, matrix_size_os_real, T(1)/W, beta );
	    //std::cout << weight << std::endl;
	    
	    // Safety measure
	    //if( !isfinite(weight) )
	    //continue;
	    
	    int im_idx = 0;
	    unsigned int block_size = 1;
	    for (unsigned int d = 0; d < D; d++) {
		im_idx += (block_size*cell_pos[d]);
		block_size *= (matrix_size_os[d]+matrix_size_wrap[d]);
	    }
	    
	    //im_idx = cell_pos[0] + cell_pos[1]*grid_dims[0];
	    //if( (D == 4 && num_coils > 0) || (D == 3 && num_coils == 0 ) )
	    //	im_idx += cell_pos[2]*grid_dims[1]*grid_dims[0];
	    
	    // Apply Kaiser-Bessel filter to input images
	    for( unsigned int coil = 0; coil < num_coils; coil++ ){
		
		complext<T> sample_val = samples->get_data_ptr()[i + coil * number_of_samples];
		//std::cout << sample_val << std::endl;
		
		// Apply Kaiser Bessel weight to ssample and adding to image matrix _tmp
		
		_tmp[coil*number_of_grid_cells + im_idx] += weight*sample_val;
	    }	    
	}
    }
    
    //* WRITE */ write_nd_array<complext<T> >(&_tmp, "/home/karen/data/images/tmp_image.cplx");
    //std::cout << "count_var: " << count_var << std::endl;
    //std::cout << "count_var2: " << count_var2 << std::endl,
	
    //std::cout << "Convolve_NFFT_NC2C: after for loop" << std::endl;

    //std::cout << "Convolve_NFFT_NC2C: accumulate: " << accumulate << std::endl;
   
    //plan->image_wrap( &_tmp, image, accumulate );
    //std::cout << "convolve_NFFT_NC2C(): call to image_wrap()" << std::endl;
    image_wrap( &_tmp, image, accumulate );
}; 


// Image wrap 

template<class T, unsigned int D> void Gadgetron::hoNFFT_plan<T,D>::image_wrap( hoNDArray<complext<T> > *source, hoNDArray<complext<T> > *target, bool accumulate )
{

    //std::cout << "Image wrap: first line" << std::endl;
    //std::cout << "Image wrap: D: " << D << std::endl;
    //std::cout << "Image wrap: src n_dims: " << source->get_number_of_dimensions() << std::endl;
    //std::cout << "Image wrap: n_frames: " << number_of_frames << std::endl;
    //std::cout << "Image wrap: src size(0): " << source->get_size(0) << std::endl; 


    unsigned int num_coils = 1;
    for( unsigned int d=D; d<source->get_number_of_dimensions(); d++ ){
	//std::cout << "Image wrap: src size(" << d << "): " << source->get_size(d) << std::endl; 
	num_coils *= source->get_size(d);
    }
    //std::cout << "Image wrap: after for loop for n_coils: " << num_coils << std::endl;
    num_coils /= number_of_frames;

    //std::cout << "Image wrap: after n_coils" << std::endl;
    
    //unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int num_elements_per_image_src = prod(matrix_size_os+matrix_size_wrap);
    //const unsigned int image_offset_src = blockIdx.y*num_elements_per_image_src;

    //std::cout << "Image wrap: after n_elem_im_src" << std::endl;

    for (int coil = 0; coil < num_coils; coil++){
	const unsigned int image_offset_src = coil * num_elements_per_image_src;
    
	for (int idx = 0; idx < prod(matrix_size_os); idx++){
	
	    //const typename uintd<D>::Type co = idx_to_co<D>(idx, matrix_size_os);
	    typename uint64d<D>::Type co;
	    int i_tmp = idx;
	    for(int d = 0; d < D; d++){
		co[d] = i_tmp % matrix_size_os[d];
		i_tmp -= co[d];
		i_tmp /= matrix_size_os[d];
	    }
	    const typename uint64d<D>::Type half_wrap(matrix_size_wrap>>1);
	
	    // Make "boolean" vectors denoting whether wrapping needs to be performed in a given direction (forwards/backwards)
	    vector_td<bool,D> B_l = vector_less( co, half_wrap );
	    vector_td<bool,D> B_r = vector_greater_equal( co, matrix_size_os-half_wrap);
	
	    complext<T>  result = source->get_data_ptr()[co_to_idx<D>(typename uint64d<D>::Type(co+half_wrap), matrix_size_os+matrix_size_wrap) + image_offset_src];
	
	    if( sum(B_l+B_r) > 0 ){
	    
		// Fold back the wrapping zone onto the image ("periodically")
		//
		// There is 2^D-1 ways to pick combinations of dimensions in D-dimensionsal space, e.g. 
		// 
		//  { x, y, xy } in 2D
		//  { x, y, x, xy, xz, yz, xyz } in 3D
		//
		// Every "letter" in each combination provides two possible wraps (eiher end of the dimension)
		// 
		// For every 2^D-1 combinations DO
		//   - find the number of dimensions, d, in the combination
		//   - create 2^(d) stride vectors and test for wrapping using the 'B'-vectors above.
		//   - accumulate the contributions
		// 
		//   The following code represents dimensions as bits in a char.
		//
	    
		for( unsigned char combination = 1; combination < (1<<D); combination++ ){
		
		    // Find d
		    unsigned char d = 0;
		    for( unsigned char i=0; i<D; i++ )
			d += ((combination & (1<<i)) > 0 );
		
		    // Create stride vector for each wrapping test
		    for( unsigned char s = 0; s < (1<<d); s++ ){
		    
			// Target for stride
			typename intd<D>::Type stride;
			char wrap_requests = 0;
			char skipped_dims = 0;
		    
			// Fill dimensions of the stride
			for( unsigned char i=1; i<D+1; i++ ){
			
			    // Is the stride dimension present in the current combination?
			    if( i & combination ){
			    
				// A zero bit in s indicates "check for left wrap" and a one bit is interpreted as "check for right wrap" 
				// ("left/right" for the individual dimension meaning wrapping on either side of the dimension).
			    
				if( i & (s<<(skipped_dims)) ){
				    if( B_r.vec[i-1] ){ // Wrapping required 
					stride[i-1] = -1;
					wrap_requests++;
				    }
				    else
					stride[i-1] = 0;
				}
				else{ 
				    if( B_l.vec[i-1] ){ // Wrapping required 
					stride[i-1] =1 ;
					wrap_requests++;
				    }
				    else
					stride[i-1] = 0;
				}
			    }
			    else{
				// Do not test for wrapping in dimension 'i-1' (for this combination)
				stride[i-1] = 0;
				skipped_dims++;
			    }
			}
		    
		    
			// Now it is time to do the actual wrapping (if needed)
			if( wrap_requests == d ){
			    typename intd<D>::Type src_co_int = vector_td<int,D>(co+half_wrap);
			    typename intd<D>::Type matrix_size_os_int = vector_td<int,D>(matrix_size_os);
			    typename intd<D>::Type co_offset_int = src_co_int + component_wise_mul<int,D>(stride,matrix_size_os_int);
			    typename uintd<D>::Type co_offset = vector_td<unsigned int,D>(co_offset_int);
			    result += source->get_data_ptr()[co_to_idx<D>(typename uint64d<D>::Type(co_offset), matrix_size_os+matrix_size_wrap)];// + image_offset_src];
			    //break; // only one stride per combination can contribute (e.g. one edge, one corner)
			}
		    } 
		}
	    }
	
	    // Output
	    //const unsigned int image_offset_tgt = blockIdx.y*prod(matrix_size_os);
	    const unsigned int image_offset_tgt = coil * prod(matrix_size_os);
	    if( accumulate ) result += target->get_data_ptr()[idx+image_offset_tgt];
	    target->get_data_ptr()[idx+image_offset_tgt] = result;
	}
    }

    //std::cout << "Image wrap: after outer for loop" << std::endl;

}


//
// Template instantion
//


template class hoNFFT_plan<float,  1>;
template class hoNFFT_plan<double, 1>;

template class hoNFFT_plan<float,  2>;
template class hoNFFT_plan<double, 2>;

template class hoNFFT_plan<float,  3>;
template class hoNFFT_plan<double, 3>;

template class hoNFFT_plan<float,  4>;
template class hoNFFT_plan<double, 4>;

