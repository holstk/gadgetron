#include "girf_trajectory_correction.h"

using namespace Gadgetron;

template <class T, unsigned int D> Gadgetron::girf_trajectory_correction<T,D>::girf_trajectory_correction(unsigned int num_profiles, 
													  hoNDArray<T> rotation_matrix, 
													  T dt, //Oversampled dwell time
													  unsigned int base_resolution, //Oversampled
													  unsigned int num_tr_history, 
													  unsigned int num_samples_to_include_prewinder,
													  hoNDArray<T> *adc)
{
    profiles = num_profiles;
    tr_history = num_tr_history;
    samples_to_include_prewinder = num_samples_to_include_prewinder;

    dwell_time = dt;
    adc_shift = -dwell_time - 0.85*1e-6;

    gamma = 2.675222005*1e8;

    calculate_adc_activation_times(adc);
    
}


template <class T, unsigned int D> Gadgetron::girf_trajectory_correction<T,D>::~girf_trajectory_correction()
{
    //Nothing yet...
}


template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::calculate_adc_activation_times(hoNDArray<T> *adc)
{
    //acd is a 1d vector

    adc_active_start.create(profiles);
    adc_active_end.create(profiles);

    unsigned int profile_count = 0;
    
    unsigned int adc_duration_control = 0;
    for (unsigned int i = 0; i < adc->get_number_of_elements()-1; i++){


	if ( (adc->get_data_ptr()[i+1] - adc->get_data_ptr()[i]) > 0 ){
	    adc_active_start[profile_count] = i+1;
	}
	else if ( (adc->get_data_ptr()[i+1] - adc->get_data_ptr()[i]) < 0 ){
	    //start_end_pair[1] = i;
	    adc_active_end[profile_count] = i;
	    //adc_active[count] = start_end_pair;
	    
	    adc_duration = adc_active_start[profile_count] - adc_active_end[profile_count] + i;
	    
	    if (i == 0)
		adc_duration_control = adc_duration;
	    
	    if ( !(adc_duration - adc_duration_control == 0) )
		throw std::runtime_error("ADC duration must not change over time!");

	    if ( !(adc->get_data_ptr()[adc_active_start[profile_count]] == 1.0) || !(adc->get_data_ptr()[adc_active_end[profile_count]] == 1.0) )
	    	throw std::runtime_error("ADC must not be 0 for the start and end index!");
		 
	    profile_count++;    

	}
    }
}


template <class T, unsigned int D> hoNDArray<typename reald<T,D>::Type> Gadgetron::girf_trajectory_correction<T,D>::trajctory_prediction(hoNDArray<T> *gradient_in, hoNDArray<complext<T> > *girf)
{
   
    hoNDArray<T> gradient_history;
    unsigned int history_start;
    unsigned int history_samples;

    hoNDArray<T> gradient_incl_prewinder;
    unsigned int incl_prewinder_start;

    hoNDArray<T> kspace_tr;

    typename reald<T,D>::Type sample_pos;
    trajectory_out.create(profiles*base_resolution);

    for (unsigned int p = 0; p < profiles; p++){

	// Start point for gradient history
	if ( p < tr_history ){
	    history_start = adc_active_start[0] - samples_to_include_prewinder;
	    history_samples = (p+1) * adc_duration + samples_to_include_prewinder;
	}
	else {
	    history_start = adc_active_start[p-tr_history+1] - samples_to_include_prewinder;
	    history_samples = tr_history * adc_duration + samples_to_include_prewinder;
	}
	
	// Extract the gradient history for the particulat tr
	cut_last_array_dim( gradient_in, &gradient_history, history_start, history_samples );
	
	// Convolve the theoretical gradient history with the gradient inpulse response function
        girf_convolution_tr( &gradient_history, girf, &gradient_history );

	// Extract only the tr = p and the prewinder samples from the predicted gradient history
 	incl_prewinder_start = ( adc_active_start[p] - samples_to_include_prewinder ) - history_start;
	cut_last_array_dim( &gradient_history, &gradient_incl_prewinder, incl_prewinder_start, adc_duration+samples_to_include_prewinder );

	// Integrate the gradients to acquire the actual k-space position for each sample
	gradient_to_kspace_pos(&gradient_incl_prewinder, &kspace_tr);

	// Cut kspace to only include the part where the ADC is active
	cut_last_array_dim(&kspace_tr, &kspace_tr, samples_to_include_prewinder, adc_duration);

	// Resample to match acquisition (currently in 0.1 us)
	hoNDArray<T> profile_resampled_point_one_us = resample_to_acquisition_fs(&kspace_tr);

	// Resample to match dwell time (in number 0.1 units, e.g. dwell_time = 2.8us = 28 units)
	hoNDArray<T> kspace_profile_dwell_time;
	resample_to_dwell_time(&profile_resampled_point_one_us, &kspace_profile_dwell_time);
	
	for (unsigned int j = 0; j < base_resolution; j++){
	    sample_pos[0] = kspace_profile_dwell_time[0*base_resolution + j];
	    sample_pos[1] = kspace_profile_dwell_time[1*base_resolution + j];
	    sample_pos[2] = kspace_profile_dwell_time[2*base_resolution + j];
	    trajectory_out[p * base_resolution + j] = sample_pos;
	}
    }

    return trajectory_out;
    
}


template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::girf_convolution_tr(hoNDArray<T> *gradient_in, hoNDArray<complext<T> > *girf, hoNDArray<T> *gradient_out)
{
    hoNDArray<T> gradient_raster({gradient_in->get_size(0), gradient_in->get_size(1)/10});

    unsigned int samples_girf = girf->get_size(1);
    unsigned int samples_gradient = gradient_in->get_size(1)/10;

    for (unsigned int j = 0; j < samples_gradient; j++){
	for (unsigned int d = 0; d < 2; d++){
	    gradient_raster[d*samples_gradient + j] = gradient_in->get_data_ptr()[d*gradient_in->get_size(1) + j*10];
	}
    }

    if (samples_girf/2 < samples_gradient){
	linear_interpolation(girf, 2*samples_gradient, girf);
	samples_girf = 2*samples_gradient;
    }
    
    if (samples_girf/2 > samples_gradient){
	pad<T,2>( {3, samples_girf/2}, &gradient_raster, &gradient_raster );
	samples_gradient = samples_girf/2;
    }

    unsigned int samples_mirrored = samples_girf;
    hoNDArray<complext<T> > gradient_mirrored(3,samples_mirrored);

    for (unsigned int j = 0; j < samples_mirrored/2; j++){
	for (unsigned int d = 0; d < 2; d++){
	    gradient_mirrored[ d*samples_mirrored  +  samples_mirrored/2 - j - 1 ] = gradient_raster[ d * samples_mirrored + j ];
	    gradient_mirrored[ d*samples_mirrored  +  samples_mirrored/2 + j ]     = gradient_raster[ d * samples_mirrored + j ];
	}
    }

    hoNDFFT<T>::instance()->fft( &gradient_mirrored, 1 );

    unsigned int samples_mirrored_us = samples_mirrored*10;
    unsigned int trunc = (samples_mirrored_us - samples_mirrored) / 2;

    hoNDArray<complext<T> > gradient_convolved(3, samples_mirrored_us);
    unsigned int shift = samples_mirrored_us / 2;

    unsigned int count = 0;
    complext<T> p;

    for (unsigned int j = 0; j < samples_mirrored_us; j++){
	for (unsigned int d = 0; d < 2; d++){
	    if (j > (trunc-1)  &&  j < (trunc + samples_mirrored)){
		p = gradient_mirrored[d*samples_mirrored + count] * girf->get_data_ptr()[d*samples_mirrored + count];
		count ++;
	    }
	    else{  
		p = 0.0;
	    }

	    if (j < shift){
		gradient_convolved[d*samples_mirrored_us  +  shift+j] = p;
	    }
	    else{
		gradient_convolved[d*samples_mirrored_us  +  -shift+j] = p;
	    }
	}
    }

    hoNDFFT<T>::instance()->ifft( &gradient_convolved, 1 );    
    
    gradient_out->create(gradient_convolved.get_dimensions());

    for (unsigned int j = 0; j < samples_mirrored_us; j++){
	for (unsigned int d = 0; d < 2; d++){
	    p = gradient_convolved[d*samples_mirrored_us + j];
	    if (imag(p) > 0)
		gradient_out->get_data_ptr()[d*samples_mirrored_us + j] = abs(p);
	    else
		gradient_out->get_data_ptr()[d*samples_mirrored_us + j] = -abs(p);
	}
    }
}


template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::gradient_to_kspace_pos( hoNDArray<T> *gradient_in, hoNDArray<T> *kspace_out )
{
    std::vector<size_t> dims = *gradient_in->get_dimensions();
    kspace_out->create(dims);
    
    kspace_out->get_data_ptr()[ 0 * dims[1] ] = gradient_in->get_data_ptr()[ 0 * dims[1] ]; 
    kspace_out->get_data_ptr()[ 1 * dims[1] ] = gradient_in->get_data_ptr()[ 1 * dims[1] ];
    kspace_out->get_data_ptr()[ 2 * dims[1] ] = gradient_in->get_data_ptr()[ 2 * dims[1] ];

   for (unsigned int j = 1; j < dims[1]; j++){
       kspace_out->get_data_ptr()[ 0 * dims[1] + j ] = kspace_out->get_data_ptr()[ 0 * dims[1] + j-1 ] + gradient_in->get_data_ptr()[ 0 * dims[1] + j ]; 
       kspace_out->get_data_ptr()[ 1 * dims[1] + j ] = kspace_out->get_data_ptr()[ 1 * dims[1] + j-1 ] + gradient_in->get_data_ptr()[ 1 * dims[1] + j ]; 
       kspace_out->get_data_ptr()[ 2 * dims[1] + j ] = kspace_out->get_data_ptr()[ 2 * dims[1] + j-1 ] + gradient_in->get_data_ptr()[ 2 * dims[1] + j ]; 
   }
}


template <class T, unsigned int D> hoNDArray<T> Gadgetron::girf_trajectory_correction<T,D>::resample_to_acquisition_fs(hoNDArray<T> *kspace_profile)
{
    std::vector<size_t> dims = *kspace_profile->get_dimensions();
    unsigned int samples_mirrored = dims[1] * 2;

    T dw = 1 / (samples_mirrored * 1e-6);
    T w;
    hoNDArray<complext<T> > adc_shift_fft(samples_mirrored);
    for (unsigned int k = 0; k < samples_mirrored; k++){
    	w = (T)k*dw;
	adc_shift_fft[k] = exp( complext<T>( 0, -2.0*M_PI*adc_shift*w ) );
    }

    hoNDArray<complext<T> > profile_mirrored(3,samples_mirrored);
    
    for (unsigned int j = 0; j < samples_mirrored/2; j++){
	for (unsigned int d = 0; d < 2; d++){
	    profile_mirrored[ d*samples_mirrored  +  samples_mirrored/2 - j - 1 ] = kspace_profile->get_data_ptr()[ d * samples_mirrored + j ];
	    
	    profile_mirrored[ d*samples_mirrored  +  samples_mirrored/2 + j ] = kspace_profile->get_data_ptr()[ d * samples_mirrored + j ];
	}
    }

    hoNDFFT<T>::instance()->fft( &profile_mirrored, 1 );

    unsigned int trunc = (samples_mirrored*10 - samples_mirrored) / 2;
    unsigned int shift = samples_mirrored*10 / 2;
    hoNDArray<complext<T> > profile_shifted_adc(3, samples_mirrored*10);

    complext<T> p;
    unsigned int count = 0;
    for (unsigned int j = 0; j < samples_mirrored*10; j++){
	for (unsigned int d = 0; d < 2; d++){
	    if (j > (trunc-1)  &&  j < (trunc + samples_mirrored)){
		p = profile_mirrored[d*samples_mirrored + count] * adc_shift_fft[count];
		count ++;
	    }
	    else{  
		p = 0.0;
	    }
	    
	    if (j < shift){
		profile_shifted_adc[d*10*samples_mirrored  +  shift+j] = p;
	    }
	    else{
		profile_shifted_adc[d*10*samples_mirrored  +  -shift+j] = p;
	    }
	}
    }
    hoNDFFT<T>::instance()->ifft( &profile_shifted_adc, 1 );

    hoNDArray<T> profile_resampled({3,10*samples_mirrored});
    for (unsigned int j = 0; j < 10*samples_mirrored; j++){
	for (unsigned int d = 0; d < 2; d++){
	    complext<T> p = profile_shifted_adc[d*10*samples_mirrored + j];
	    if (real(p) > 0)
		profile_resampled[d*10*samples_mirrored + j] = abs(p);
	    else
		profile_resampled[d*10*samples_mirrored + j] = -abs(p);
	}
    }

    return profile_resampled;
}


template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::resample_to_dwell_time(hoNDArray<T> *kspace_profile_point_one_us, hoNDArray<T> *kspace_profile_dwell_time)
{
    int dwell_time_point_one_us = (int)(dwell_time * 1e7);
    kspace_profile_dwell_time->create({3,base_resolution});
    for (unsigned int j = 0; j < base_resolution; j++){
	for(unsigned int d = 0; d < 2; d++){
	    kspace_profile_dwell_time->get_data_ptr()[d*base_resolution + j] =
		kspace_profile_point_one_us->get_data_ptr()[d*kspace_profile_point_one_us->get_size(1) + j*dwell_time_point_one_us];
	}
    }
}

template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::cut_last_array_dim(hoNDArray<T> *array_in, hoNDArray<T> *array_out, unsigned int start_idx, unsigned int l)
{
    std::vector<size_t> array_dims_in = *array_in->get_dimensions();
    std::vector<size_t> array_dims_out = { array_dims_in[0], l};

    array_out->create(array_dims_out);
    
    for (unsigned int j = 0; j < l; j++){
	for (unsigned int d = 0; d < 3; d++){
	    array_out->get_data_ptr()[d*array_dims_out[1] + j] = array_in->get_data_ptr()[d*array_dims_in[1] + start_idx + j];
	}
    }

}


template <class T, unsigned int D> void Gadgetron::girf_trajectory_correction<T,D>::linear_interpolation(hoNDArray<complext<T> > *original_array, unsigned int N_new, hoNDArray<complext<T> > *new_array)
{
    if ( original_array->get_number_of_dimensions() > 2 )
	throw std::runtime_error("Interpolation array has to be 2-dimensional!");

    if ( !( original_array->get_size(0) == 3 ) )
	throw std::runtime_error("Interpolation array has to include an x, y, and z axix!");

    unsigned int N_orig = original_array->get_size(1);

    if (N_new < N_orig)
	throw std::runtime_error("Linear interpolation only implemented for input smaller than output.");

    new_array->create({3,N_new});

    unsigned int n1, n2;
    complext<T> p1, p2, pn;
    T n;
    T fraction;
    for (unsigned int j = 0; j < N_new; j++){

	n = (T)j / (T)(N_new-1) * (T)(N_orig-1);
	fraction = n - std::floor(n);
	n1 = std::floor(n);
	n2 = n1 + 1;

	for (unsigned int d = 0; d < 2; d++){
	    if (j == N_new-2)
		pn =  original_array->get_data_ptr()[d*N_orig + N_orig-1];
	    else{
		p1 = original_array->get_data_ptr()[d*N_orig + n1];
		p2 = original_array->get_data_ptr()[d*N_orig + n2];
		pn = (p2 - p1) * fraction + p1;
	    }
	    new_array->get_data_ptr()[d*N_new + j] = pn;
	}
    }
 }


template class girf_trajectory_correction<float,2>;
template class girf_trajectory_correction<float,3>;

template class girf_trajectory_correction<double,2>;
template class girf_trajectory_correction<double,3>;
