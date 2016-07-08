//#ifndef GIRF_TRAJECTORY_CORRECTION_H
//#define GIRF_TRAJECTORY_CORRECTION_H

#include <iostream>
#include <complex>

#include "math.h"
#include "hoNDArray.h"
#include "hoNDArray_elemwise.h"
#include "hoNFFT.h"
#include "hoNDFFT.h"

#include "hoNDArray_utils.h"
#include "vector_td_utilities.h"
#include "hoNDArray_reductions.h"


using namespace Gadgetron;

namespace Gadgetron{
    template <class T, unsigned int D> class girf_trajectory_correction
    {
    public: 
	girf_trajectory_correction(unsigned int num_profiles, 
				   hoNDArray<T> rotation_matrix, 
				   T dt, 
				   unsigned int base_resolution,
				   unsigned int num_tr_history, 
				   unsigned int num_samples_to_include_prewinder,
				   hoNDArray<T> *adc);

	~girf_trajectory_correction();

	void calculate_adc_activation_times(hoNDArray<T> *adc);
    
	hoNDArray<typename reald<T,D>::Type> trajctory_prediction(hoNDArray<T> *gradient_in, hoNDArray<complext<T> > *girf);

	void girf_convolution_tr(hoNDArray<T> *gradient_in, hoNDArray<complext<T> > *girf, hoNDArray<T> *gradient_out);

	void gradient_to_kspace_pos( hoNDArray<T> *gradient_in, hoNDArray<T> *kspace_out );

	hoNDArray<T> resample_to_acquisition_fs(hoNDArray<T> *kspace_profile);

        void resample_to_dwell_time(hoNDArray<T> *kspace_profile_point_one_us, hoNDArray<T> *kspace_profile_dwell_time);

	void cut_last_array_dim(hoNDArray<T> *array_in, hoNDArray<T> *array_out, unsigned int start_idx, unsigned int l);

        void linear_interpolation(hoNDArray<complext<T> > *original_array, unsigned int N_new, hoNDArray<complext<T> > *new_array);

    private:
	unsigned int profiles;
	unsigned int tr_history;
	unsigned int samples_to_include_prewinder;
	T dwell_time;
	unsigned int base_resolution;
	T adc_shift;
	T gamma;
	hoNDArray<T> rotation_matrix;
	unsigned int adc_duration;
	hoNDArray<unsigned int> adc_active_start;
	hoNDArray<unsigned int> adc_active_end;
	hoNDArray<typename reald<T,D>::Type> trajectory_out;
    };
}

//#endif //GIRF_TRAJECTORY_CORRECTION_H
