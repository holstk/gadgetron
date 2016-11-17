
#include "cpuNFFTGadget3d_bin.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"
#include "hoNFFT.h"
#include "NFFT.h"

#include "hoNDArray_elemwise.h"
#include "hoNDArray_utils.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"
#include "GadgetronTimer.h"
#include "complext.h"
#include "hoNDArray_reductions.h"
#include "rotate_kspace.h"

//#include "cpu_b1_map.h"

//#include "hoNonCartesianSenseOperator.h"
//#include "hoCgPreconditioner.h"
//#include "hoImageOperator.h"
//#include "hoCgSolver.h"

namespace Gadgetron{

cpuNFFTGadget3d_bin::cpuNFFTGadget3d_bin()
  : image_counter_(0)
{
}


int cpuNFFTGadget3d_bin::process( GadgetContainerMessage<IsmrmrdReconData>* m1)
{
    GadgetronTimer nfft_clock;
    std::string data_set = "150708/A/";
    int bin = 1;
    std::cout << data_set << std::endl;
    std::string data_path = "/mnt/scratch/karen/shared_folder/" + data_set;
    std::string save_path = data_path + "/go/bin" + std::to_string(bin) + "/";
    bool do_save = true;


    //Iterate over all the recon bits
    for(std::vector<IsmrmrdReconBit>::iterator it = m1->getObjectPtr()->rbit_.begin();
        it != m1->getObjectPtr()->rbit_.end(); ++it)
    {
	std::cout << "In 3d bin NFFT gadget" << std::endl;

	
	// ============================================== //
	// ======== Initial general gadget steps ======== //
	// ============================================== //

        //Grab a reference to the buffer containing the imaging data
        //We are ignoring the reference data
        IsmrmrdDataBuffered & dbuff = it->data_;

        //Data 7D, fixed order [E0, E1, E2, CHA, N, S, LOC]
        uint16_t E0 = dbuff.data_.get_size(0);
        uint16_t E1 = dbuff.data_.get_size(1);
        uint16_t E2 = dbuff.data_.get_size(2);
        uint16_t CHA = dbuff.data_.get_size(3);
        uint16_t N = dbuff.data_.get_size(4);
        uint16_t S = dbuff.data_.get_size(5);
        uint16_t LOC = dbuff.data_.get_size(6);
      

        //Create an image array message
        GadgetContainerMessage<IsmrmrdImageArray>* cm1 = 
                new GadgetContainerMessage<IsmrmrdImageArray>();


        //Grab references to the image array data and headers
        IsmrmrdImageArray & imarray = *cm1->getObjectPtr();

        std::vector<size_t> data_dims(7);
        data_dims[0] = E0;
        data_dims[1] = E1;
        data_dims[2] = E2;
        data_dims[3] = 1;
        data_dims[4] = N;
        data_dims[5] = S;
        data_dims[6] = LOC;        

	std::cout << "E0, E1, E2, CHA: " << E0 << ", " << E1 << ", " << E2 << ", " << CHA << std::endl;




	// ============================================== //
	// ========= Define constants and image ========= //
	// ============================================== //

        constexpr unsigned int dim = 3;
	grid_oversampling_factor = 2.0; //alpha
	ro_oversampling_factor = 2.0;
	kernel_width = 5.5;  //original: 5.5;
	cut_off = 0;


        num_samples_per_spoke = E0;
	num_spokes = E1 * E2;


	im_size = num_samples_per_spoke/(int)ro_oversampling_factor;
	im_size_os = std::ceil(im_size * grid_oversampling_factor);

	matrix_size = typename uint64d<dim>::Type(im_size);
	matrix_size_os = typename uint64d<dim>::Type(im_size_os);
	alpha = (float)matrix_size_os[0] / (float)matrix_size[0];

	//Image matrix - dimension dependent
	imarray.data_.create(matrix_size[0], matrix_size[1], matrix_size[2], data_dims[3], data_dims[4], data_dims[5], data_dims[6]);
        
	std::vector<size_t> image_dims = to_std_vector(matrix_size);
	image_dims.push_back(CHA);
	hoNDArray< complext<float> > image(&image_dims);
	std::vector<size_t> output_dims = to_std_vector(matrix_size);




	// ============================================== //
	// ========== Reading in spoke indexes ========== //
	// ============================================== //


	std::string spoke_idx_filename = data_path + "acq_idx/acq_idx_" + std::to_string(bin) + ".idx";
	std::cout << spoke_idx_filename << std::endl;
	std::ifstream in_acq(spoke_idx_filename.c_str(), std::ios::binary);

	in_acq.read((char*)&num_spokes, sizeof(long int));

	long int *spoke_idx;
	spoke_idx = new long int[num_spokes];
	in_acq.read((char*)spoke_idx, num_spokes*sizeof(long int));

	in_acq.close();

	if (do_save){
	    hoNDArray<long int> go(num_spokes, (long int*) spoke_idx);
	    std::string save_name = save_path + "spoke_idx.long";
	    write_nd_array<long int>(&go, save_name.c_str());
	}

	num_samples_per_spoke = E0;






	// ============================================== //
	// ============= Trajectory and DCF ============= //
	// ============================================== //

	//Calculate the trajectory and DCF for recon	
	GATrajectory<float,dim> traj;

	hoNDArray<typename reald<float,dim>::Type> trajectory;
	try{ trajectory = traj.calculateTrajectory(num_spokes, num_samples_per_spoke, dim, spoke_idx, 0);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}
	    
	try{  DCF = traj.calculateDCF(num_spokes, num_samples_per_spoke, dim, cut_off, im_size);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}

	if (do_save){
	    std::vector<size_t> trajectory_dims(2); 
	    trajectory_dims[0] = dim; 
	    trajectory_dims[1] = num_spokes*num_samples_per_spoke;
	    hoNDArray<float> go(trajectory_dims);
	    for (unsigned int ii = 0; ii < trajectory.get_number_of_elements(); ii++){
	    	typename reald<float,dim>::Type tmp = trajectory(ii);
		go[ii*3+0] = tmp[0];
		go[ii*3+1] = tmp[1];
		go[ii*3+2] = tmp[2];
	    }
	    std::string save_name = save_path + "trajectory.real";
	    write_nd_array<float>(&go, save_name.c_str());
	}
	if (do_save){
	    //hoNDArray<float> go(trajectory.get_dimensions(), (float*) trajectory.begin());
	    std::string save_name = save_path + "dcw.real";
	    write_nd_array<float>(&DCF, save_name.c_str());
	}


	std::vector<size_t> rotation_dims(2);
	rotation_dims[0] = 3; rotation_dims[1] = 3;
	hoNDArray<float> rotation(&rotation_dims);
	/*rotation[0] = 0; 
	rotation[1] = 1; 
	rotation[2] = 0; 
	rotation[3] = 1; 
	rotation[4] = 0; 
	rotation[5] = 0; 
	rotation[6] = 0; 
	rotation[7] = 0; 
	rotation[8] = 1;*/
	rotation[0] =  0.87; 
	rotation[1] = -0.32; 
	rotation[2] = -0.37; 
	rotation[3] =  0.49; 
	rotation[4] =  0.60; 
	rotation[5] =  0.64; 
	rotation[6] =  0.01; 
	rotation[7] = -0.73; 
	rotation[8] =  0.68;
	
	rotate_kspace<float> rot;
	rot.rotate_trajectory(&trajectory, rotation);

	if (do_save){
	    std::vector<size_t> trajectory_dims(2); 
	    trajectory_dims[0] = dim; 
	    trajectory_dims[1] = num_spokes*num_samples_per_spoke;
	    hoNDArray<float> go(trajectory_dims);
	    for (unsigned int ii = 0; ii < trajectory.get_number_of_elements(); ii++){
	    	typename reald<float,dim>::Type tmp = trajectory(ii);
		go[ii*3+0] = tmp[0];
		go[ii*3+1] = tmp[1];
		go[ii*3+2] = tmp[2];
	    }
	    std::string save_name = save_path + "trajectory_rotated.real";
	    write_nd_array<float>(&go, save_name.c_str());
	}




	// ============================================== //
	// ============ Initiating NFFT plan ============ //
	// ============================================== //

	//Setting up the plan
	hoNFFT_plan<float,dim> plan( matrix_size, matrix_size_os, kernel_width );
        int NFFT_prep_mode = 2; //1=NFFT_PREP_C2NC, 2=NFFT_PREP_NC2C, 4=NFFT_PREP_ALL
	plan.preprocess( &trajectory, NFFT_prep_mode );






	// ============================================== //
	// ========== Looping over N, S and LOC ========= //
	// ============================================== //


        //ImageHeaders will be [N, S, LOC]
        std::vector<size_t> header_dims(3);
        header_dims[0] = N;
        header_dims[1] = S;
        header_dims[2] = LOC;        
        imarray.headers_.create(&header_dims);
 
        for (uint16_t loc=0; loc < LOC; loc++) {
            for (uint16_t s=0; s < S; s++) {                
                for (uint16_t n=0; n < N; n++) {
                    

                    //Set some information into the image header
                    //Use the middle acquisition header for some info
                    //[E1, E2, N, S, LOC]
                    ISMRMRD::AcquisitionHeader & acqhdr = dbuff.headers_(dbuff.sampling_.sampling_limits_[1].center_,
                                                                         dbuff.sampling_.sampling_limits_[2].center_,
                                                                         n, s, loc);                    
                    imarray.headers_(n,s,loc).matrix_size[0]     = image_dims[0];
                    imarray.headers_(n,s,loc).matrix_size[1]     = image_dims[1];
                    imarray.headers_(n,s,loc).matrix_size[2]     = image_dims[2];
                    imarray.headers_(n,s,loc).field_of_view[0]   = dbuff.sampling_.recon_FOV_[0];
                    imarray.headers_(n,s,loc).field_of_view[1]   = dbuff.sampling_.recon_FOV_[1];
                    imarray.headers_(n,s,loc).field_of_view[2]   = dbuff.sampling_.recon_FOV_[2];
                    imarray.headers_(n,s,loc).channels           = 1;                    
                    imarray.headers_(n,s,loc).average = acqhdr.idx.average;
                    imarray.headers_(n,s,loc).slice = acqhdr.idx.slice;
                    imarray.headers_(n,s,loc).contrast = acqhdr.idx.contrast;
                    imarray.headers_(n,s,loc).phase = acqhdr.idx.phase;
                    imarray.headers_(n,s,loc).repetition = acqhdr.idx.repetition;
                    imarray.headers_(n,s,loc).set = acqhdr.idx.set;
                    imarray.headers_(n,s,loc).acquisition_time_stamp = acqhdr.acquisition_time_stamp;
                    imarray.headers_(n,s,loc).position[0] = acqhdr.position[0];
                    imarray.headers_(n,s,loc).position[1] = acqhdr.position[1];
                    imarray.headers_(n,s,loc).position[2] = acqhdr.position[2];
                    imarray.headers_(n,s,loc).read_dir[0] = acqhdr.read_dir[0];
                    imarray.headers_(n,s,loc).read_dir[1] = acqhdr.read_dir[1];
                    imarray.headers_(n,s,loc).read_dir[2] = acqhdr.read_dir[2];
                    imarray.headers_(n,s,loc).phase_dir[0] = acqhdr.phase_dir[0];
                    imarray.headers_(n,s,loc).phase_dir[1] = acqhdr.phase_dir[1];
                    imarray.headers_(n,s,loc).phase_dir[2] = acqhdr.phase_dir[2];
                    imarray.headers_(n,s,loc).slice_dir[0] = acqhdr.slice_dir[0];
                    imarray.headers_(n,s,loc).slice_dir[1] = acqhdr.slice_dir[1];
                    imarray.headers_(n,s,loc).slice_dir[2] = acqhdr.slice_dir[2];
                    imarray.headers_(n,s,loc).patient_table_position[0] = acqhdr.patient_table_position[0];
                    imarray.headers_(n,s,loc).patient_table_position[1] = acqhdr.patient_table_position[1];
                    imarray.headers_(n,s,loc).patient_table_position[2] = acqhdr.patient_table_position[2];
                    imarray.headers_(n,s,loc).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                    imarray.headers_(n,s,loc).image_index = ++image_counter_;




		    // ============================================== //
		    // ============ Reading out the data ============ //
		    // ============================================== //

		    std::vector<size_t> raw_sample_dims(4);    
		    raw_sample_dims[0] = E0;    
		    raw_sample_dims[1] = E1;   
		    raw_sample_dims[2] = E2;     
		    raw_sample_dims[3] = CHA;
		    hoNDArray<std::complex<float> > raw_samples_tmp = hoNDArray<std::complex<float> >(raw_sample_dims,  &dbuff.data_(0,0,0,0,n,s,loc));

		    hoNDArray<complext<float> > raw_samples(raw_samples_tmp.get_dimensions(), (complext<float>*) raw_samples_tmp.begin());

		    std::vector<size_t> sample_dims(3);
		    sample_dims[0] = num_samples_per_spoke;
		    sample_dims[1] = num_spokes;
		    sample_dims[2] = CHA;

		    hoNDArray< complext<float> > samples(&sample_dims);
		    
		    for (int i = 0; i < samples.get_number_of_elements(); i++)
			samples[i] = raw_samples[i];

		    if (do_save){
			hoNDArray<std::complex<float> > go(samples.get_dimensions(), (std::complex<float>*) samples.begin());
			std::string save_name = save_path + "samples.cplx";
			write_nd_array<std::complex<float> >(&go, save_name.c_str());
		    }

		 



		    // ============================================== //
		    // ================= NFFT recon ================= //
		    // ============================================== //
		   
		    int NFFT_comp_mode = 8; //NFFT_BACKWARDS_NC2C
		    plan.compute( &samples, &image, &DCF, NFFT_comp_mode );
		    
		    if (do_save){
			hoNDArray<std::complex<float> > go(image.get_dimensions(), (std::complex<float>*) image.begin());
			std::string save_name = save_path + "coil_images.cplx";
			write_nd_array<std::complex<float> >(&go, save_name.c_str());
		    }


		    
		    
                    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(output_dims, &imarray.data_(0,0,0,0,n,s,loc));
                    clear(output);

		    hoNDArray<std::complex<float> > image_out(image.get_dimensions(), (std::complex<float>*)image.begin());

		    multiplyConj( image_out, image_out, image_out );  

		    for (unsigned int c = 0; c < CHA; c++)
			output += hoNDArray<std::complex<float> >(&output_dims, &image_out(0,0,0,c));

                    sqrt_inplace(&output);                    
		}
            }
        }

        //Pass the image array down the chain
        if (this->next()->putq(cm1) < 0) {
	  m1->release();
          return GADGET_FAIL;
        }
    }

    m1->release();
    nfft_clock.set_timing_in_destruction(true);
    return GADGET_OK;  

}

GADGET_FACTORY_DECLARE(cpuNFFTGadget3d_bin)
}
