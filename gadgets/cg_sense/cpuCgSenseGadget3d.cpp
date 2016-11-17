  
#include "cpuCgSenseGadget3d.h"
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

#include "cpu_b1_map.h"

#include "hoNonCartesianSenseOperator.h"
#include "hoCgPreconditioner.h"
#include "hoImageOperator.h"
#include "hoCgSolver.h"

namespace Gadgetron{

cpuCgSenseGadget3d::cpuCgSenseGadget3d()
  : image_counter_(0)
{
}


int cpuCgSenseGadget3d::process( GadgetContainerMessage<IsmrmrdReconData>* m1)
{
    GadgetronTimer nfft_clock;
    std::string save_path_3d = "/mnt/scratch/karen/go/";
    bool do_save_3d = false;


    //Iterate over all the recon bits
    for(std::vector<IsmrmrdReconBit>::iterator it = m1->getObjectPtr()->rbit_.begin();
        it != m1->getObjectPtr()->rbit_.end(); ++it)
    {
	std::cout << "In 3d sense gadget" << std::endl;

	
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
	// ============= Trajectory and DCF ============= //
	// ============================================== //

	//Calculate the trajectory and DCF for recon	
	GATrajectory<float,dim> traj;

	hoNDArray<typename reald<float,dim>::Type> trajectory;
	try{  trajectory = traj.calculateTrajectory(num_spokes, num_samples_per_spoke, dim);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}
	    
	try{  DCF = traj.calculateDCF(num_spokes, num_samples_per_spoke, dim, cut_off, im_size);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
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



		    // ============================================== //
		    // ============ Coil sensitivity map ============ //
		    // ============================================== //
		    
		    std::cout << "Calculating coil sensitivity map:\n";
		    GadgetronTimer coil_sensitivity_time;

		    std::cout << "   NFFT for coil images\n";
		    hoNDArray< complext<float> > coil_images(&image_dims);
		    plan.compute( &samples, &coil_images, &DCF, 8 );

		    std::cout << "   Calculating CSM\n";
		    hoNDArray< complext<float> > csm(&image_dims);
		    estimate_b1_map<float,dim>( &coil_images, &csm, CHA);

		    

		    /* WRITE */ 
		    if (do_save_3d){
			hoNDArray<std::complex<float> > go(csm.get_dimensions(), (std::complex<float>*) csm.get_data_ptr());
			std::string filename = save_path_3d + "csm.cplx";
			write_nd_array<std::complex<float> >(&go, filename.c_str());
		    }
		    if (do_save_3d){
			hoNDArray<std::complex<float> > go(coil_images.get_dimensions(), (std::complex<float>*) coil_images.get_data_ptr());
			std::string filename = save_path_3d + "coil_images.cplx";
			write_nd_array<std::complex<float> >(&go, filename.c_str());
		    }

		    

		    // ============================================== //
		    // ================= Reg image ================== //
		    // ============================================== //
		    
		    hoNDArray< complext<float> > reg_image(&output_dims);   clear(reg_image);
		    multiplyConj( coil_images, coil_images, coil_images );    
		    for (unsigned int ii = 0; ii < prod(matrix_size); ii++){
			for (unsigned int c = 0; c < CHA; c++) {
			    reg_image[ii] += coil_images[c*prod(matrix_size)+ii];
			}
		    }                    
		    sqrt_inplace(&reg_image);
		   
		    


		    // ============================================== //
		    // =============== Sense operator =============== //
		    // ============================================== //

		    int num_iterations = 8;
		    float kappa = 0.3;

		    boost::shared_ptr< hoNDArray<float> >dcw( new hoNDArray<float>(DCF) );
		    boost::shared_ptr< hoNDArray<complext<float> > >csm_( new hoNDArray<complext<float> >(csm) );
		    boost::shared_ptr< hoNDArray<complext<float> > > reg_image_( new hoNDArray<complext<float> >(reg_image));

	
		    boost::shared_ptr< hoNonCartesianSenseOperator<float,dim> > E( new hoNonCartesianSenseOperator<float,dim>() );  
		    E->setup( matrix_size, matrix_size_os, kernel_width );
		    E->set_dcw(dcw) ;
		    E->set_csm(csm_);
		    E->set_domain_dimensions(&output_dims);
		    E->set_codomain_dimensions(&sample_dims);
		    E->preprocess(&trajectory);
		    std::cout << "After E->preprocess" << std::endl;



		    // ============================================== //
		    // =============== Image operator =============== //
		    // ============================================== //

		    boost::shared_ptr< hoImageOperator<complext<float> > > R( new hoImageOperator<complext<float> >() );
		    R->set_weight( kappa );
		    R->compute( reg_image_.get() );

		    //Calculating precon weights
		    hoNDArray<complext<float> > csm_copy(csm.get_dimensions(), csm.begin());
		    multiplyConj(csm_copy,csm_copy,csm_copy);
		    hoNDArray<float> pw(&output_dims);   clear(pw);		    
                    for (unsigned int ii = 0; ii < prod(matrix_size); ii++) {
			for (unsigned int c = 0; c < CHA; c++){
			    pw[ii] += csm_copy[c*prod(matrix_size)+ii].real();
			}
                    }  
		    
     		    boost::shared_ptr< hoNDArray<float> > _precon_weights( new hoNDArray<float>(pw) );
		    boost::shared_ptr< hoNDArray<float> > R_diag = R->get();

		    *R_diag *= kappa;
		    *_precon_weights += *R_diag;

		    R_diag.reset();
		    reciprocal_sqrt_inplace(_precon_weights.get());
		    boost::shared_ptr< hoNDArray<complext<float> > > precon_weights = real_to_complex<complext<float> >( _precon_weights.get() );
		    _precon_weights.reset();


		    /* WRITE */ 
		    if (do_save_3d){
			hoNDArray<std::complex<float> > go(precon_weights->get_dimensions(), (std::complex<float>*)precon_weights.get()->begin());
			std::string filename = save_path_3d + "precon_weights.cplx";
			write_nd_array<std::complex<float> >(&go, filename.c_str());
		    }
		    if (do_save_3d){
			hoNDArray<std::complex<float> > go(reg_image_->get_dimensions(), (std::complex<float>*)reg_image_.get()->begin());
			std::string filename = save_path_3d + "reg_image_.cplx";
			write_nd_array<std::complex<float> >(&go, filename.c_str());
		    }




		    // ============================================== //
		    // =============== Preconditioning ============== //
		    // ============================================== //

		    // Define preconditioning matrix
		    boost::shared_ptr< hoCgPreconditioner<complext<float> > > D( new hoCgPreconditioner<complext<float> >() );
		    D->set_weights( precon_weights );
		    precon_weights.reset();
		    csm_.reset();
		    



		    // ============================================== //
		    // ================== CG solver ================= //
		    // ============================================== //

		    // Setup conjugate gradient solver
		    hoCgSolver<complext<float>, float > cg;
		    cg.set_preconditioner ( D );           // preconditioning matrix
		    cg.set_max_iterations( num_iterations );
		    cg.set_tc_tolerance( 1e-6 );
		    cg.set_output_mode( hoCgSolver<complext<float>, float >::OUTPUT_VERBOSE );
		    cg.set_encoding_operator( E );        // encoding matrix
		    cg.add_regularization_operator( R );  // regularization matrix
	





		    // ============================================== //
		    // ============= Reconstruct images ============= //
		    // ============================================== //

		    boost::shared_ptr< hoNDArray<complext<float> > > cgresult;
		    
		    //std::cout << "Just before solve" << std::endl;
		    cgresult = cg.solve(&samples);
		    std::cout << "Just after solve" << std::endl;
		    
		    
                    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(output_dims, &imarray.data_(0,0,0,0,n,s,loc));
                    clear(output);

		    hoNDArray<std::complex<float> > image_out(cgresult.get()->get_dimensions(), (std::complex<float>*)cgresult.get()->begin());
		    output += hoNDArray<std::complex<float> >(output_dims, image_out.begin());
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

GADGET_FACTORY_DECLARE(cpuCgSenseGadget3d)
}
