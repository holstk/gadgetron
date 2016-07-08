  
#include "cpuCgSenseGadget2d.h"
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

cpuCgSenseGadget2d::cpuCgSenseGadget2d()
  : image_counter_(0)
{
}


int cpuCgSenseGadget2d::process( GadgetContainerMessage<IsmrmrdReconData>* m1)
{
    GadgetronTimer nfft_clock;

    //Iterate over all the recon bits
    for(std::vector<IsmrmrdReconBit>::iterator it = m1->getObjectPtr()->rbit_.begin();
        it != m1->getObjectPtr()->rbit_.end(); ++it)
    {

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


        //constexpr unsigned int D = 2;
	grid_oversampling_factor = 2.0; //alpha
	ro_oversampling_factor = 2.0;
	kernel_width = 5.5;  //original: 5.5;
	cut_off = 0;


        num_samples_per_spoke = E0;
	num_spokes = E1 * E2;


	im_size = num_samples_per_spoke/(int)ro_oversampling_factor;
	im_size_os = std::ceil(im_size * grid_oversampling_factor);

	matrix_size = uint64d2(im_size);
	matrix_size_os =  uint64d2(im_size_os);
	alpha = (float)matrix_size_os[0] / (float)matrix_size[0];



	//Calculate the trajectory and DCF for recon	
	GATrajectory<float,2> traj;

	hoNDArray<typename reald<float,2>::Type> trajectory;
	try{  trajectory = traj.calculateTrajectory(num_spokes, num_samples_per_spoke, 2);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}
	    
	try{  DCF = traj.calculateDCF(num_spokes, num_samples_per_spoke, 2, cut_off, im_size);  }
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}


	//Setting up the plan
	hoNFFT_plan<float,2> plan( matrix_size, matrix_size_os, kernel_width );
        int NFFT_prep_mode = 2; //1=NFFT_PREP_C2NC, 2=NFFT_PREP_NC2C, 4=NFFT_PREP_ALL
	plan.preprocess( &trajectory, NFFT_prep_mode );



	//Image matrix
	imarray.data_.create(matrix_size[0], matrix_size[1], 1, data_dims[3], data_dims[4], data_dims[5], data_dims[6]);
        
	std::vector<size_t> image_dims = to_std_vector(matrix_size);
	image_dims.push_back(CHA);
	hoNDArray< complext<float> > image(&image_dims);



	//Dimensions
	std::vector<size_t> chunk_dims(4);    chunk_dims[0] = num_samples_per_spoke;    chunk_dims[1] = num_spokes;   chunk_dims[2] = E2;     chunk_dims[3] = CHA;
	std::vector<size_t> sample_dims(3);   sample_dims[0] = num_samples_per_spoke;   sample_dims[1] = num_spokes;       sample_dims[2] = CHA;



        //ImageHeaders will be [N, S, LOC]
        std::vector<size_t> header_dims(3);
        header_dims[0] = N;
        header_dims[1] = S;
        header_dims[2] = LOC;        
        imarray.headers_.create(&header_dims);




	//Calculate regularization image
	size_t reg_spokes = 100;
	std::vector<size_t> reg_chunk_dims(4);    reg_chunk_dims[0] = num_samples_per_spoke;    reg_chunk_dims[1] = reg_spokes;   reg_chunk_dims[2] = E2;    reg_chunk_dims[3] = CHA;
	std::vector<size_t> reg_sample_dims(3);   reg_sample_dims[0] = num_samples_per_spoke;   reg_sample_dims[1] = reg_spokes;                             reg_sample_dims[2] = CHA;

	hoNDArray<std::complex<float> > reg_chunk_tmp = hoNDArray<std::complex<float> >(chunk_dims,  &dbuff.data_(0,0,0,0,0,0,0));
	hoNDArray<std::complex<float> > reg_chunk(&reg_chunk_dims);

	for (unsigned int spoke = 0; spoke < reg_spokes; spoke++){
	    for (unsigned int sample = 0; sample < num_samples_per_spoke; sample++){
		for (unsigned int c = 0; c < CHA; c++){
		    reg_chunk[c*reg_spokes*num_samples_per_spoke + spoke*num_samples_per_spoke + sample] 
			= reg_chunk_tmp[c*num_spokes*num_samples_per_spoke + spoke*num_samples_per_spoke + sample];
		}
	    }
	}

	hoNDArray<complext<float> > reg_samples_tmp(reg_chunk.get_dimensions(), (complext<float>*) reg_chunk.begin());
	hoNDArray<complext<float> > reg_samples(&reg_sample_dims);
		    
	for (int i = 0; i < reg_samples.get_number_of_elements(); i++)
	    reg_samples[i] = reg_samples_tmp[i];

	GATrajectory<float,2> reg_traj;
	hoNDArray<typename reald<float,2>::Type> reg_trajectory;
	reg_trajectory = reg_traj.calculateTrajectory(reg_spokes, num_samples_per_spoke, 2); 	    
	hoNDArray<float> reg_DCF = reg_traj.calculateDCF(reg_spokes, num_samples_per_spoke, 2, cut_off, im_size);

	std::vector<size_t> reg_dims_tmp = to_std_vector(matrix_size);
	reg_dims_tmp.push_back(CHA);
	hoNDArray< complext<float> > reg_image_tmp(&reg_dims_tmp);

	hoNFFT_plan<float,2> reg_plan( matrix_size, matrix_size_os, kernel_width );
	reg_plan.preprocess( &reg_trajectory, 2 );
	reg_plan.compute( &reg_samples, &reg_image_tmp, &DCF, 8 );

	hoNDArray<complext<float> > im_tmp(&reg_image_tmp);
	hoNDArray<complext<float> > csm_tmp;
	estimate_b1_map<float,2>( &im_tmp, &csm_tmp, im_tmp.get_size(2));

	std::vector<size_t> reg_dims = to_std_vector(matrix_size);
	hoNDArray<complext<float> > reg_image = hoNDArray<complext<float> >(reg_dims);
	multiplyConj( reg_image_tmp, reg_image_tmp, reg_image_tmp );    
	
	for (size_t c = 0; c < CHA; c++) {
	    reg_image += hoNDArray<complext<float> >(reg_dims, &reg_image_tmp(0,0,c));
	}                    
	sqrt_inplace(&reg_image);

	write_nd_array<complext<float> >(&reg_image, "/home/holstk/data/2d_test/reg_im.cplx");	
	write_nd_array<complext<float> >(&csm_tmp, "/home/holstk/data/2d_test/csm_tmp.cplx");	
	
	
	int num_iterations = 10;
	float kappa = 0.3;
	int num_frames = 1;

	std::vector<size_t> recon_dims = to_std_vector(matrix_size);
	recon_dims.push_back(num_frames);

	boost::shared_ptr< hoNDArray<float> >dcw( new hoNDArray<float>(DCF) );
	boost::shared_ptr< hoNDArray<complext<float> > >csm( new hoNDArray<complext<float> >(csm_tmp) );
	boost::shared_ptr< hoNDArray<complext<float> > > reg_im( new hoNDArray<complext<float> >(reg_image));

	
	boost::shared_ptr< hoNonCartesianSenseOperator<float,2> > E( new hoNonCartesianSenseOperator<float,2>() );  
	E->setup( matrix_size, matrix_size_os, kernel_width );
	E->set_dcw(dcw) ;
	E->set_csm(csm);
	E->set_domain_dimensions(&recon_dims);
	E->set_codomain_dimensions(&sample_dims);
	E->preprocess(&trajectory);
	//std::cout << "After E->preprocess" << std::endl;
 

	boost::shared_ptr< hoImageOperator<complext<float> > > R( new hoImageOperator<complext<float> >() );
	//std::cout << "After R" << std::endl;
	R->set_weight( kappa );
	//std::cout << "After R->set_weight" << std::endl;
	R->compute( reg_im.get() );
	//std::cout << "After R->compute" << std::endl;


	boost::shared_ptr< hoNDArray<float> > _precon_weights = sum(abs_square(csm.get()).get(), 2);
	boost::shared_ptr< hoNDArray<float> > R_diag = R->get();
	//std::cout << "_precon_weights dims: " << _precon_weights->get_number_of_dimensions() << std::endl;
	//std::cout << "_precon_weights dims: " << _precon_weights->get_size(0) << std::endl;
	//std::cout << "_precon_weights dims: " << _precon_weights->get_size(1) << std::endl;
	//std::cout << "R_diag dims: " << R_diag->get_number_of_dimensions() << std::endl;
	//std::cout << "R_diag dims: " << R_diag->get_size(0) << std::endl;
	//std::cout << "R_diag dims: " << R_diag->get_size(1) << std::endl;
	//std::cout << "R_diag dims: " << R_diag->get_size(2) << std::endl;
	//std::cout << "After R->get" << std::endl;
	*R_diag *= kappa;
	//std::cout << "After R_diag *= kappa" << std::endl;
	*_precon_weights += *R_diag;
	//std::cout << "After precon += R_diag" << std::endl;
	R_diag.reset();
	reciprocal_sqrt_inplace(_precon_weights.get());
	boost::shared_ptr< hoNDArray<complext<float> > > precon_weights = real_to_complex<complext<float> >( _precon_weights.get() );
	_precon_weights.reset();


	// Define preconditioning matrix
	boost::shared_ptr< hoCgPreconditioner<complext<float> > > D( new hoCgPreconditioner<complext<float> >() );
	D->set_weights( precon_weights );
	precon_weights.reset();
	csm.reset();


	// Setup conjugate gradient solver
	hoCgSolver<complext<float> > cg;
	cg.set_preconditioner ( D );           // preconditioning matrix
	cg.set_max_iterations( num_iterations );
	cg.set_tc_tolerance( 1e-6 );
	cg.set_output_mode( hoCgSolver<complext<float> >::OUTPUT_VERBOSE );
	cg.set_encoding_operator( E );        // encoding matrix
	cg.add_regularization_operator( R );  // regularization matrix
	


        //Loop over S and N and LOC
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
                    imarray.headers_(n,s,loc).matrix_size[2]     = 1;
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


		    size_t recon_spokes = 100;
		    std::vector<size_t> recon_chunk_dims(4); recon_chunk_dims[0] = num_samples_per_spoke; recon_chunk_dims[1] = recon_spokes; recon_chunk_dims[2] = E2; recon_chunk_dims[3] = CHA;
		    std::vector<size_t> recon_sample_dims(3);   recon_sample_dims[0] = num_samples_per_spoke;   recon_sample_dims[1] = recon_spokes;        recon_sample_dims[2] = CHA;

		    hoNDArray<std::complex<float> > chunk = hoNDArray<std::complex<float> >(chunk_dims,  &dbuff.data_(0,0,0,0,n,s,loc));
		    hoNDArray<std::complex<float> > recon_chunk(&recon_chunk_dims);
		    
		    for (unsigned int spoke = 0; spoke < recon_spokes; spoke++){
			for (unsigned int sample = 0; sample < num_samples_per_spoke; sample++){
			    for (unsigned int c = 0; c < CHA; c++){
				recon_chunk[c*recon_spokes*num_samples_per_spoke + spoke*num_samples_per_spoke + sample] 
				    = chunk[c*num_spokes*num_samples_per_spoke + spoke*num_samples_per_spoke + sample];
			    }
			}
		    }
		    
		    hoNDArray<complext<float> > samples_tmp(recon_chunk.get_dimensions(), (complext<float>*) recon_chunk.begin());
		    hoNDArray<complext<float> > samples(&recon_sample_dims);






		    
		    //hoNDArray<std::complex<float> > chunk = hoNDArray<std::complex<float> >(chunk_dims,  &dbuff.data_(0,0,0,0,n,s,loc));
		    //hoNDArray<complext<float> > samples(&sample_dims);
		    //hoNDArray<complext<float> > samples_tmp(chunk.get_dimensions(), (complext<float>*) chunk.begin());
		    
		    for (int i = 0; i < samples.get_number_of_elements(); i++)
			samples[i] = samples_tmp[i];

		    boost::shared_ptr< hoNDArray<complext<float> > > cgresult;
		    
		    //std::cout << "Just before solve" << std::endl;
		    cgresult = cg.solve(&samples);
		    std::cout << "Just after solve" << std::endl;
		    


		    //int NFFT_comp_mode = 8; //NFFT_BACKWARDS_NC2C
		    //plan.compute( &samples, &image, &DCF, NFFT_comp_mode );

		    
		    std::vector<size_t> output_dims = to_std_vector(matrix_size);
                    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(output_dims, &imarray.data_(0,0,0,0,n,s,loc));
                    clear(output);

		    hoNDArray<std::complex<float> > image_out(cgresult.get()->get_dimensions(), (std::complex<float>*)cgresult.get()->begin());
		    output += hoNDArray<std::complex<float> >(output_dims, image_out.begin());
                    //multiplyConj( image_out, image_out, image_out );    

                    //for (size_t c = 0; c < CHA; c++) {
		    //	    output += hoNDArray<std::complex<float> >(output_dims, &image_out(0,0,c));
                    //}                    

                    //sqrt_inplace(&output);   
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

GADGET_FACTORY_DECLARE(cpuCgSenseGadget2d)
}