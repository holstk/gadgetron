  
#include "cpuNFFTGadget2d.h"
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


namespace Gadgetron{

cpuNFFTGadget2d::cpuNFFTGadget2d()
  : image_counter_(0)
{
}


int cpuNFFTGadget2d::process( GadgetContainerMessage<IsmrmrdReconData>* m1)
{
    std::cout << "\nIN CPUNFFT2D!\n" << std::endl;
    GadgetronTimer nfft_clock;// = clock();
    std::cout << "Clock start!" << std::endl;
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

	//std::cout << "E0: " << E0 << ", E1: " << E1 << ", E2: " << E2 << ", CHA: " << CHA << ", N: " << N << ", S: " << S << ", LOC: " << LOC << std::endl;
	std::string save_path = "/home/kh/data/go/";
	bool do_save = true;

	Nspokes = E1*E2;
	std::string spoke_idx_filename = "";// "/home/holstk/data/spoke_idx_files/spoke_idx.idx";
	bool is_spoke_idx = false;
	long int *spoke_idx = 0x0;
	if (is_spoke_idx){
	    std::ifstream infile_spoke_idx(spoke_idx_filename.c_str(), std::ios::binary);
	    infile_spoke_idx.read((char*)&Nspokes,sizeof(long int));
	    std::cout << "num_spokes: " << Nspokes << std::endl;
	    spoke_idx = new long int[Nspokes];
	    infile_spoke_idx.read((char*)spoke_idx,Nspokes*sizeof(long int));
	    //std::cout << "spoke_idx: " << spoke_idx[0] << ", " << spoke_idx[1] << ", " << spoke_idx[2] << ", " << spoke_idx[3] << ", " << spoke_idx[4] << std::endl;
	    int spoke_counter = 0;
	}


      
        //Create an image array message
        GadgetContainerMessage<IsmrmrdImageArray>* cm1 = 
                new GadgetContainerMessage<IsmrmrdImageArray>();

        //Grab references to the image array data and headers
        IsmrmrdImageArray & imarray = *cm1->getObjectPtr();

        //The image array data will be [E0,E1,E2,1,N,S,LOC] big
        //Will collapse across coils at the end
        std::vector<size_t> data_dims(7);
        data_dims[0] = E0;
        data_dims[1] = E1;
        data_dims[2] = E2;
        data_dims[3] = 1;
        data_dims[4] = N;
        data_dims[5] = S;
        data_dims[6] = LOC;        

	grid_oversampling_factor = 1.5; //alpha
	ro_oversampling_factor = 2.0;
	im_size = E0/(int)ro_oversampling_factor;
	grid_size = std::ceil(im_size * grid_oversampling_factor);

	grid_dims[0] = grid_size;
	grid_dims[1] = grid_size;
	im_dims[0] = im_size;
	im_dims[1] = im_size;
	if (E2 == 1) { // 2D
	    Ndims = 2;
	    grid_dims[2] = 1;
	    im_dims[2] = 1;
	}
	else {// 3D
	    Ndims = 3;
	    grid_dims[2] = grid_size;
	    im_dims[2] = im_size;
	}

        constexpr unsigned int D = 2;
	int correction_type = 0;

	imarray.data_.create(im_dims[0], im_dims[1], im_dims[2], data_dims[3], data_dims[4], data_dims[5], data_dims[6]);
        
        //ImageHeaders will be [N, S, LOC]
        std::vector<size_t> header_dims(3);
        header_dims[0] = N;
        header_dims[1] = S;
        header_dims[2] = LOC;        
        imarray.headers_.create(&header_dims);

        //We will not add any meta data
        //so skip the meta_ part

	cut_off = 0;
	NsamplesPerSpoke = E0;
	//Nspokes = E1 * E2;

	//Calculate the trajectory and DCF for recon

	std::cout << "Before calculating trajectory\n";

	GATrajectory<float,D> traj;
	hoNDArray<typename reald<float,D>::Type> trajectory;
	try{
	  if (is_spoke_idx){
	      trajectory = traj.calculateTrajectory(Nspokes, NsamplesPerSpoke, Ndims, spoke_idx, correction_type);
	  }
	  else{
	      trajectory = traj.calculateTrajectory(Nspokes, NsamplesPerSpoke, Ndims, 0, correction_type);
	  }
	}
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}

	std::cout << "After calculating trajectory\n";
	    
	try{
	    DCF = traj.calculateDCF(Nspokes, NsamplesPerSpoke, Ndims, cut_off, im_size);
	}
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}

	typename uint64d<D>::Type matrix_size(im_size);
	typename uint64d<D>::Type matrix_size_os(grid_size);
	float kernel_width = 4.0;//5.5;
	float alpha = (float)matrix_size_os[0]/(float)matrix_size[0];

	std::vector<size_t> image_dims = to_std_vector(matrix_size);
	image_dims.push_back(CHA);
	hoNDArray< complext<float> > image(&image_dims);

       
	hoNFFT_plan<float,D> plan( matrix_size, matrix_size_os, kernel_width );
	std::vector<long unsigned int> traj_dims_tmp = *trajectory.get_dimensions();
        int NFFT_prep_mode = 2; //1=NFFT_PREP_C2NC, 2=NFFT_PREP_NC2C, 4=NFFT_PREP_ALL
	
	plan.preprocess( &trajectory, NFFT_prep_mode );
        
        //Loop over S and N and LOC
        for (uint16_t loc=0; loc < LOC; loc++) {
            for (uint16_t s=0; s < S; s++) {                
                for (uint16_t n=0; n < N; n++) {
                    

		    //GDEBUG("Before all header definitions.\n");
                    //Set some information into the image header
                    //Use the middle acquisition header for some info
                    //[E1, E2, N, S, LOC]
                    ISMRMRD::AcquisitionHeader & acqhdr = dbuff.headers_(dbuff.sampling_.sampling_limits_[1].center_,
                                                                         dbuff.sampling_.sampling_limits_[2].center_,
                                                                         n, s, loc);                    
                    imarray.headers_(n,s,loc).matrix_size[0]     = im_dims[0];
                    imarray.headers_(n,s,loc).matrix_size[1]     = im_dims[1];
                    imarray.headers_(n,s,loc).matrix_size[2]     = im_dims[2];
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

                    //Grab a wrapper around the relevant chunk of data [E0,E1,E2,CHA] for this loc, n, and s
                    //Each chunk will be [E0,E1,E2,CHA] big
                    std::vector<size_t> chunk_dims(4);
                    chunk_dims[0] = E0;
                    chunk_dims[1] = E1;
                    chunk_dims[2] = E2;
                    chunk_dims[3] = CHA;
                    hoNDArray<std::complex<float> > chunk = hoNDArray<std::complex<float> >(chunk_dims, &dbuff.data_(0,0,0,0,n,s,loc));

		    /* WRITE */ 
		    if (do_save){
			std::string filename = save_path + "chunk.cplx";
			write_nd_array<std::complex<float> >(&chunk, filename.c_str());
		    }

		    hoNDArray<complext<float> > samples_tmp(chunk.get_dimensions(), (complext<float>*) chunk.begin());
		    
		    std::vector<size_t> sample_dims(3);
                    sample_dims[0] = chunk_dims[0];
		    sample_dims[1] = chunk_dims[1] * chunk_dims[2];
		    sample_dims[2] = chunk_dims[3];
		    hoNDArray< complext<float> > samples(&sample_dims);
		    //samples = new hoNDArray< std::complex<float> > (&sample_dims);
		    //samples(chunk);
		    
		    for (int i = 0; i < samples.get_number_of_elements(); i++)
			samples[i] = samples_tmp[i];
		   
		    
		    int NFFT_comp_mode = 8; //NFFT_BACKWARDS_NC2C
		    plan.compute( &samples, &image, &DCF, NFFT_comp_mode );
		   

		    GDEBUG("im_dims[0]: %d, im_dims[1]: %d, im_dims[2]: %d\n", im_dims[0], im_dims[1], im_dims[2]);
                    //Square root of the sum of squares
                    //Each image will be [E0,E1,E2,1] big
                    std::vector<size_t> img_dims(2);
                    img_dims[0] = im_dims[0];
                    img_dims[1] = im_dims[1];
                    img_dims[2] = im_dims[2];

		    //std::cout << "img_dims: " << img_dims[0] << ", " << img_dims[1] << ", " << img_dims[2] << std::endl;

		    GDEBUG("Root-SS...\n");

                    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(img_dims, &imarray.data_(0,0,0,0,n,s,loc));
		    GDEBUG("Output created\n");
                    //Zero out the output
                    clear(output);
		    GDEBUG("Output cleared\n");

                    //Compute d* d in place
                    //multiplyConj(images_cut, images_cut, images_cut);  
		    hoNDArray<std::complex<float> > image_out(image.get_dimensions(), (std::complex<float>*)image.begin());

		    /* WRITE */ 
		    if (do_save){
			std::string filename = save_path + "image_out1.cplx";
			write_nd_array<std::complex<float> >(&image_out, filename.c_str());
		    }

                    multiplyConj( image_out, image_out, image_out );    

		    /* WRITE */ 
		    if (do_save){
			std::string filename = save_path + "image_out2.cplx";
			write_nd_array<std::complex<float> >(&image_out, filename.c_str());
		    }

		    GDEBUG("Im squared\n");
                    //Add up
                    for (size_t c = 0; c < CHA; c++) {
			output += hoNDArray<std::complex<float> >(img_dims, &image_out(0,0,c));
                    }     

		    /* WRITE */ 
		    if (do_save){
			std::string filename = save_path + "output1.cplx";
			write_nd_array<std::complex<float> >(&output, filename.c_str());
		    }

		    GDEBUG("Summed\n");
                    //Take the square root in place
                    sqrt_inplace(&output);                    
		    GDEBUG("Squared\n");
		    
		    /* WRITE */ 
		    if (do_save){
			std::string filename = save_path + "output2.cplx";
			write_nd_array<std::complex<float> >(&output, filename.c_str());
		    }

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
    //clock_t end = clock();
    //GadgetronTimer end;
    //double elapsed_time = double(end-start) / CLOCKS_PER_SEC;
    //std::cout << "Elapsed time for NFFT gadget: " << std::floor(elapsed_time/60) << ":" << int(elapsed_time)%60 << std::endl;
    //std::cout << "Elapsed time for NFFT gadget: " << elapsed_time << std::endl;
    nfft_clock.set_timing_in_destruction(true);
    return GADGET_OK;  

}

GADGET_FACTORY_DECLARE(cpuNFFTGadget2d)
}