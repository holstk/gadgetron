  
#include "cpuNFFTGadget3d_lin.h"
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


/**
constexpr unsigned int foo(unsigned int i)
{
    //int res = i;
    //return const_cast<int&>(i);
    //return (i==2 ? 2 : 3);
    constexpr unsigned int res =  this->value;
    return res;
}
*/

namespace Gadgetron{

cpuNFFTGadget3d_linear::cpuNFFTGadget3d_linear()
  : image_counter_(0)
{
}


int cpuNFFTGadget3d_linear::process( GadgetContainerMessage<IsmrmrdReconData>* m1)
{
    std::cout << "\nIN CPUNFFT3D!\n" << std::endl;
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
	std::cout << "E0: " << E0 << ", E1: " << E1 << ", E2: " << E2 << ", CHA: " << CHA << ", N: " << N << ", S: " << S << ", LOC: " << LOC << std::endl;

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

        constexpr unsigned int D = 3;
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
	
	LinearRadialTrajectory<float,D> traj;
	hoNDArray<typename reald<float,D>::Type> trajectory;
	try{
	    //if (is_spoke_idx){
	    //trajectory = traj.calculateTrajectory(Nspokes, NsamplesPerSpoke, Ndims, spoke_idx);
	    //}
	    //else{
	    trajectory = traj.calculateTrajectory(Nspokes, NsamplesPerSpoke, Ndims, 0, correction_type);
	    //}
	}
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}
	    
	try{
	    DCF = traj.calculateDCF(Nspokes, NsamplesPerSpoke, Ndims, cut_off, im_size);
	}
	catch(std::invalid_argument& e){
	    GERROR("Dimension error.\n");
	    return GADGET_FAIL;
	}

	hoNDArray<float> trajectory_out(trajectory.get_number_of_elements()*3, (float*)trajectory.begin());
	//hoNDArray<float> dcw_out(DCF.get_dimensions(), (float*)DCF.begin());
	write_nd_array<float >(&trajectory_out, "/home/holstk/data/gadget_output/trajectory.real");
	//write_nd_array<float>(&dcw_out, "/home/holstk/data/gadget_output/dcw.real");

	//std::cout << "Trajectory: \ndims: " << trajectory.get_number_of_dimensions() << "\nelements: " << trajectory.get_number_of_elements() << std::endl;
	//std::cout << "Trajectory out: \ndims: " << trajectory_out.get_number_of_dimensions() << "\nelements: " << trajectory_out.get_number_of_elements() << std::endl;

	typename uint64d<D>::Type matrix_size(im_size);
	typename uint64d<D>::Type matrix_size_os(grid_size);
	float kernel_width = 4.0;//5.5;
	float alpha = (float)matrix_size_os[0]/(float)matrix_size[0];

	std::vector<size_t> image_dims = to_std_vector(matrix_size);
	image_dims.push_back(CHA);
	hoNDArray< complext<float> > image(&image_dims);

        //Gridder<float,2>  myGrid(trajectory, im_dims, grid_oversampling_factor);

	//GDEBUG("Before plan create\n");
	hoNFFT_plan<float,D> plan( matrix_size, matrix_size_os, kernel_width );
	//std::cout << "Initializing plan" << std::endl;
	//hoNFFT_plan<float,2> plan_nfft( );
	//GDEBUG("Before preprocess\n");
	std::vector<long unsigned int> traj_dims_tmp = *trajectory.get_dimensions();
	//std::cout << traj_dims_tmp[0] << std::endl;
        int NFFT_prep_mode = 2; //1=NFFT_PREP_C2NC, 2=NFFT_PREP_NC2C, 4=NFFT_PREP_ALL
	//std::cout << "\n(NFFT_prep_mode_tmp == ""NFFT_prep_mode_tmp""): " << (NFFT_prep_mode_tmp == "NFFT_prep_mode_tmp") << "\n" << std::endl;
	plan.preprocess( &trajectory, NFFT_prep_mode );
	//std::cout << "Plan preprocessed" << std::endl;
	//plan.preprocess( &trajectory, hoNFFT_plan<float,D>::NFFT_PREP_NC2C );

	/**
       	GDEBUG("Before creating kspace.\n");
	if (Ndims==2)
	{
	    kspace.create(grid_size, grid_size, 1, CHA);
	}
	else
	{
	  kspace.create(grid_size, grid_size, grid_size, CHA);
	}
	*/
        
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
		    
		    //for (int c = 0; c < CHA; c++){
		    //for (int i = 0; i < E2; i++){
		    //samples[NsamplesPerSpoke*E1];
		    //}
		    //}
		    
		    //GDEBUG("Before compute\n");
		    //std::cout << "Gadget: sample dims: " << sample_dims[0] << " " << sample_dims[1] << " " << sample_dims[2] << std::endl;
		    //std::cout << "Gadget: chunk num elements: " << chunk.get_number_of_elements() << std::endl;
		    //std::cout << "Gadget: samples num elements: " << samples.get_number_of_elements() << std::endl;
		    //plan.compute( &samples, &image, &DCF, hoNFFT_plan<float,D>::NFFT_BACKWARDS_NC2C );
		    int NFFT_comp_mode = 8; //NFFT_BACKWARDS_NC2C
		    plan.compute( &samples, &image, &DCF, NFFT_comp_mode );
		    //GDEBUG("After compute");

		    //* WRITE */ write_nd_array<std::complex<float>>(&image, "/home/karen/data/images/image_from_plan.cplx");

		    /**
		    for (int c = 0 ; c < CHA ; c++)
		    {
			GDEBUG("Gridding channel %d of %d\n", c+1, CHA);
			samples_single_coil.create(NsamplesPerSpoke, Nspokes);

			for (int i = 0 ; i < samples_single_coil.get_number_of_elements() ; i++)
			{
			    samples_single_coil[i] = chunk[NsamplesPerSpoke*Nspokes*c + i];
			}
			hoNDArray< std::complex<float> > grid_out;
			GDEBUG("Just before convolution\n");
			if (Ndims==2)
			{
			    grid_out = myGrid.convolution_2d(samples_single_coil, &DCF, 1);
			    GDEBUG("Just after convolution\n");
			}
			else
			{
			    grid_out = myGrid.convolution_3d(samples_single_coil, &DCF, 1);
			    if (c == CHA-1)
			    {
				GDEBUG("grid_out dim 0: %d, grid_out dim 1: %d, grid_out dim 2: %d\n", grid_out.get_size(0), grid_out.get_size(1), grid_out.get_size(2));
			    }
			}
			
			if (c == CHA-1)
			{
			    GDEBUG("Gridding of last channel OK.\n");
			    //* WRITE * / write_nd_array<std::complex<float>>(&grid_out, "/home/kh/data/images/gadget_output_gridout.cplx");
			}
			
			for (int j = 0 ; j < grid_out.get_number_of_elements() ; j++)
			{
			    if (Ndims==2)
			    {
				kspace[grid_size*grid_size*c + j] = grid_out[j];
			    }
			    else
			    {		
				int64_t count_var = grid_size * grid_size * grid_size * (int64_t)c  +  (int64_t)j;
				kspace[count_var] = grid_out[j];
			    }
			}
		    }

		    GDEBUG("kspace Ndims: %d, dims: %d, %d, %d, %d\n", kspace.get_number_of_elements(), kspace.get_size(0), kspace.get_size(1), kspace.get_size(2), kspace.get_size(3));

		    //* WRITE * / write_nd_array<std::complex<float>>(&kspace, "/home/kh/data/images/gadget_output_kspace.cplx");

		    GDEBUG("IFFT...\n");

		    //Do the FFTs in place
                    hoNDFFT<float>::instance()->ifft(&kspace,0); GDEBUG("Finshed ifft in dim 0\n");
                    hoNDFFT<float>::instance()->ifft(&kspace,1); GDEBUG("Finshed ifft in dim 1\n");
                    if (E2>1) {
                        hoNDFFT<float>::instance()->ifft(&kspace,2); GDEBUG("Finshed ifft in dim 2\n");
		    }

		    GDEBUG("IFFT done\n!");

		    hoNDArray<std::complex<float> > images(kspace);

		    GDEBUG("images matrix allocated");

		    //* WRITE * / write_nd_array<std::complex<float>>(&images, "/home/kh/data/images/gadget_output_ifft.cplx");

		    GDEBUG("Image cutting and deapodization...\n");

		    hoNDArray< std::complex<float> > a = myGrid.calculate_deapodization_filter();
		    //* WRITE * / write_nd_array< std::complex<float> >(&a, "/home/kh/data/images/gadget_output_a.cplx");

		    //Cutting over sampling and applying deapodization
		    int64_t edge = std::floor((im_size*grid_oversampling_factor - im_size)/2);  GDEBUG("Edge: %d\n", edge);
		    hoNDArray< std::complex<float> > images_cut;
		    if (Ndims==2)
		    {
			images_cut.create(im_size, im_size, 1, CHA);
			for (int c = 0 ; c < CHA ; c++)
			{
			    GDEBUG("Image cutting and deapodization of channel %d of %d\n", c+1, CHA);
			    for (int x = edge ; x < edge+im_size ; x++)
			    {
				for (int y = edge ; y < edge+im_size ; y++)
				{
				    images_cut[(x-edge) + (y-edge)*im_size + c*im_size*im_size] = images[x + y*grid_size + c*grid_size*grid_size] / a[x + y*grid_size];
				}
			    }
			}
		    }
		    else
		    {
			GDEBUG("Before creating images_cut, im_size: %d, CHA: %d\n", im_size, CHA);
			images_cut.create(im_size, im_size, im_size, CHA);
			GDEBUG("After creating images_cut\n");
			for (int64_t c = 0 ; c < CHA ; c++)
			{
			    GDEBUG("Image cutting and deapodization of channel %d of %d\n", c+1, CHA);
			    for (int64_t x = edge ; x < edge+im_size ; x++)
			    {
				for (int64_t y = edge ; y < edge+im_size ; y++)
				{
				    for (int64_t z = edge ; z < edge+im_size ; z++)
				      {
					images_cut[(x-edge) + (y-edge)*im_size + (z-edge)*im_size*im_size + c*im_size*im_size*im_size] = images[x + y*grid_size + z*grid_size*grid_size + c*grid_size*grid_size*grid_size] / a[x + y*grid_size + z*grid_size*grid_size];
				    }
				}
			    }
			}
		    }
		    */
		    //* WRITE */ write_nd_array<std::complex<float>>(&images_cut, "/home/kh/data/images/gadget_output_im_cut.cplx");
		    

		    GDEBUG("im_dims[0]: %d, im_dims[1]: %d, im_dims[2]: %d\n", im_dims[0], im_dims[1], im_dims[2]);
                    //Square root of the sum of squares
                    //Each image will be [E0,E1,E2,1] big
                    std::vector<size_t> img_dims(3);
                    img_dims[0] = im_dims[0];
                    img_dims[1] = im_dims[1];
                    img_dims[2] = im_dims[2];

		    GDEBUG("Root-SS...\n");

                    hoNDArray<std::complex<float> > output = hoNDArray<std::complex<float> >(img_dims, &imarray.data_(0,0,0,0,n,s,loc));
		    GDEBUG("Output created\n");
                    //Zero out the output
                    clear(output);
		    GDEBUG("Output cleared\n");

                    //Compute d* d in place
                    //multiplyConj(images_cut, images_cut, images_cut);  
		    hoNDArray<std::complex<float> > image_out(image.get_dimensions(), (std::complex<float>*)image.begin());
                    multiplyConj( image_out, image_out, image_out );    
		    //* WRITE */ write_nd_array<std::complex<float>>(&image, "/home/karen/data/images/image_multConj.cplx");

		    GDEBUG("Im squared\n");
                    //Add up
                    for (size_t c = 0; c < CHA; c++) {
			if (Ndims == 2)
			    output += hoNDArray<std::complex<float> >(img_dims, &image_out(0,0,c));
			else
			    output += hoNDArray<std::complex<float> >(img_dims, &image_out(0,0,0,c));

			if (c==CHA-1)
			    //* WRITE */ write_nd_array<std::complex<float>>(&output, "/home/karen/data/images/output_last_cha.cplx");
			if (c==CHA-1){
			    hoNDArray<std::complex<float> > im_tmp_first_cha = hoNDArray<std::complex<float> >(img_dims, &image_out(0,0,0,c));
			    //* WRITE */ write_nd_array<std::complex<float>>(&im_tmp_first_cha, "/home/karen/data/images/image_last_cha.cplx");
			}
                    }                    
		    //* WRITE */ write_nd_array<std::complex<float>>(&output, "/home/karen/data/images/image_sum.cplx");
		    GDEBUG("Summed\n");
                    //Take the square root in place
                    sqrt_inplace(&output);                    
		    GDEBUG("Squared\n");
		    
		    //* WRITE */ write_nd_array<std::complex<float>>(&output, "/home/karen/data/images/gadget_output.cplx");
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

GADGET_FACTORY_DECLARE(cpuNFFTGadget3d_linear)
}
