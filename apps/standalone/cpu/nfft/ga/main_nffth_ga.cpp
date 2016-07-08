#include "hoNDArray_fileio.h"
#include "parameterparser.h"
#include "ga_trajectory.cpp"
#include "hoNDFFT.h"
#include "hoNDArray.h"
#include "gridder.h"
#include "gridder.cpp"
#include "radial_utilities.h"
#include "vector_td_utilities.h"
#include <complex>
#include <ctime>


#include <iostream>

using namespace Gadgetron;



int main( int argc, char** argv) 
{
  clock_t all_beg = clock(); 
  
  // Parse command line
  ParameterParser parms;
  parms.add_parameter( 'd', COMMAND_LINE_STRING, 1, "Input samples file name (.cplx)", true );
  parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Output image file name (.cplx)", true, "result.cplx" );
  parms.add_parameter( 'm', COMMAND_LINE_INT,    1, "Matrix size", true );
  parms.add_parameter( 'o', COMMAND_LINE_INT,    1, "Oversampled matrix size", true );
  parms.add_parameter( 'k', COMMAND_LINE_FLOAT,  1, "Kernel width", true, "5.5" );
  parms.add_parameter( 'a', COMMAND_LINE_INT,    1, "Number of dimensions", true);
  parms.add_parameter( 'c', COMMAND_LINE_INT,    1, "Do cut off", true);

  parms.parse_parameter_list(argc, argv);
  if( parms.all_required_parameters_set() ){
    std::cout << " Running reconstruction with the following parameters: " << std::endl;
    parms.print_parameter_list();
  }
  else{
    std::cout << " Some required parameters are missing: " << std::endl;
    parms.print_parameter_list();
    parms.print_usage();
    return 1;
  }
  


  // Rading data and extracting parameters
  boost::shared_ptr < hoNDArray<std::complex<float>> > samples_host = read_nd_array<std::complex<float>> ((char*)parms.get_parameter('d')->get_string_value());
  if( !(samples_host->get_number_of_dimensions() == 3) ){
    std::cout << std::endl << "Samples ndarray is not three-dimensional (samples/profile x #profiles x #coils). Quitting.\n" << std::endl;
    return 1;
  }

  hoNDArray< std::complex<float> > samples(samples_host.get());

  int NsamplesPerSpoke = samples_host->get_size(0);
  int Nspokes = samples_host->get_size(1);
  int Ncoils = samples_host->get_size(2);
  int grid_size = parms.get_parameter('m')->get_int_value();
  int over_sampling = parms.get_parameter('o')->get_int_value();
  float sigma = 0.7 / float(grid_size);
  int Ndim = parms.get_parameter('a')->get_int_value();
 

  
  //Trajectory and DCF
  GATrajectory traj;
  hoNDArray<float> trajectory(Nspokes*NsamplesPerSpoke,Ndim); 
  bool cut_off = parms.get_parameter('c')->get_int_value();
  traj.calculateTrajectory(trajectory, Nspokes, NsamplesPerSpoke);
  //std::cout << "Before DCF." << std::endl;
  hoNDArray<float> DCF(Nspokes*NsamplesPerSpoke);
  traj.calculateDCF(DCF, Nspokes, NsamplesPerSpoke, Ndim, cut_off, grid_size);
  //std::cout << "After DCF." << std::endl;

  
  //Gridding 
  int output_dimensions[Ndim];
  output_dimensions[0] = grid_size;  output_dimensions[1] = grid_size; 
  if (Ndim ==3)  output_dimensions[2] = grid_size;; 
   
  for (int n = 0 ; n < trajectory.get_number_of_elements() ; n++)
  {
    trajectory[n] = trajectory[n]*grid_size;
  }  

  int grid_s = grid_size * (int)over_sampling;

  Gridder<float>  myGrid(trajectory, output_dimensions, over_sampling);
  std::cout << "Calculating deapodization...." << std::endl;
  hoNDArray< std::complex<float> > a = myGrid.calculate_deapodization_filter();
  hoNDFFT<float>::instance()->ifft(&a,0);
  hoNDFFT<float>::instance()->ifft(&a,1);
  if (Ndim==3)   hoNDFFT<float>::instance()->ifft(&a,2);

  hoNDArray< std::complex <float> > kspace;
  if (Ndim==2)
  {
    kspace.create(grid_s, grid_s, Ncoils);
  }
  else
  {
    kspace.create(grid_s, grid_s, grid_s, Ncoils);
  }

  std::cout << "Gridding..." << std::endl;
  hoNDArray< std::complex<float> > samples_single_coil;
  
  clock_t grid_tot_beg = clock();
  for (int c = 0 ; c < Ncoils ; c++)
  {
    clock_t grid_beg = clock();
    std::cout << "    Coil " << c+1 << " of " << Ncoils;
    samples_single_coil.create(NsamplesPerSpoke, Nspokes);

    for (int i = 0 ; i < samples_single_coil.get_number_of_elements() ; i++)
    {
      samples_single_coil[i] = samples[NsamplesPerSpoke*Nspokes*c + i];
    }
    hoNDArray< std::complex<float> > grid_out;
    if (Ndim==2)
    {
      grid_out = myGrid.convolution_2d(samples_single_coil, &DCF, 1);
    }
    else
    {
      grid_out = myGrid.convolution_3d(samples_single_coil, &DCF, 1);
      //write_nd_array< std::complex<float> >(&grid_out, "/home/holstk/kspace.cplx");
    }

    for (int j = 0 ; j < grid_out.get_number_of_elements() ; j++)
    {
      if (Ndim==2)
      {
	kspace[grid_s*grid_s*c + j] = grid_out[j];
      }
      else
      {
	kspace[grid_s*grid_s*grid_s*c + j] = grid_out[j];
      }
    }
    clock_t grid_end = clock();
    double elapsed_sec_grid = double(grid_end-grid_beg)/CLOCKS_PER_SEC;
    std::cout << ", " << elapsed_sec_grid << " seconds" << std::endl;
  }
  clock_t grid_tot_end = clock();
  double elapsed_sec_grid_tot = double(grid_tot_end-grid_tot_beg)/CLOCKS_PER_SEC;
  std::cout << "Total gridder time: " << elapsed_sec_grid_tot << " seconds" << std::endl;
  
  //write_nd_array< std::complex<float> >(&kspace, "/home/holstk/kspace.cplx");

  //IFFT
  std::cout << "IFFT...";
  clock_t ifft_beg = clock();
  hoNDArray<std::complex<float>> images(kspace);
  hoNDFFT<float>::instance()->ifft(&images,0);
  hoNDFFT<float>::instance()->ifft(&images,1);
  if (Ndim == 3)
  {
    hoNDFFT<float>::instance()->ifft(&images,2);
  }
  clock_t ifft_end = clock();
  double elapsed_sec_ifft = double(ifft_end-ifft_beg)/CLOCKS_PER_SEC;
  std::cout << ", " << elapsed_sec_ifft << " seconds" <<std::endl;

  //Cutting off oversampled edges
  int edge = floor((grid_size*over_sampling-grid_size)/2);
  std::cout << "edge: " << edge << std::endl;
  hoNDArray< std::complex<float> > images_cut;
  if (Ndim==2)
  {
    images_cut.create(grid_size,grid_size,Ncoils);
    for (int c = 0 ; c < Ncoils ; c++)
    {
      for (int x = edge ; x < edge+grid_size ; x++)
      {
	for (int y = edge ; y < edge+grid_size ; y++)
	{
	  images_cut[(x-edge) + (y-edge)*grid_size + c*grid_size*grid_size] = images[x + y*grid_s + c*grid_s*grid_s] / a[x + y*grid_s];
	  //std::cout << "x = " << x << ", y = " << y << ", c = " << c << std::endl;
	}
      }
    }
  }
  else
  {
    images_cut.create(grid_size,grid_size,grid_size,Ncoils);
    for (int c = 0 ; c < Ncoils ; c++)
    {
      for (int x = edge ; x < edge+grid_size ; x++)
      {
	for (int y = edge ; y < edge+grid_size ; y++)
	{
	  for (int z = edge ; z < edge+grid_size ; z++)
	  {
	    images_cut[(x-edge) + (y-edge)*grid_size + (z-edge)*grid_size*grid_size + c*grid_size*grid_size*grid_size] = images[x + y*grid_s + z*grid_s*grid_s + c*grid_s*grid_s*grid_s] / a[x + y*grid_s + z*grid_s*grid_s];
	  }
	}
      }
    }
  }
  //Root-SS and applying deappodization
  std::cout << "Applying deapodization and performing root-SS...";
  int elements_per_coil;
  hoNDArray<float> im;
  clock_t begin = clock();
  if (Ndim == 2)
  {
    elements_per_coil = grid_size * grid_size;
    im.create(grid_size, grid_size);
  }
  else
  {
    elements_per_coil = grid_size * grid_size * grid_size;
    im.create(grid_size, grid_size, grid_size);
  }

  for (int i = 0 ; i < elements_per_coil ; i++)
  {
    float SS = 0;
    std::complex<float> weighted = 0;
    for (int c = 0 ; c < Ncoils ; c++)
    {
      weighted = images_cut[i + c*elements_per_coil];
      SS = SS + abs(weighted)*abs(weighted);
    }
    im[i] = sqrt(SS);
  }
  clock_t end = clock();
  double elapsed_sec = double(end-begin)/CLOCKS_PER_SEC;
  std::cout << ", " << elapsed_sec << " seconds" << std::endl;

  //Writing the image
  write_nd_array<float>(&im, parms.get_parameter('r')->get_string_value());

  
  



  /*  
  // Configuration from the command line
  uint64d2 matrix_size = uint64d2(parms.get_parameter('m')->get_int_value(), parms.get_parameter('m')->get_int_value());
  uint64d2 matrix_size_os = uint64d2(parms.get_parameter('o')->get_int_value(), parms.get_parameter('o')->get_int_value());
  _real kernel_width = parms.get_parameter('k')->get_float_value();

  unsigned int num_profiles = host_samples->get_size(1);
  unsigned int samples_per_profile = host_samples->get_size(0);  
  _real alpha = (_real)matrix_size_os.vec[0]/(_real)matrix_size.vec[0];

  // Upload host data to device
  timer = new GPUTimer("Uploading samples to device");
  cuNDArray<_complext> samples(host_samples.get());
  delete timer;
  
  // Setup resulting image array
  vector<size_t> image_dims = to_std_vector(matrix_size);
  cuNDArray<_complext> image(&image_dims);
  
  // Initialize plan
  timer = new GPUTimer("Initializing plan");
  plan_type plan( matrix_size, matrix_size_os, kernel_width );
  delete timer;

  // Compute trajectories
  timer = new GPUTimer("Computing golden ratio radial trajectories");
  boost::shared_ptr< cuNDArray<_reald2> > traj = compute_radial_trajectory_golden_ratio_2d<_real>( samples_per_profile, num_profiles,  1 );
  delete timer;
  
  // Preprocess
  timer = new GPUTimer("NFFT preprocessing");
  plan.preprocess( traj.get(), plan_type::NFFT_PREP_NC2C );
  delete timer;

  // Compute density compensation weights
  timer = new GPUTimer("Computing density compensation weights");
  boost::shared_ptr< cuNDArray<_real> > dcw = compute_radial_dcw_golden_ratio_2d
    ( samples_per_profile, num_profiles, alpha, _real(1)/((_real)samples_per_profile/(_real)matrix_size.vec[0]) );
  delete timer;

  // Gridder
  timer = new GPUTimer("Computing adjoint nfft (gridding)");
  plan.compute( &samples, &image, dcw.get(), plan_type::NFFT_BACKWARDS_NC2C );
  delete timer;

  //
  // Output result
  //
  
  timer = new GPUTimer("Output result to disk");

  boost::shared_ptr< hoNDArray<_complext> > host_image = image.to_host();
  write_nd_array<_complext>( host_image.get(), (char*)parms.get_parameter('r')->get_string_value());


  boost::shared_ptr< hoNDArray<_real> > host_norm = abs(&image)->to_host();
  write_nd_array<_real>( host_norm.get(), "result.real" );

  delete timer;
  */

  clock_t all_end = clock();
  double elapsed_sec_all = double(all_end-all_beg)/CLOCKS_PER_SEC;
  std::cout << "Total recon time: " << elapsed_sec_all << " seconds" <<  std::endl;

  return 0;
}
