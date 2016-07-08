//#include "cuNFFT.h"
#include "radial_utilities.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
//#include "cuNDArray_elemwise.h"
//#include "GPUTimer.h"
#include "parameterparser.h"
//#include "complext.h"
#include <complex>
#include "ga2d_trajectory.cpp"
#include "/home/holstk/code/gadgetron/toolboxes/fft/cpu/hoNDFFT.h"
#include "hoNDArray.h"

#include <iostream>

//using namespace std;
using namespace Gadgetron;

// Define desired precision
//typedef float _real; 
//typedef complext<_real> _complext;
//typedef reald<_real,2>::Type _reald2;
//typedef cuNFFT_plan<_real,2> plan_type;

int main( int argc, char** argv) 
{
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
  int oversampling = parms.get_parameter('o')->get_int_value();
  float sigma = 0.7 / float(grid_size);
  int Ndim = parms.get_parameter('a')->get_int_value();
  std::cout << "Ndims: " << Ndim << std::endl;
  std::cout << "Ncoils: " << Ncoils << std::endl;


  
  //Trajectory and DCF
  ga2dTraj ga2dobj;
  hoNDArray<float> trajectory(Ndim, Nspokes*NsamplesPerSpoke); 
  bool cut_off = parms.get_parameter('c')->get_int_value();
  std::cout << "cout_off: " << cut_off << std::endl;
  ga2dobj.calculateTrajectory(trajectory, Nspokes, NsamplesPerSpoke);
  std::cout << "Ndim in trajectory: " << trajectory.get_size(0) << std::endl;

  hoNDArray<float> DCF(Nspokes*NsamplesPerSpoke);
  ga2dobj.calculateDCF(DCF, Nspokes, NsamplesPerSpoke, Ndim, cut_off, grid_size);


  
  //Gridding
  hoNDArray<std::complex<float>> kspace_grid;
  if (Ndim == 2)
  {
    kspace_grid.create(grid_size, grid_size, Ncoils);
  }
  else
  {
    kspace_grid.create(grid_size, grid_size, grid_size, Ncoils);
  }
  ga2dobj.gridder(samples, kspace_grid, trajectory, DCF, sigma, NsamplesPerSpoke, Nspokes, grid_size, Ndim, Ncoils);

  //Writing the gridded k-space
  //write_nd_array<std::complex<float>>(&kspace_grid, "/home/holstk/grid_result_one_coil.cplx");

  


  //IFFT
  hoNDArray<std::complex<float>> image(kspace_grid);
  hoNDFFT<float>::instance()->ifft(&image,0);
  hoNDFFT<float>::instance()->ifft(&image,1);
  if (Ndim == 3)
  {
    hoNDFFT<float>::instance()->ifft(&image,2);
  }
  

  
  //Apodization
  hoNDArray< std::complex<float> > W;

  if (Ndim == 2){
    W.create(grid_size,grid_size);
    ga2dobj.apodization(sigma, W);

    for (int nx = 0 ; nx < grid_size ; nx++)
    {
      for (int ny = 0 ; ny < grid_size ; ny++)
      {
	for (int c = 0 ; c < Ncoils ; c++)
	{
	  image[(nx + ny*grid_size) + c*grid_size*grid_size]  =  image[(nx + ny*grid_size) + c*grid_size*grid_size]  /  W[nx + ny*grid_size];
	}
      }
    }
  }
  else {
    W.create(grid_size,grid_size,grid_size);
    ga2dobj.apodization(sigma, W);
   
    for (int nx = 0 ; nx < grid_size ; nx++)
    {
      for (int ny = 0 ; ny < grid_size ; ny++)
      {
	for (int nz = 0 ; nz < grid_size ; nz++)
	{
	  for (int c = 0 ; c < Ncoils ; c++)
	  {
	    image[(nx + ny*grid_size + nz*grid_size*grid_size) + c*grid_size*grid_size*grid_size]  =  image[(nx + ny*grid_size + nz*grid_size*grid_size) + c*grid_size*grid_size*grid_size]  /  W[nx + ny*grid_size + nz*grid_size*grid_size];
	  }
	}
      }
    }
  }
  


  
  //Root-SS
  int elements_per_coil;
  hoNDArray<float> im;
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
    float tmp = 0;
    for (int c = 0 ; c < Ncoils ; c++)
    {
      tmp = tmp + abs(image[i + c*elements_per_coil]) * abs(image[i + c*elements_per_coil]);
    }
    im[i] = sqrt(tmp);
  }



  //Writing the image
  //write_nd_array<float>(&im, "/home/holstk/image3D.real");
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

  return 0;
}
