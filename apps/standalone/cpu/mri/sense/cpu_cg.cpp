/*

  Sample application of the NFFT toolbox: standalone "gridding" example.

  -----------

  The nfft is written generically and templetized to
  - transform arbitrary trajectories
  - transform an arbitrary number of dimensions (currently instantiated for 1d/2d/3d/4d)
  - support both single and double precision

  General principles of the implementation can be found in:

  Accelerating the Non-equispaced Fast Fourier Transform on Commodity Graphics Hardware.
  T.S. Sørensen, T. Schaeffter, K.Ø. Noe, M.S. Hansen. 
  IEEE Transactions on Medical Imaging 2008; 27(4):538-547.

  Real-time Reconstruction of Sensitivity Encoded Radial Magnetic Resonance Imaging Using a Graphics Processing Unit.
  T.S. Sørensen, D. Atkinson, T. Schaeffter, M.S. Hansen.
  IEEE Transactions on Medical Imaging 2009; 28(12):1974-1985. 

  This example programme of the nnft utilizes golden ratio based radial trajectories 
  and outputs gridded images from 2D multislice input ndarrays of the corresponding samples, trajectory, and density compensation weights.

*/

#include "hoNFFT.h"
#include "ga_trajectory.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_reductions.h"
#include "hoNonCartesianSenseOperator.h"
#include "hoCgPreconditioner.h"
#include "hoImageOperator.h"
#include "hoCgSolver.h"
#include "parameterparser.h"
#include "complext.h"
#include "ga_trajectory.h"
#include "cpu_b1_map.h"

#include <iostream>

using namespace std;
using namespace Gadgetron;

// Define desired precision
typedef float _real; 
typedef complext<_real> _complext;
typedef reald<_real,2>::Type _reald2;

int main( int argc, char** argv) 
{

    //
    // Parse command line
    //

    ParameterParser parms;
    parms.add_parameter( 'd', COMMAND_LINE_STRING, 1, "Input samples file name (.cplx)", true );
    std::cout << "Input samples" << std::endl;
    //parms.add_parameter( 't', COMMAND_LINE_STRING, 1, "Input trajectories file name (.real)", true );
    //parms.add_parameter( 'w', COMMAND_LINE_STRING, 1, "Input density compensation weights file name (.real)", true );

    parms.add_parameter( 'c', COMMAND_LINE_STRING, 1, "Input coil sensitivity maps file name (.cplx)", true );
    parms.add_parameter( 'g', COMMAND_LINE_STRING, 1, "Input regularization image file name (.cplx)", true );
    std::cout << "Input reg image" << std::endl;

    parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Output image file name (.cplx)", true, "result.cplx" );
    parms.add_parameter( 'i', COMMAND_LINE_INT,    1, "Number of iterations", true, "10" );
    parms.add_parameter( 'l', COMMAND_LINE_FLOAT,  1, "Regularization weight", true, "0.3" );
    parms.add_parameter( 'k', COMMAND_LINE_FLOAT,  1, "Kernel width", true, "5.5" );
    parms.add_parameter( 'a', COMMAND_LINE_FLOAT,  1, "Oversampling factor", true, "2.0" );
    std::cout << "Input all" << std::endl;

    parms.parse_parameter_list(argc, argv);
    if( parms.all_required_parameters_set() ){
	cout << " Running reconstruction with the following parameters: " << endl;
	parms.print_parameter_list();
    }
    else{
	cout << " Some required parameters are missing: " << endl;
	parms.print_parameter_list();
	parms.print_usage();
	return 1;
    }
  
  
    // Load data from disk
    boost::shared_ptr< hoNDArray<_complext> > host_samples = read_nd_array<_complext> ((char*)parms.get_parameter('d')->get_string_value());
    std::cout << "Loading host_samples" << std::endl;
    hoNDArray<_complext> samples(host_samples.get());
    std::cout << "Loading samples" << std::endl;
    //boost::shared_ptr< hoNDArray<_reald2> >   host_traj    = read_nd_array<_reald2>   ((char*)parms.get_parameter('t')->get_string_value());
    //boost::shared_ptr< hoNDArray<_real> >     host_dcw     = read_nd_array<_real>     ((char*)parms.get_parameter('w')->get_string_value());
    boost::shared_ptr< hoNDArray<_complext> > csm     = read_nd_array<_complext> ((char*)parms.get_parameter('c')->get_string_value());
    boost::shared_ptr< hoNDArray<_complext> > host_reg     = read_nd_array<_complext> ((char*)parms.get_parameter('g')->get_string_value());
    std::cout << "Loading reg image" << std::endl;

    int NsamplesPerSpoke = samples.get_size(0);
    int Nspokes = samples.get_size(1);
    int Ndims = 2;
    GATrajectory<float,2> traj;
    hoNDArray<typename reald<float,2>::Type> trajectory;
    std::cout << "Traj dims: " << trajectory.get_number_of_dimensions() << std::endl;
    trajectory = traj.calculateTrajectory(Nspokes, NsamplesPerSpoke, Ndims);
    std::cout << "Calc trajectory" << std::endl;
    hoNDArray<float> dcw_tmp = traj.calculateDCF(Nspokes, NsamplesPerSpoke, Ndims, 100, NsamplesPerSpoke/2); //cut_off and grid_size in the end
    boost::shared_ptr< hoNDArray<_real> >dcw( new hoNDArray<_real>(dcw_tmp) );
    std::cout << "Calc dcw" << std::endl;

    //hoNDArray<complext<float> > csm_tmp;
    //estimate_b1_map<float,2>( &samples, &csm_tmp, samples.get_size(2));
    //std::cout << "Calc csm" << std::endl;
    //boost::shared_ptr< hoNDArray<_complext> >csm( new hoNDArray<_complext>(csm_tmp) );
    //std::cout << "Make csm shared pointer" << std::endl;

    /* {
       std::vector<size_t> dims;
       dims.push_back(host_traj->get_size(0));
       dims.push_back(host_samples->get_number_of_elements()/dims[0]);
       host_samples->reshape(&dims);
       } */

    std::cout << "Traj dims: " << trajectory.get_number_of_dimensions() << std::endl;
    std::cout << "Sample dims: " << samples.get_number_of_dimensions() << std::endl;
    //if( !(samples.get_number_of_dimensions() == 2 && trajectory.get_number_of_dimensions() == 2) ){
    //	cout << endl << "Samples/trajectory arrays must be two-dimensional: (dim 0: samples/profile x #profiles/frame; dim 1: #frames). Quitting.\n" << endl;
    //	return 1;
    //}

    if( !(csm->get_number_of_dimensions() == 3 )){
	cout << endl << "Coil sensitivity maps must be three-dimensional. Quitting.\n" << endl;
	return 1;
    }

    //if( !(host_reg->get_number_of_dimensions() == 2 )){
    //	cout << endl << "Regularization image must be two-dimensional. Quitting.\n" << endl;
    //	return 1;
    //}

    // Configuration from the command line
    uint64d2 matrix_size = uint64d2(csm->get_size(0), csm->get_size(0));
    size_t _matrix_size_os = size_t((float)matrix_size[0]*parms.get_parameter('a')->get_float_value());
    uint64d2 matrix_size_os = uint64d2(_matrix_size_os, _matrix_size_os);
    int num_iterations = parms.get_parameter('i')->get_int_value();
    _real kernel_width = parms.get_parameter('k')->get_float_value();
    _real alpha = parms.get_parameter('a')->get_float_value();
    _real kappa = parms.get_parameter('l')->get_float_value();
  
    unsigned int num_frames = trajectory.get_size(1);  
    unsigned int num_coils = csm->get_size(2);

    std::cout << "num frames, coils: " << num_frames << ", " << num_coils << std::endl;

    std::vector<size_t> recon_dims = to_std_vector(matrix_size);
    recon_dims.push_back(num_frames);

    // Upload arrays to device
    //hoNDArray<_complext> samples(host_samples.get());
    //hoNDArray<_reald2> trajectory(host_traj.get());
    //boost::shared_ptr< hoNDArray<_complext> > csm( new hoNDArray<_complext>(host_csm.get()));
    boost::shared_ptr< hoNDArray<_complext> > reg_image( new hoNDArray<_complext>(host_reg.get()));
    //boost::shared_ptr< hoNDArray<_real> > dcw( new hoNDArray<_real>(host_dcw.get()));

    // Define encoding matrix for non-Cartesian SENSE
    boost::shared_ptr< hoNonCartesianSenseOperator<_real,2> > E( new hoNonCartesianSenseOperator<_real,2>() );  
    E->setup( matrix_size, matrix_size_os, kernel_width );
    E->set_dcw(dcw) ;
    E->set_csm(csm);
    E->set_domain_dimensions(&recon_dims);
    E->set_codomain_dimensions(samples.get_dimensions().get());
    E->preprocess(&trajectory);
    std::cout << "After E->preprocess" << std::endl;
  
    // Define regularization operator
    boost::shared_ptr< hoImageOperator<_complext> > R( new hoImageOperator<_complext>() );
    std::cout << "After R" << std::endl;
    R->set_weight( kappa );
    std::cout << "After R->set_weight" << std::endl;
    R->compute( reg_image.get() );
    std::cout << "After R->compute" << std::endl;

    boost::shared_ptr< hoNDArray<_real> > _precon_weights = sum(abs_square(csm.get()).get(), 2);
    boost::shared_ptr< hoNDArray<_real> > R_diag = R->get();
    std::cout << "_precon_weights dims: " << _precon_weights->get_number_of_dimensions() << std::endl;
    std::cout << "_precon_weights dims: " << _precon_weights->get_size(0) << std::endl;
    std::cout << "_precon_weights dims: " << _precon_weights->get_size(1) << std::endl;
    std::cout << "R_diag dims: " << R_diag->get_number_of_dimensions() << std::endl;
    std::cout << "R_diag dims: " << R_diag->get_size(0) << std::endl;
    std::cout << "R_diag dims: " << R_diag->get_size(1) << std::endl;
    std::cout << "R_diag dims: " << R_diag->get_size(2) << std::endl;
    std::cout << "After R->get" << std::endl;
    *R_diag *= kappa;
    std::cout << "After R_diag *= kappa" << std::endl;
    *_precon_weights += *R_diag;
    std::cout << "After precon += R_diag" << std::endl;
    R_diag.reset();
    reciprocal_sqrt_inplace(_precon_weights.get());
    boost::shared_ptr< hoNDArray<_complext> > precon_weights = real_to_complex<_complext>( _precon_weights.get() );
    _precon_weights.reset();

    // Define preconditioning matrix
    boost::shared_ptr< hoCgPreconditioner<_complext> > D( new hoCgPreconditioner<_complext>() );
    D->set_weights( precon_weights );
    precon_weights.reset();
    csm.reset();

    // Setup conjugate gradient solver
    hoCgSolver<_complext> cg;
    cg.set_preconditioner ( D );           // preconditioning matrix
    cg.set_max_iterations( num_iterations );
    cg.set_tc_tolerance( 1e-6 );
    cg.set_output_mode( hoCgSolver< _complext>::OUTPUT_VERBOSE );
    cg.set_encoding_operator( E );        // encoding matrix
    cg.add_regularization_operator( R );  // regularization matrix

    //
    // Invoke conjugate gradient solver
    //
  
    boost::shared_ptr< hoNDArray<_complext> > cgresult;
    //hoNDArray<_complext>* cgresult;
    {
	//GPUTimer timer("GPU Conjugate Gradient solve");
	std::cout << "Just before solve" << std::endl;
	cgresult = cg.solve(&samples);
	std::cout << "Just after solve" << std::endl;
    }
  
    //
    // Output result
    //
  
    //timer = new GPUTimer("Output result to disk");
    //boost::shared_ptr< hoNDArray<_complext> > host_image = cgresult->to_host();
    write_nd_array<_complext>( cgresult.get(), (char*)parms.get_parameter('r')->get_string_value() );
    //write_nd_array<_real>( abs(cgresult.get()), "result.real" );
    //delete timer;

    return 0;
}
