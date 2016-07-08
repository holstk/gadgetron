#include <iostream>
#include <complex>
#include "math.h"
#include "hoNDArray.h"
#include "vector_td_utilities.h"
#include "hoNDArray_elemwise.h"
#include "hoNDFFT.h"

using namespace Gadgetron;

class GATrajectory {
  public:
  int calculateTrajectory(hoNDArray<float>& trajectory, int Nspokes, int NsamplesPerSpoke){
    if (trajectory.get_size(1) < 2 || trajectory.get_size(1) > 3)
    {
      std::cerr << "Only 2 or 3 dimensions are supported." << std::endl;
      return 1;
    }

    if (trajectory.get_size(1) == 2) // 2D
    {
      float goldenRatio = M_PI / ((sqrt(5)+1)/2);
    
      float basisSpoke[NsamplesPerSpoke];
      for (int n = 0 ; n < NsamplesPerSpoke ; n++)
      {
	basisSpoke[n] = -0.5 + 1/float(NsamplesPerSpoke) * n;
      }
 
      for (int sp = 0 ; sp < Nspokes ; sp++)
      {
	for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
	{
	  //trajectory[sp*NsamplesPerSpoke*2 + sa*2]   = basisSpoke[sa]*cos(sp*goldenRatio);  
	  //trajectory[sp*NsamplesPerSpoke*2 + sa*2+1] = basisSpoke[sa]*sin(sp*goldenRatio);
	  trajectory[sp*NsamplesPerSpoke + sa]   = basisSpoke[sa]*cos(sp*goldenRatio);  
	  trajectory[Nspokes*NsamplesPerSpoke + sp*NsamplesPerSpoke + sa] = basisSpoke[sa]*sin(sp*goldenRatio);	}
      }
    }
    else
    {
      float golden_mean1 = 0.465571231876768;
      float golden_mean2 = 0.682327803828020;

      float basisSpoke[NsamplesPerSpoke];
      for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
      {
	basisSpoke[sa] = -0.5 + 1/float(NsamplesPerSpoke) * sa;
      }
      
      float fractional_part1 = 0;
      float fractional_part2 = 0;
      float polar_angle = 0;
      float azimuthal_angle = 0;
      for (int sp = 0 ; sp < Nspokes ; sp++)
      {
	fractional_part1 = sp * golden_mean1 - floor(sp * golden_mean1);
	fractional_part2 = sp * golden_mean2 - floor(sp * golden_mean2);
	polar_angle = acos(fractional_part1);
	azimuthal_angle = 2 * M_PI * fractional_part2;

	for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
	{
	  //trajectory[sp*NsamplesPerSpoke*3 + sa*3]     = basisSpoke[sa] * cos(azimuthal_angle) * sin(polar_angle);
	  //trajectory[sp*NsamplesPerSpoke*3 + sa*3 + 1] = basisSpoke[sa] * sin(azimuthal_angle) * sin(polar_angle);
	  //trajectory[sp*NsamplesPerSpoke*3 + sa*3 + 2] = basisSpoke[sa] *                        cos(polar_angle);
	  trajectory[                             sp*NsamplesPerSpoke + sa] = basisSpoke[sa] * cos(azimuthal_angle) * sin(polar_angle);
	  trajectory[Nspokes*NsamplesPerSpoke   + sp*NsamplesPerSpoke + sa] = basisSpoke[sa] * sin(azimuthal_angle) * sin(polar_angle);
	  trajectory[Nspokes*NsamplesPerSpoke*2 + sp*NsamplesPerSpoke + sa] = basisSpoke[sa] *                        cos(polar_angle);
	}
      }
    }
    return 0;
  }




  int calculateDCF(hoNDArray<float>& DCF_out, int Nspokes, int NsamplesPerSpoke, int Ndim, bool cut_off, int grid_size){

    if (Ndim > 3 && Ndim < 2){
      std::cerr << "Only 2 or 3 dimensions are supported." << std::endl;
      return 1;
    }
      
    float full_sampling = 0;
    if (Ndim == 2){
      full_sampling = grid_size * M_PI / 2;
    }
    else {
      full_sampling = grid_size * grid_size * M_PI / 2;
    }
    float cut;
    if (cut_off){
      cut = Nspokes / full_sampling;
    }
    else {
      cut = 1;
    }
    int cut_idx = round(NsamplesPerSpoke / 2 * cut);

    

    float DCFstep = 2/(float(NsamplesPerSpoke) - 1);
    float baseDCF[NsamplesPerSpoke];
    for (int n = 0 ; n < NsamplesPerSpoke ; n++)
    {
      if (n > NsamplesPerSpoke/2 - cut_idx - 1  &&  n < NsamplesPerSpoke/2 + cut_idx){
	baseDCF[n] = abs(-1 + n * DCFstep);
      }
      else {
	baseDCF[n] = abs(-1 + (NsamplesPerSpoke/2-cut_idx) * DCFstep);
      }
    }
    
   
    for (int sp = 0 ; sp < Nspokes ; sp++)
    {
      for (int sa = 0 ; sa < NsamplesPerSpoke ; sa ++)
      {
	if (Ndim == 2){
	  DCF_out[sp * NsamplesPerSpoke + sa] = baseDCF[sa];
	}
	else {
	  DCF_out[sp * NsamplesPerSpoke + sa] = baseDCF[sa] * baseDCF[sa];
	}
      }
    }
    
    return 0;
  }




  float gaussKernel(float sigma,float f[], int Ndim){
    float w;
    if (Ndim == 2)
    {
      w = exp(-(f[0]*f[0] + f[1]*f[1]) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
    }
    else
    {
      w = exp(-(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]) / (2.0 * sigma * sigma));// / (2.0 * M_PI * sigma * sigma);
      w = w / ( (sqrt(2.0 * M_PI) * sigma)  *  (sqrt(2.0 * M_PI) * sigma)  *  (sqrt(2.0 * M_PI) * sigma) );
    }
    return w;
  }
  


  int deapodization(float sigma, hoNDArray< std::complex<float> >& W){
    if (W.get_number_of_dimensions() == 2){
      float f[2];
      for (int nx = 0 ; nx < W.get_size(0) ; nx++)
      {
	for (int ny = 0 ; ny < W.get_size(1) ; ny++)
	{
	  f[0] = -0.5 + 1/float(W.get_size(0)) * float(nx);
	  f[1] = -0.5 + 1/float(W.get_size(1)) * float(ny);
	  W[nx + ny*W.get_size(0)] = gaussKernel(sigma, f, W.get_number_of_dimensions());
	}
      }
      hoNDFFT<float>::instance()->ifft(&W,0);
      hoNDFFT<float>::instance()->ifft(&W,1);
    }
    else {
      float f[3];
      for (int nx = 0 ; nx < W.get_size(0) ; nx++)
      {
	for (int ny = 0 ; ny < W.get_size(1) ; ny++)
	{
	  for (int nz = 0 ; nz < W.get_size(2) ; nz++)
	  {
	    f[0] = -0.5 + 1/float(W.get_size(0)) * float(nx);
	    f[1] = -0.5 + 1/float(W.get_size(1)) * float(ny);
	    f[2] = -0.5 + 1/float(W.get_size(2)) * float(nz);
	    W[nx + ny*W.get_size(0) + nz*W.get_size(0)*W.get_size(1)] = gaussKernel(sigma, f, W.get_number_of_dimensions());
	  }
	}
      }
      hoNDFFT<float>::instance()->ifft(&W,0);
      hoNDFFT<float>::instance()->ifft(&W,1);
      hoNDFFT<float>::instance()->ifft(&W,2);
    }

    return 0;
  }



  int  gridder(hoNDArray< std::complex<float> >& samples, hoNDArray< std::complex<float> >& kspace_grid, hoNDArray<float>& trajectory, hoNDArray<float>& DCF, float sigma, int NsamplesPerSpoke, int Nspokes, int grid_size, int Ndim, int Ncoils){
    //int hoNDArray<std::complex<float>> gridder(hoNDArray< std::complex<float> > samples_new, std::complex<float>* trajectory, float* DCF, float sigma, int NsamplesPerSpoke, int Nspokes, int grid_size){
    /*if (trajectory == 0) {
        std::cerr << "Error: passing zero-pointer" << std::endl;
        return 1;
    }
    if (DCF == 0) {
        std::cerr << "Error: passing zero-pointer" << std::endl;
        return 1;
    }*/
    
    //hoNDArray<std::complex<float>> kspace_grid(grid_size,grid_size);

    float gridStmp = float(grid_size);
    
    if (trajectory.get_size(1) == 2)
    {
      float f[2];
      for (int n = 0 ; n < NsamplesPerSpoke*Nspokes ; n++)
	{
	  float kx_nc = trajectory[2*n];  
	  float ky_nc = trajectory[2*n+1]; 
	  float kx_min = floor((kx_nc - 3*sigma) * gridStmp) + grid_size/2;   if (kx_min < 0 || isinf(kx_min))  kx_min = 0;            //std::cout << "kx_min: " << kx_min << std::endl;
	  float kx_max = ceil( (kx_nc + 3*sigma) * gridStmp) + grid_size/2;   if (kx_max > grid_size-1)         kx_max = grid_size-1;  //std::cout << "kx_max: " << kx_max << std::endl;
	  float ky_min = floor((ky_nc - 3*sigma) * gridStmp) + grid_size/2;   if (ky_min < 0 || isinf(kx_min))  ky_min = 0;            //std::cout << "ky_min: " << ky_min << std::endl;
	  float ky_max = ceil( (ky_nc + 3*sigma) * gridStmp) + grid_size/2;   if (ky_max > grid_size-1)         ky_max = grid_size-1;  //std::cout << "ky_max: " << ky_max << std::endl;
	  for (int nx = kx_min ; nx < kx_max+1 ; nx++)
	    {
	      for (int ny = ky_min ; ny < ky_max+1 ; ny++)
		{
		  f[0] = -0.5 + float(nx)/grid_size - kx_nc;
		  f[1] = -0.5 + float(ny)/grid_size - ky_nc;
		  float kernelVal = gaussKernel(sigma, f, trajectory.get_size(0)); 
		  for (int c = 0 ; c < Ncoils ; c++)
		    {
		      kspace_grid[(nx + ny*grid_size) + c*grid_size*grid_size] = kspace_grid[(nx + ny*grid_size) + c*grid_size*grid_size] + kernelVal * samples[n + c*NsamplesPerSpoke*Nspokes] * DCF[n];
		    }
		}

	    }
	}
    }
    else
      {
      float f[3];
      for (int n = 0 ; n < NsamplesPerSpoke*Nspokes ; n++)
	{
	  float kx_nc = trajectory[3*n];  
	  float ky_nc = trajectory[3*n+1];
	  float kz_nc = trajectory[3*n+2];
	  float kx_min = floor((kx_nc - 3*sigma) * gridStmp) + grid_size/2;   if (kx_min < 0 || isinf(kx_min))  kx_min = 0;            //std::cout << "kx_min: " << kx_min << endl;
	  float kx_max = ceil( (kx_nc + 3*sigma) * gridStmp) + grid_size/2;   if (kx_max > grid_size-1)         kx_max = grid_size-1;  //std::cout << "kx_max: " << kx_max << endl;
	  float ky_min = floor((ky_nc - 3*sigma) * gridStmp) + grid_size/2;   if (ky_min < 0 || isinf(ky_min))  ky_min = 0;            //std::cout << "ky_min: " << ky_min << endl;
	  float ky_max = ceil( (ky_nc + 3*sigma) * gridStmp) + grid_size/2;   if (ky_max > grid_size-1)         ky_max = grid_size-1;  //std::cout << "ky_max: " << ky_max << endl;
	  float kz_min = floor((kz_nc - 3*sigma) * gridStmp) + grid_size/2;   if (kz_min < 0 || isinf(kz_min))  kz_min = 0;            //std::cout << "ky_min: " << ky_min << endl;
	  float kz_max = ceil( (kz_nc + 3*sigma) * gridStmp) + grid_size/2;   if (kz_max > grid_size-1)         kz_max = grid_size-1;  //std::cout << "ky_max: " << ky_max << endl;
	  for (int nx = kx_min ; nx < kx_max+1 ; nx++)
	    {
	      for (int ny = ky_min ; ny < ky_max+1 ; ny++)
		{
		  for (int nz = kz_min ; nz < kz_max+1 ; nz++)
		    {
		      f[0] = -0.5+float(nx)/grid_size - kx_nc;
		      f[1] = -0.5+float(ny)/grid_size - ky_nc;
		      f[2] = -0.5+float(nz)/grid_size - kz_nc;
		      float kernelVal = gaussKernel(sigma, f, trajectory.get_size(0)); 
		      for (int c = 0 ; c < Ncoils ; c++)
			{
			  kspace_grid[(nx + ny*grid_size + nz*grid_size*grid_size) + c*grid_size*grid_size*grid_size] = kspace_grid[(nx + ny*grid_size + nz*grid_size*grid_size) + c*grid_size*grid_size*grid_size] + kernelVal * samples[n + c*NsamplesPerSpoke*Nspokes] * DCF[n];
			}
		    }
		}
	    }
	}
      }
    
    return 0;
  }



};
