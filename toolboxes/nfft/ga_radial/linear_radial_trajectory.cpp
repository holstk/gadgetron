#include "linear_radial_trajectory.h"
#include "hoNFFT.h"

using namespace Gadgetron;

//template <class T> void GATrajectory<T>::GATrajectory();

//template <class T> void GATrajectory<T>::~GATrajectory();

template <class T, unsigned int D> hoNDArray<typename reald<T,D>::Type> LinearRadialTrajectory<T,D>::calculateTrajectory(int Nspokes, int NsamplesPerSpoke, int Ndim, long int *spoke_idx, int correction_type)
{
    if (D < 2 || D > 3)
    {
	throw std::invalid_argument("Received invalid dimension, only 2 or 3 dimensions are supported.");
    }

    int number_of_samples = NsamplesPerSpoke * Nspokes;
    trajectory.create(number_of_samples);//,Ndim);

    if (D == 2){ // 2D

	T angle_increment = M_PI / Nspokes;
    
	T basisSpoke[NsamplesPerSpoke];
	for (int n = 0 ; n < NsamplesPerSpoke ; n++)
	{
	    basisSpoke[n] = -0.5 + 1/float(NsamplesPerSpoke) * n;
	}
	
	typename reald<T,D>::Type sample_pos;

	for (int sp = 0 ; sp < Nspokes ; sp++)
	{
	    for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
	    {
		//trajectory[sp*NsamplesPerSpoke*2 + sa*2]   = basisSpoke[sa]*cos(sp*goldenRatio);  
		//trajectory[sp*NsamplesPerSpoke*2 + sa*2+1] = basisSpoke[sa]*sin(sp*goldenRatio);
		sample_pos[0] = basisSpoke[sa]*cos(sp*angle_increment);
		sample_pos[1] = basisSpoke[sa]*sin(sp*angle_increment);
		//trajectory[sp*NsamplesPerSpoke + sa]   = basisSpoke[sa]*cos(sp*goldenRatio);  
		//trajectory[Nspokes*NsamplesPerSpoke + sp*NsamplesPerSpoke + sa] = basisSpoke[sa]*sin(sp*goldenRatio);	}
		trajectory[sp*NsamplesPerSpoke + sa] = sample_pos;
	    }
	}
    }
    else
    {
	T basisSpoke[NsamplesPerSpoke];
       
	for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
	{
	    basisSpoke[sa] = -0.5 + 1/float(NsamplesPerSpoke) * sa;
	}
      
	if ( correction_type == 2 )
	    Nspokes = Nspokes - 6;

	calculate_angles(Nspokes, correction_type);

	typename reald<T,D>::Type sample_pos;
	for (int sp = 0 ; sp < Nspokes ; sp++)
	{	    
	    for (int sa = 0 ; sa < NsamplesPerSpoke ; sa++)
	    {
		sample_pos[0] = basisSpoke[sa] * cos(azi[sp]) * sin(pol[sp]);
		sample_pos[1] = basisSpoke[sa] * sin(azi[sp]) * sin(pol[sp]);
		sample_pos[2] = basisSpoke[sa] *                cos(pol[sp]);

		trajectory[sp*NsamplesPerSpoke + sa] = sample_pos;
	    }
	}
    }
    return trajectory;
}


template <class T, unsigned int D> hoNDArray<T> LinearRadialTrajectory<T,D>::calculateDCF(int Nspokes, int NsamplesPerSpoke, int Ndim, bool cut_off, int grid_size){

    if (Ndim > 3 || Ndim < 2)
    {
	throw std::invalid_argument("Received invalid dimension, only 2 or 3 dimensions are supported.");    
    }
      
    T full_sampling = 0;
    if (Ndim == 2)
    {
	full_sampling = grid_size * M_PI / 2;
    }
    else 
    {
	full_sampling = grid_size * grid_size * M_PI / 2;
    }
    T cut;
    if (cut_off){
	cut = Nspokes / full_sampling;
    }
    else {
	cut = 1;
    }
    int cut_idx = round(NsamplesPerSpoke / 2 * cut);

    DCF_out.create(NsamplesPerSpoke*Nspokes);

    T DCFstep = 2/(float(NsamplesPerSpoke) - 1);
    T baseDCF[NsamplesPerSpoke];

    for (int n = 0 ; n < NsamplesPerSpoke ; n++)
    {
	if (n > NsamplesPerSpoke/2 - cut_idx - 1  &&  n < NsamplesPerSpoke/2 + cut_idx)
	{
	    baseDCF[n] = abs(-1 + n * DCFstep);
	}
	else 
	{
	    baseDCF[n] = abs(-1 + (NsamplesPerSpoke/2-cut_idx) * DCFstep);
	}
    }

    
   
    for (int sp = 0 ; sp < Nspokes ; sp++)
    {
	for (int sa = 0 ; sa < NsamplesPerSpoke ; sa ++)
	{
	    if (Ndim == 2)
	    {
		DCF_out[sp * NsamplesPerSpoke + sa] = baseDCF[sa];
	    }
	    else 
	    {
		DCF_out[sp * NsamplesPerSpoke + sa] = baseDCF[sa] * baseDCF[sa];
	    }
	}
    }
    
    return DCF_out;
}

template <class T, unsigned int D> void LinearRadialTrajectory<T,D>::calculate_angles(int Nspokes, int correction_type)
{
    
    if ( D == 2 ){
	//Nothing yet...
    }
    else if ( D == 3 ){
	pol.create(Nspokes);                                        
	azi.create(Nspokes);
	
	T dH;
	T tmp;
	
	if ( correction_type == 0 || correction_type == 2 ){
	    for (int sp = 0; sp < Nspokes; sp++){
		dH = -1 + 2 * (T)sp / (T)Nspokes;
		pol[sp] = acos(dH);
		if ( sp == 0 )
		    azi[sp] = 0;
		else {
		    tmp = azi[sp-1] + 3.6 / sqrt(Nspokes*(1.0-dH*dH));
		    azi[sp] = tmp - floor(tmp/(2*M_PI)) * 2*M_PI;
		}
	    }
	}
	else if ( correction_type == 1 ){
	    for (int sp = 0; sp < Nspokes/2; sp++){
		dH = -1 + 2 * (T)sp / (T)(Nspokes/2);
		pol[sp*2]   = acos(dH);
		pol[sp*2+1] = acos(dH);
		if (sp == 0){
		    azi[2*sp]   = 0.0;
		    azi[2*sp+1] = 0.0;
		}
		else {
		    tmp = azi[sp*2-1] + 3.6 / sqrt(Nspokes/2*(1.0-dH*dH));
		    azi[2*sp]   = tmp - floor(tmp/(2*M_PI)) * 2*M_PI;
		azi[2*sp+1] = tmp - floor(tmp/(2*M_PI)) * 2*M_PI;
		}
	    }
	}
    }
}

//template class GATrajectory<float,1>;
template class LinearRadialTrajectory<float,2>;
template class LinearRadialTrajectory<float,3>;
//template class GATrajectory<float,4>;

//template class GATrajectory<double,1>;
template class LinearRadialTrajectory<double,2>;
template class LinearRadialTrajectory<double,3>;
//template class GATrajectory<double,4>;

