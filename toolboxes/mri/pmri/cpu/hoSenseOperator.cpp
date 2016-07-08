#include "hoSenseOperator.h"
//#include "sense_utilities.h"
#include "vector_td_utilities.h"

namespace Gadgetron{
    
    template<class REAL, unsigned int D> void
    hoSenseOperator<REAL,D>::mult_csm( hoNDArray<complext<REAL> >* in, hoNDArray<complext<REAL> >* out )
    {  
	if( in->get_number_of_dimensions() < D  || in->get_number_of_dimensions() > D+1 ){
	    throw std::runtime_error("mult_csm: unexpected input dimensionality");
	}
	
	if( in->get_number_of_dimensions() > out->get_number_of_dimensions() ){
	    throw std::runtime_error("mult_csm: input dimensionality cannot exceed output dimensionality");
	}
	
	hoNDArray<complext< REAL > > *csm = this->csm_.get();
	
	if( csm->get_number_of_dimensions() != D+1 ) {
	  throw std::runtime_error("mult_csm: input dimensionality of csm not as expected");
	}
	
	unsigned int num_coils = csm->get_size(D);
	
	unsigned int num_image_elements = 1;
	for( unsigned int d=0; d<D; d++ )
	    num_image_elements *= in->get_size(d);
	
	unsigned int num_frames = in->get_number_of_elements() / num_image_elements;

	for (unsigned int i = 0; i < num_image_elements; i++){
	    for (unsigned int f = 0; f < num_frames; f++){
		complext<REAL> _in = in->get_data_ptr()[i+f*num_image_elements];
		for( unsigned int c = 0; c < num_coils; c++) {
		    out->get_data_ptr()[i + f*num_image_elements + c*num_image_elements*num_frames] =  _in * csm->get_data_ptr()[i+c*num_image_elements];
		}
	    }
	}
    }
    
    
    template<class REAL, unsigned int D> void
    hoSenseOperator<REAL,D>::mult_csm_conj_sum( hoNDArray<complext<REAL> > *in, hoNDArray<complext<REAL> > *out )
    {
	//std::cout << "hoSenseOp: mult_csm_conj_sum: beginning" << std::endl;

	if( out->get_number_of_dimensions() < D  || out->get_number_of_dimensions() > D+1 ){
	    throw std::runtime_error("mult_csm_conj_sum: unexpected output dimensionality");
	}

	if( out->get_number_of_dimensions() > in->get_number_of_dimensions() ){
	    throw std::runtime_error("mult_csm_conj_sum: output dimensionality cannot exceed input dimensionality");
	}

	hoNDArray<complext< REAL > > *csm = this->csm_.get();
	//std::cout << "hoSenseOp: mult_csm_conj_sum: csm_.get" << std::endl;

	if( csm->get_number_of_dimensions() != D+1 ) {
	    throw std::runtime_error("mult_csm_conj_sum: input dimensionality of csm not as expected");
	}
	
	unsigned int num_coils = csm->get_size(D);
	unsigned int num_image_elements = 1;
	for( unsigned int d=0; d<D; d++ )
	    num_image_elements *= out->get_size(d);
	
	unsigned int num_frames = out->get_number_of_elements() / num_image_elements;
	//std::cout << "hoSenseOp: mult_csm_conj_sum: num elem and num frames" << std::endl;
	//std::cout << "hoSenseOp: mult_csm_conj_sum: num coils, elem, frames: " << num_coils << ", " << num_image_elements << ", " << num_frames << std::endl;
	//std::cout << "hoSenseOp: mult_csm_conj_sum: out elems, dims: " << out->get_number_of_elements() << ", " << out->get_number_of_dimensions() << std::endl;
	//std::cout << "hoSenseOp: mult_csm_conj_sum: out dims: " << out->get_size(0) << ", " << out->get_size(1) << ", " << out->get_size(2) << std::endl;

	for (unsigned int i = 0; i < num_image_elements; i++){
	    for (unsigned int f = 0; f < num_frames; f++){
		complext<REAL> _out = complext<REAL>(0);
		for( unsigned int c = 0; c < num_coils; c++ ) {
		    _out += in->get_data_ptr()[i+f*num_image_elements+c*num_frames*num_image_elements] * conj(csm->get_data_ptr()[i+c*num_image_elements]);
		}
		out->get_data_ptr()[i+f*num_image_elements] = _out;
	    }
	}
	//std::cout << "hoSenseOp: mult_csm_conj_sum: end" << std::endl;
    }
    
    
    //
    // Instantiations
    //
    
    template class hoSenseOperator<float,1>;
    template class hoSenseOperator<float,2>;
    template class hoSenseOperator<float,3>;
    template class hoSenseOperator<float,4>;
    
    template class hoSenseOperator<double,1>;
    template class hoSenseOperator<double,2>;
    template class hoSenseOperator<double,3>;
    template class hoSenseOperator<double,4>;
}
