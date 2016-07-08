#include "hoNFFTOperator.h"
//#include "hoNDArray_operators.h"
#include "hoNDArray_elemwise.h"
//#include "hoNDArray_blas.h"

namespace Gadgetron{
    
    template<class T, unsigned int D> void hoNFFTOperator<T,D>::mult_M( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate )
    {
	if( !in || !out ){
	    throw std::runtime_error("hoNFFTOperator::mult_M : 0x0 input/output not accepted");
	}
	
	hoNDArray<complext<T> > *tmp_out;

	if( accumulate ){
	    tmp_out = new hoNDArray<complext<T> >(out->get_dimensions());
	}
	else{
	    tmp_out = out;
	}
	
	//plan_->compute( in, tmp_out, dcw_.get(), hoNFFT_plan<T,D>::NFFT_FORWARDS_C2NC );
	plan_->compute( in, tmp_out, dcw_.get(), 1 ); //NFFT_FORWARDS_C2NC

	if( accumulate ){
	    *out += *tmp_out;
	    delete tmp_out;
	}
    }

    
    template<class T, unsigned int D> void hoNFFTOperator<T,D>::mult_MH( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate )
    {
	if( !in || !out ){
	    throw std::runtime_error("hoNFFTOperator::mult_MH : 0x0 input/output not accepted");
	}
	
	hoNDArray<complext<T> > *tmp_out;
	
	if( accumulate ){
	    tmp_out = new hoNDArray<complext<T> >(out->get_dimensions());
	}
	else{
	    tmp_out = out;
	}
	
	//plan_->compute( in, tmp_out, dcw_.get(), hoNFFT_plan<T,D>::NFFT_BACKWARDS_NC2C );
	plan_->compute( in, tmp_out, dcw_.get(), 8 ); //NFFT_BACKWARDS_NC2C
	if( accumulate ){
	    *out += *tmp_out;
	    delete tmp_out;
	}
    }
    
    template<class T, unsigned int D> void hoNFFTOperator<T,D>::mult_MH_M( hoNDArray<complext<T> > *in, hoNDArray<complext<T> > *out, bool accumulate )
    {
	if( !in || !out ){
	    throw std::runtime_error("hoNFFTOperator::mult_MH_M : 0x0 input/output not accepted");
	}
    
	boost::shared_ptr< std::vector<size_t> > codomain_dims = this->get_codomain_dimensions();
	if( codomain_dims.get() == 0x0 || codomain_dims->size() == 0 ){
	    throw std::runtime_error("hoNFFTOperator::mult_MH_M : operator codomain dimensions not set");
	}
	
	hoNDArray<complext<T> > *tmp_out;
	
	if( accumulate ){
	    tmp_out = new hoNDArray<complext<T> >(out->get_dimensions());
	}
	else{
	    tmp_out = out;
	}
	
	plan_->mult_MH_M( in, tmp_out, dcw_.get(), *codomain_dims );
	
	if( accumulate ){
	    *out += *tmp_out;
	    delete tmp_out;
	} 
    }
    
  template<class T, unsigned int D> void hoNFFTOperator<T,D>::setup( typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os, T W )
  {  
      plan_->setup( matrix_size, matrix_size_os, W );  
  }
    
    template<class T, unsigned int D> void hoNFFTOperator<T,D>::preprocess( hoNDArray<typename reald<T,D>::Type> *trajectory )
    {
	if( trajectory == 0x0 ){
	    throw std::runtime_error("hoNFFTOperator::preprocess : 0x0 trajectory provided.");
	}
	
	//plan_->preprocess( trajectory, hoNFFT_plan<T,D>::NFFT_PREP_ALL );
	plan_->preprocess( trajectory, 3 ); //NFFT_PREP_ALL
        }
    
    
  //
  // Instantiations
  //

  template class hoNFFTOperator<float,1>;
  template class hoNFFTOperator<float,2>;
  template class hoNFFTOperator<float,3>;
  template class hoNFFTOperator<float,4>;

  template class hoNFFTOperator<double,1>;
  template class hoNFFTOperator<double,2>;
  template class hoNFFTOperator<double,3>;
  template class hoNFFTOperator<double,4>;
}
