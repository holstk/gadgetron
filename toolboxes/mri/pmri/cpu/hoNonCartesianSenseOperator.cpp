#include "hoNonCartesianSenseOperator.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;


template<class REAL, unsigned int D> void
hoNonCartesianSenseOperator<REAL,D>::mult_M( hoNDArray< complext<REAL> >* in, hoNDArray< complext<REAL> >* out, bool accumulate )
{
    if( !in || !out ){
	throw std::runtime_error("hoNonCartesianSenseOperator::mult_M : 0x0 input/output not accepted");
    }
    if ( !in->dimensions_equal(&this->domain_dims_) || !out->dimensions_equal(&this->codomain_dims_)){
	throw std::runtime_error("hoNonCartesianSenseOperator::mult_H: input/output arrays do not match specified domain/codomains");
    }
       
    std::vector<size_t> full_dimensions = *this->get_domain_dimensions();
    full_dimensions.push_back(this->ncoils_);
    hoNDArray< complext<REAL> > tmp(&full_dimensions);  
    this->mult_csm( in, &tmp );
    //std::cout << "hoNonCartSenseOp: mult_M: mult_csm done" << std::endl;
    //std::cout << "hoNonCartSenseOp: mult_M: accumulate: " << accumulate << std::endl;
    
    // Forwards NFFT
    
    int NFFT_comp_mode = 1; //NFFT_FORWARDS_C2NC		   
    if( accumulate ){
	hoNDArray< complext<REAL> > tmp_out(out->get_dimensions());
	plan_->compute( &tmp, &tmp_out, dcw_.get(), NFFT_comp_mode );
	*out += tmp_out;
    }
    else
	plan_->compute( &tmp, out, dcw_.get(), NFFT_comp_mode );

    //std::cout << "hoNFFT: mult_MH_M: plan compute done" << std::endl;

}



template<class REAL, unsigned int D> void
hoNonCartesianSenseOperator<REAL,D>::mult_MH( hoNDArray< complext<REAL> >* in, hoNDArray< complext<REAL> >* out, bool accumulate )
{
    //std::cout << "hoNonCartSenseOp: mult_MH: beginning" << std::endl;
    if( !in || !out ){
	throw std::runtime_error("hoNonCartesianSenseOperator::mult_MH : 0x0 input/output not accepted");
    }
    
    if ( !in->dimensions_equal(&this->codomain_dims_) || !out->dimensions_equal(&this->domain_dims_)){
	throw std::runtime_error("hoNonCartesianSenseOperator::mult_MH: input/output arrays do not match specified domain/codomains");
    }
    std::vector<size_t> tmp_dimensions = *this->get_domain_dimensions();
    tmp_dimensions.push_back(this->ncoils_);
    hoNDArray< complext<REAL> > tmp(&tmp_dimensions);
    
    // Do the NFFT
    int NFFT_comp_mode = 8; //NFFT_BACKWARDS_NC2C		   
    plan_->compute( in, &tmp, dcw_.get(), NFFT_comp_mode );
    //std::cout << "hoNonCartSenseOp: mult_MH: plan->compute" << std::endl;
    
    if( !accumulate ){
	clear(out);    
    }
    
    this->mult_csm_conj_sum( &tmp, out );
    //std::cout << "hoNonCartSenseOp: mult_MH: mult_csm_conj_sum" << std::endl;
}

template<class REAL, unsigned int D> void
hoNonCartesianSenseOperator<REAL,D>::setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W )
{  
    plan_->setup( matrix_size, matrix_size_os, W );
}

template<class REAL, unsigned int D> void
hoNonCartesianSenseOperator<REAL,D>::preprocess( hoNDArray<_reald> *trajectory ) 
{
    if( trajectory == 0x0 ){
	throw std::runtime_error( "hoNonCartesianSenseOperator: cannot preprocess 0x0 trajectory.");
    }
    
    boost::shared_ptr< std::vector<size_t> > domain_dims = this->get_domain_dimensions();
    if( domain_dims.get() == 0x0 || domain_dims->size() == 0 ){
	throw std::runtime_error("hoNonCartesianSenseOperator::preprocess : operator domain dimensions not set");
    }
    int NFFT_prep_mode = 4; //NFFT_PREP_ALL		   
    plan_->preprocess( trajectory, NFFT_prep_mode );
    is_preprocessed_ = true;
}

template<class REAL, unsigned int D> void
hoNonCartesianSenseOperator<REAL,D>::set_dcw( boost::shared_ptr< hoNDArray<REAL> > dcw ) 
{
    dcw_ = dcw;  
}

//
// Instantiations
//

template class hoNonCartesianSenseOperator<float,1>;
template class hoNonCartesianSenseOperator<float,2>;
template class hoNonCartesianSenseOperator<float,3>;
template class hoNonCartesianSenseOperator<float,4>;

template class hoNonCartesianSenseOperator<double,1>;
template class hoNonCartesianSenseOperator<double,2>;
template class hoNonCartesianSenseOperator<double,3>;
template class hoNonCartesianSenseOperator<double,4>;
