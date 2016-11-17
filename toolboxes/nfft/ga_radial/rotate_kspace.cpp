#include "rotate_kspace.h"
#include "hoNFFT.h"

using namespace Gadgetron;


template <class T> void rotate_kspace<T>::rotate_trajectory(hoNDArray<typename reald<T,3>::Type> *trajectory, hoNDArray<T> rotation)
{

    //Make sure the sum of the rotations does not excede 1
    for (unsigned int ii = 0; ii < 3; ii++){
	T ref = rotation[ii*3+0]*rotation[ii*3+0] + rotation[ii*3+1]*rotation[ii*3+1] + rotation[ii*3+2]*rotation[ii*3+2];
	sqrt(ref);
	std::cout << "ref = " << ref << std::endl;
	std::cout << "rotation[ii*3+0] = " << rotation[ii*3+0] << std::endl;
	std::cout << "rotation[ii*3+1] = " << rotation[ii*3+1] << std::endl;
	std::cout << "rotation[ii*3+2] = " << rotation[ii*3+2] << std::endl;
	rotation[ii*3+0] /= ref;
	rotation[ii*3+1] /= ref;
	rotation[ii*3+2] /= ref;
	std::cout << "rotation[ii*3+0] = " << rotation[ii*3+0] << std::endl;
	std::cout << "rotation[ii*3+1] = " << rotation[ii*3+1] << std::endl;
	std::cout << "rotation[ii*3+2] = " << rotation[ii*3+2] << std::endl;
    }

    unsigned int elements = trajectory->get_number_of_elements();

    for (unsigned int ii = 0; ii < elements; ii++){

	typename reald<T,3>::Type old_coordinate = trajectory->get_data_ptr()[ii];

	typename reald<T,3> ::Type new_coordinate;
	
	for (unsigned int n = 0; n < 3; n++){
	    new_coordinate(n) = rotation[n*3+0]*old_coordinate(0) + rotation[n*3+1]*old_coordinate(1) + rotation[n*3+2]*old_coordinate(2);
	    if (new_coordinate(n) < -0.5)
		new_coordinate(n) = -0.5;
	    if (new_coordinate(n) > 0.5)
		new_coordinate(n) =  0.5;
	}

	trajectory->get_data_ptr()[ii] = new_coordinate;
	
    }
}

template class rotate_kspace<float>;
template class rotate_kspace<double>;
