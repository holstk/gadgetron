#pragma once

#include "Gadget.h"
#include "gadgetronpython_export.h"

#include <ismrmrd/ismrmrd.h>
#include <boost/python.hpp>

namespace Gadgetron{

  class EXPORTGADGETSPYTHON GadgetReference
  {

  public:
    GadgetReference();
    ~GadgetReference();

    int set_gadget(Gadget* g)
    {
      gadget_ = g;
      return 0;
    }

    template<class T> int return_data(T header, boost::python::object arr, const char* meta = 0);
    int return_acquisition(ISMRMRD::AcquisitionHeader acq, boost::python::object arr);
    int return_image(ISMRMRD::ImageHeader img, boost::python::object arr);
    int return_image_attr(ISMRMRD::ImageHeader img, boost::python::object arr, const char* meta);

  protected:
    Gadget* gadget_;
  };
}
