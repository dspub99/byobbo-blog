#ifndef __CSVECTOR__
#define __CSVECTOR__

#include <cstring>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "vUtil.hpp"
#include "bpUtil.hpp"

namespace ublas = boost::numeric::ublas;
namespace bp = boost::python;
namespace np = boost::python::numpy;

class CommonStoreVector
{
  double *data_;
  ublas::vector<double, ublas::shallow_array_adaptor<double>> vBlas_;
  np::ndarray ndArray_;
  
public:
  typedef ublas::shallow_array_adaptor<double> allocator_type;
  
  CommonStoreVector(unsigned int size) : 
					 data_(new double[size]),
					 vBlas_(vUtil::backedVector(data_, size)),
					 ndArray_(bpUtil::backedArray(data_, size))
  {}

  auto &ublas() {return vBlas_;}
  
  auto &numpy() {return ndArray_;}

  virtual ~CommonStoreVector() { delete data_; }

  
private:
  CommonStoreVector(const CommonStoreVector &) :
    data_(nullptr),
    ndArray_(np::array(bp::list()))
  {
    throw std::string("no copying");
  }

  CommonStoreVector(CommonStoreVector&&) : data_(nullptr), ndArray_(np::array(bp::list())) { throw std::string("no movinng"); }
  
  CommonStoreVector& operator=(const CommonStoreVector&) { throw std::string("no assignment"); }

  
  
  
};

#endif
