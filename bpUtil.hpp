#ifndef __BPUTIL__
#define __BPUTIL__

#include <iostream>
#include <vector>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace ublas = boost::numeric::ublas;
namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace bpUtil
{
 
  auto backedArray(double *data, unsigned int size)
  {
    return np::from_data(data, np::dtype::get_builtin<double>(),
			 bp::make_tuple(size),
			 bp::make_tuple(sizeof(double)),
			 bp::object());
  }
  
  std::vector<unsigned int> listToVec(const bp::list &nHidden)
  {
    unsigned int n = bp::len(nHidden);
    std::vector<unsigned int> vNHidden(n);
    for (unsigned int i=0; i<n; ++i)
      {
	vNHidden[i] = bp::extract<unsigned int>(nHidden[i]);
      }
    return vNHidden;
  }

  std::vector<double> arrayToVector(const np::ndarray &a)
  {
    unsigned int n = a.shape(0);
    std::vector<double> v(n);
    for (unsigned int i=0; i<n; ++i)
      {
	v[i] = bp::extract<double>(a[i]);
      }
    return v;
  }

  void copyUToArray(np::ndarray &a, const ublas::vector<double> &u)
  {
    for (unsigned int i=0; i<u.size(); ++i)
      {
	a[i] = u(i);
      }
  }
  
  
  
}

#endif
