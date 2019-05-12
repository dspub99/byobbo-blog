#ifndef __VUTIL__
#define __VUTIL__

#include <memory>
#include <iostream>
#include <random>

#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;


template<class A>
inline std::ostream& operator<<(std::ostream &os, const ublas::vector<double, A> &x)
{
  for (unsigned int i=0; i<x.size(); ++i)
    {
      os << x(i) << " ";
    }
  return os;
}
  


namespace vUtil
{
  template<class T>
  unsigned int copyData(T &dst, const std::vector<double> &src, unsigned int i0, unsigned int len)
  {
    for (unsigned int i=0; i<len; ++i)
      {
	dst[i] = src[i0++];
      }
    return i0;
  }

  template<class T>
  unsigned int copyParams(T &dst, const std::vector<double> &src, unsigned int i0, unsigned int len, double c)
  {
    for (unsigned int i=0; i<len; ++i)
      {
	dst[i] = c*src[i0++];
      }
    return i0;
  }

  
  inline auto backedVector(double *data, unsigned int size)
  {
    auto shared = std::make_shared<ublas::shallow_array_adaptor<double>>(size, data);
    return ublas::vector<double, ublas::shallow_array_adaptor<double> >(size, *shared);
  }

  template<class A>
  inline void randNorm01(ublas::vector<double, A> &v)
  {
     std::random_device dev;
     std::mt19937 rng(dev());
     std::normal_distribution<double> normal_dist(0, 1);
     for (unsigned int i=0; i<v.size(); ++i)
       {
	 v(i) = normal_dist(rng);
       }
  }

  template<class A1, class A2>
  inline void copy(ublas::vector<double, A1> &dst, const ublas::vector<double, A2> &src)
  {
    for (unsigned int i=0; i<dst.size(); ++i)
      {
	dst(i) = src(i);
      }
  }

  template<class A1, class A2>
  inline void copyI(ublas::vector<double, A1> &dst, unsigned int id0, const ublas::vector<double, A2> &src)
  {
    for (unsigned int i=0; i<src.size(); ++i, id0++)
      {
	dst(id0) = src(i);
      }
  }

  template<class A, class F>
  inline void func(F f, ublas::vector<double, A> &x)
  {
    // TODO: figure out how to do SSE
    unsigned int iMax = x.size();
    for (unsigned int i=0; i<iMax; ++i)
      {
	x[i] = f(x[i]);
      }    
  }

  inline double tanh(double x)
  {
    return ::tanh(x);
  }

  inline double logistic(double x)
  {
    return 1 / (1 + ::exp(-x));
  }
    

  inline double relu(double x)
  {
    if (x<0)
      {
	return 0;
      }
    return x;
  }
  
}


#endif
