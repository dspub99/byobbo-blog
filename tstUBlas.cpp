
#include <iostream>
#include <chrono>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>

namespace ublas = boost::numeric::ublas;

// g++ -O3 -DNDEBUG tstUBlas.cpp

std::chrono::microseconds timeProd(const ublas::matrix<double> &W,
				   const ublas::vector<double> &x,
				   const ublas::vector<double> &y0)
{
  ublas::vector<double> y(5);
  
  auto t0 = std::chrono::steady_clock::now();
  for (int i=0; i<1000; ++i)
    {
      // (noopts), w/o noalias, 1600 us
      // (noopts), noalias, 1500 us
      // -O3, noalias, 163 us
      // -O3 -DNDEBUG, noalias, 26 us
      // -O3 -DNDEBUG, axpy_prod(true), 231 us
      // -O3 -DNDEBUG, axpy_prod(false), 223 us

      
      ublas::noalias(y) = ublas::prod(W,x);
      //ublas::axpy_prod(W,x,y,true);
    }
  auto t1 = std::chrono::steady_clock::now();

  if (y.size() != y0.size())
    {
      throw std::string("wrong size ") + std::to_string(y.size()) + " " + std::to_string(y0.size());
    }
  for (unsigned int i=0; i<y0.size(); ++i)
    {
      if (y(i) != y0(i))
	{
	  throw std::string("wrong value ") + std::to_string(y(i)) + " " + std::to_string(y0(i));
	}
    }

  return std::chrono::duration_cast<std::chrono::microseconds>(t1-t0);
}

int main(int, char **)
{
  ublas::matrix<double> W(5,3);
  ublas::vector<double> x(3);

  x[0] = 1;
  x[1] = 2;
  x[2] = 3;

  for (unsigned int i=0; i<W.size1(); ++i)
    {
      for (unsigned int j=0; j<W.size2(); ++j)
	{
	  W(i, j) = i*j;
	}
    }

  unsigned int wSize = W.size1()*W.size2();
  for (unsigned int i=0; i<wSize; ++i)
    {
      std::cout << "flat " << i << " " << W.data()[i] << std::endl;
    }

  
  for (unsigned int i=0; i<W.size1(); ++i)
    {
      for (unsigned int j=0; j<W.size2(); ++j)
	{
	  std::cout << i << " " << j << ": " << W(i,j) << std::endl;
	}
    }


  ublas::vector<double> y(5);
  ublas::noalias(y) = ublas::prod(W,x);

  for (unsigned int i=0; i<y.size(); ++i)
    {
      std::cout << i << ": " << y(i) << std::endl;
    }
  std::cout << timeProd(W, x, y).count() << std::endl;
}
