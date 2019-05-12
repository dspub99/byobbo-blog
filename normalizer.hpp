#ifndef __NORMALIZER__
#define __NORMALIZER__

#include <cassert>
#include <vector>

#include <boost/numeric/ublas/vector.hpp>

#include "minMax.hpp"
#include "vUtil.hpp"

namespace ublas = boost::numeric::ublas;

class Normalizer
{
  unsigned int numQualities_;
  ublas::vector<double> qOnes_, dq_;
  
public:
  Normalizer(unsigned int numQualities);
  
  template<class A1, class A2>
  void normalize(unsigned int nSteps, const std::vector <ublas::vector<double, A1>> &qualities, std::vector <ublas::vector<double, A2>> &outQualities);
};


template<class A1, class A2>
void Normalizer::normalize(unsigned int nSteps, const std::vector <ublas::vector<double, A1>> &qualities, std::vector <ublas::vector<double, A2>> &outQualities)
{
  assert(qualities.size() == outQualities.size());
  assert(qualities.size() >= nSteps);
  
  if (nSteps == 0)
    {
      return;
    }

  if (nSteps == 1)
    {
      assert(outQualities[0].size() == qOnes_.size());
      vUtil::copy(outQualities[0], qOnes_);
      return;
    }

  std::vector<MinMax> minMax(numQualities_);

  for (unsigned int i=0; i<nSteps; ++i)
    {
      const auto &q = qualities[i];
      for (unsigned int j=0; j<numQualities_; ++j)
	{
	  minMax[j].update(q(j));
	}
    }

  for (unsigned int j=0; j<numQualities_; ++j)
    {
      MinMax &mm = minMax[j];
      dq_(j) = mm.max() - mm.min();
    }
  
  for (unsigned int i=0; i<nSteps; ++i)
    {
      const ublas::vector<double> &q = qualities[i];
      ublas::vector<double, A2> &oq = outQualities[i];
      
      for (unsigned int j=0; j<numQualities_; ++j)
	{
	  const MinMax &mm = minMax[j];
	  const double &ddq = dq_(j);
	  if (ddq == 0)
	    {
	      oq(j) = 0;
	    }
	  else
	    {
	      oq(j) = 2* ((q(j) - mm.min()) / ddq) - 1;
	    }
	}
    }

}


#endif
