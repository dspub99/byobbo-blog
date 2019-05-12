#ifndef __BBO__
#define __BBO__

#include <cassert>
#include <vector>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>


#include "minMax.hpp"
#include "vUtil.hpp"
#include "rnn.hpp"
#include "normalizer.hpp"
#include "rnnLayer.hpp"

namespace ublas = boost::numeric::ublas;

template<typename InternalLayer=RNNLayerPlain, typename OutThetaAlloc=ublas::unbounded_array<double>>
class BBO
{
  ublas::vector<double> w0_;
  unsigned int numThetas_, numQualities_, numParams_;
  Normalizer normalizer_;
  ublas::vector<double> x_;
  std::vector <ublas::vector<double>> normQualities_;
  RNN<InternalLayer, OutThetaAlloc> rnn_;
  
public:
  BBO(unsigned int numThetas, unsigned int numQualities, const std::vector<unsigned int> &nHidden);

  unsigned int numParams() const {return numParams_;}

  void setChronoInit(unsigned int tMax) {rnn_.setChronoInit(tMax);}
  
  void setParams(const std::vector<double> &params);

  const ublas::vector<double> & getW0() const {return w0_;}
  
  template<class A>
  void query(unsigned int nSteps, const std::vector<ublas::vector<double, A>> &thetas, const std::vector <ublas::vector<double, A>> &qualities, ublas::vector<double, OutThetaAlloc> &outThetas);

};


template<typename InternalLayer, typename OutThetaAlloc>
template<class A>
void BBO<InternalLayer, OutThetaAlloc>::query(unsigned int nSteps, const std::vector<ublas::vector<double, A>> &thetas, const std::vector <ublas::vector<double, A>> &qualities, ublas::vector<double, OutThetaAlloc> &outThetas)
{
  // initial guess                                                                                                                                                                                                                                                                 
  if (nSteps == 0)
    {
      for (unsigned int i=0; i<w0_.size(); ++i)
        {
          outThetas[i] = w0_[i];
        }
      return;
    }

  while (normQualities_.size() < qualities.size())
    {
      normQualities_.emplace_back(ublas::vector<double, A>(numQualities_));
    }

  assert(qualities.size() == normQualities_.size());
  normalizer_.normalize(nSteps, qualities, normQualities_);

  rnn_.reset();
  for (unsigned int i=0; i<nSteps; ++i)
    {
      const ublas::vector<double> &th = thetas[i];
      const ublas::vector<double> &q = normQualities_[i];

      assert(th.size() == numThetas_);
      assert(q.size() == numQualities_);

      vUtil::copyI(x_, 0, th);
      vUtil::copyI(x_, th.size(), q);

      rnn_.calc(x_, outThetas);
    }
}

template<typename InternalLayer, typename OutThetaAlloc>
BBO<InternalLayer, OutThetaAlloc>::BBO(unsigned int numThetas, unsigned int numQualities, const std::vector<unsigned int> &nHidden)
  :
  w0_(numThetas)
  ,numThetas_(numThetas), numQualities_(numQualities)
  ,normalizer_(Normalizer(numQualities)), x_(numThetas + numQualities)
  ,rnn_(numThetas + numQualities, numThetas, nHidden)

{
  numParams_ = w0_.size() + rnn_.numParams();
  w0_[0]=1;
  w0_[w0_.size()-1]=2;
}

template<typename InternalLayer, typename OutThetaAlloc>
void BBO<InternalLayer, OutThetaAlloc>::setParams(const std::vector<double> &params)
{
  std::copy(params.begin(), params.begin()+w0_.size(), w0_.begin());
  vUtil::func(vUtil::tanh, w0_); // output w's must be in our bounding box of [-1,1]^nDim
  rnn_.setParams(params, w0_.size());
}
  

#endif
