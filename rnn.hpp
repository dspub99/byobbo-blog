#ifndef __RNN__
#define __RNN__


#include <vector>
#include <memory>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>


#include "rnnLayer.hpp"
#include "vUtil.hpp"


namespace ublas = boost::numeric::ublas;

class RNNLayer;

template<typename InternalLayer=RNNLayerPlain, typename YOutAlloc=ublas::unbounded_array<double>, typename XAlloc=ublas::unbounded_array<double>>
class RNN
{  
  std::vector<RNNLayer *> layers_;
  unsigned int nParams_, maxM1_;
  
public:
  RNN(unsigned int nx, unsigned int ny, const std::vector<unsigned int> &nHidden);

  unsigned int numParams() const {return nParams_;}

  void setChronoInit(unsigned int tMax);
  
  unsigned int setParams(const std::vector<double> &params, unsigned int i0);
  
  void reset();

  void calc(const ublas::vector<double, XAlloc> &x, ublas::vector<double, YOutAlloc> &yOut);

  virtual ~RNN();

};



////

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
void RNN<InternalLayer, YOutAlloc, XAlloc>::calc(const ublas::vector<double, XAlloc> &x, ublas::vector<double, YOutAlloc> &yOut)
{
  const ublas::vector<double, XAlloc> *a = &x;

  for (unsigned int i=0; i<maxM1_; ++i)
    {
      InternalLayer* ly = (InternalLayer*)layers_[i];
      ly->update(*a);
      a = &ly->activation();
    }

  RNNLayerFinal* ly = (RNNLayerFinal*) layers_[maxM1_];
  ly->update(*a, yOut);
}

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
void RNN<InternalLayer, YOutAlloc, XAlloc>::setChronoInit(unsigned int tMax)
{
  for (unsigned int i=0; i<maxM1_; ++i)
    {
      InternalLayer* ly = (InternalLayer*)layers_[i];
      ly->setChronoInit(tMax);
    }
}

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
RNN<InternalLayer, YOutAlloc, XAlloc>::RNN(unsigned int nx, unsigned int ny, const std::vector<unsigned int> &nHidden)
{
  unsigned int nPrev = nx;
  for (auto nh : nHidden)
    {
      layers_.push_back(new InternalLayer(nh, nPrev));
      nPrev = nh;
    }
  layers_.push_back(new RNNLayerFinal(ny, nPrev));

  
  unsigned int nParams = 0;
  for (auto ly : layers_)
    {
      nParams += ly->numParams();
    }

  nParams_ = nParams;
  maxM1_ = layers_.size()-1;
}

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
unsigned int RNN<InternalLayer, YOutAlloc, XAlloc>::setParams(const std::vector<double> &params, unsigned int i0)
{
  if ((params.size() - i0) != nParams_)
    {
      throw std::string("Wrong number of parameters ") + std::to_string(params.size() - i0) + std::string(" != numParams() = ") + std::to_string(nParams_);
    }

  unsigned int i = i0;
  for (auto ly : layers_)
    {
      i = ly->setParams(params, i);
    }
  return i;
}

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
void RNN<InternalLayer, YOutAlloc, XAlloc>::reset()
{
  for (auto ly : layers_)
    {
      ly->reset();
    }
}

template<typename InternalLayer, typename YOutAlloc, typename XAlloc>
RNN<InternalLayer, YOutAlloc, XAlloc>::~RNN()
{

}

#endif
