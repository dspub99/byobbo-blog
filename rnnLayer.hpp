#ifndef __RNNLAYER__
#define __RNNLAYER__

#include <vector>
#include <cmath>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/operation.hpp> // axpy

#include "vUtil.hpp"

namespace ublas = boost::numeric::ublas;


class RNNLayer
{
public:
  virtual void reset() = 0;
  virtual unsigned int numParams() = 0;
  virtual void setChronoInit(unsigned int tMax) = 0;
  virtual unsigned int setParams(const std::vector<double> &params, unsigned int i0) = 0;
};


class RNNLayerPlain : public RNNLayer
{
  ublas::matrix<double> WSelf_, WPrev_;
  ublas::vector<double> W0_, h_, htmp1_, htmp2_;
  unsigned int nSelf_, nPrev_;
  double cIn_;
  
public:
  RNNLayerPlain(unsigned int nOut, unsigned int nIn) :
    WSelf_(nOut, nOut), WPrev_(nOut, nIn),
    W0_(nOut), h_(nOut), htmp1_(nOut), htmp2_(nOut),
    nSelf_(WSelf_.size1()*WSelf_.size2()),
    nPrev_(WPrev_.size1()*WPrev_.size2()),
    cIn_(1/sqrt(nIn))
    
  {}
  

  void reset() {h_.clear();}
  unsigned int numParams() { return nSelf_ + nPrev_ + W0_.size(); }

  const ublas::vector<double> &activation() const { return h_; }

  void setChronoInit(unsigned int) {}
  
  unsigned int setParams(const std::vector<double> &params, unsigned int i0)
  {
    i0 = vUtil::copyData(WSelf_.data(), params, i0, nSelf_);
    i0 = vUtil::copyData(WPrev_.data(), params, i0, nPrev_);
    i0 = vUtil::copyData(W0_.data(), params, i0, W0_.size());
    //i0 = vUtil::copyParams(WPrev_.data(), params, i0, nPrev_, cIn_);
    //i0 = vUtil::copyParams(W0_.data(), params, i0, W0_.size(), cIn_);

    return i0;
  }

  template<class A1>
  void update(const ublas::vector<double, A1> &x)
  {
    ublas::noalias(htmp1_) = ublas::prod(WSelf_, h_);
    ublas::noalias(htmp2_) = ublas::prod(WPrev_, x);
    noalias(h_) = W0_ + htmp1_ + htmp2_;
    
    vUtil::func(vUtil::tanh, h_);
  }
};


class RNNLayerSimple : public RNNLayer
{
  ublas::matrix<double> WPrev_;
  ublas::vector<double> W0_, WSelf_, h_, htmp1_, htmp2_;
  unsigned int nSelf_, nPrev_;
  double cIn_;
  
public:
  RNNLayerSimple(unsigned int nOut, unsigned int nIn) :
    WPrev_(nOut, nIn),
    W0_(nOut), WSelf_(nOut),
    h_(nOut), htmp1_(nOut), htmp2_(nOut),
    nSelf_(WSelf_.size()),
    nPrev_(WPrev_.size1()*WPrev_.size2()),
    cIn_(1/sqrt(nIn))
    
  {
    /*
    double norm = nOut-1;
    for (unsigned int i=0; i<nOut; ++i)
      {
	WSelf_[i] = i / norm;
      }
    */
  }
  

  void reset() {h_.clear();}
  unsigned int numParams() { return nPrev_ + W0_.size() + nSelf_; }

  const ublas::vector<double> &activation() const { return h_; }

  void setChronoInit(unsigned int) {}
  
  unsigned int setParams(const std::vector<double> &params, unsigned int i0)
  {
    i0 = vUtil::copyData(WSelf_.data(), params, i0, nSelf_);
    i0 = vUtil::copyData(WPrev_.data(), params, i0, nPrev_);
    i0 = vUtil::copyData(W0_.data(), params, i0, W0_.size());

    return i0;
  }

  template<class A1>
  void update(const ublas::vector<double, A1> &x)
  {
    ublas::noalias(htmp1_) = ublas::element_prod(WSelf_, h_);
    ublas::noalias(htmp2_) = ublas::prod(WPrev_, x);
    noalias(h_) = W0_ + htmp1_ + htmp2_;
    
    vUtil::func(vUtil::tanh, h_);
  }
};




class RNNLayerRes : public RNNLayer
{
  ublas::matrix<double> WSelf_, WPrev_, WPrev2_;
  ublas::vector<double> W0_, h_, a_, htmp1_, htmp2_;
  unsigned int nSelf_, nPrev_;
  
  
public:
  RNNLayerRes(unsigned int nOut, unsigned int nIn) :
    WSelf_(nOut, nOut), WPrev_(nOut, nIn), WPrev2_(nOut, nIn),
    W0_(nOut), h_(nOut), a_(nOut),
    htmp1_(nOut), htmp2_(nOut),
    nSelf_(WSelf_.size1()*WSelf_.size2()),
    nPrev_(WPrev_.size1()*WPrev_.size2())

    
  {}
  

  void reset() {h_.clear();}
  unsigned int numParams() { return nSelf_ + 2*nPrev_ + W0_.size(); }

  const ublas::vector<double> &activation() const { return a_; }

  void setChronoInit(unsigned int) {}
  
  unsigned int setParams(const std::vector<double> &params, unsigned int i0)
  {
    i0 = vUtil::copyData(WSelf_.data(), params, i0, nSelf_);
    i0 = vUtil::copyData(WPrev_.data(), params, i0, nPrev_);
    i0 = vUtil::copyData(WPrev2_.data(), params, i0, nPrev_);
    i0 = vUtil::copyData(W0_.data(), params, i0, W0_.size());

    return i0;
  }

  template<class A1>
  void update(const ublas::vector<double, A1> &x)
  {
    ublas::noalias(htmp1_) = ublas::prod(WSelf_, h_);
    ublas::noalias(htmp2_) = ublas::prod(WPrev_, x);
    noalias(h_) = W0_ + htmp1_ + htmp2_;
    vUtil::func(vUtil::tanh, h_);
    
    ublas::noalias(htmp2_) = ublas::prod(WPrev2_, x);
    noalias(a_) = h_ + htmp2_;
  }
};


class RNNLayerJANET : public RNNLayer
{
  ublas::matrix<double> Wsh_, Wsx_;
  ublas::matrix<double> Wch_, Wcx_;
  ublas::vector<double> Ws0_, Wc0_, beta_;
  ublas::vector<double> ones_, h_, s_, sb_, ctilde_;
  ublas::vector<double> vTmp1_, vTmp2_;
  unsigned int nWh_, nWx_, nv_, nParams_, nOut_;
  bool bChrono_;
  unsigned int tMax_;

public:
  RNNLayerJANET(unsigned int nOut, unsigned int nIn) :
    Wsh_(nOut, nOut), Wsx_(nOut, nIn),
    Wch_(nOut, nOut), Wcx_(nOut, nIn),
    Ws0_(nOut), Wc0_(nOut), beta_(nOut),
    ones_(nOut), h_(nOut), s_(nOut), sb_(nOut), ctilde_(nOut),
    vTmp1_(nOut), vTmp2_(nOut),
    nWh_(nOut*nOut), nWx_(nOut*nIn), nv_(nOut),
    nOut_(nOut),
    bChrono_(false),
    tMax_(0)
    
  {
    for (unsigned int i=0; i<nOut; ++i)
      {
	ones_[i] = 1;
      }
    nParams_ = 2 * nWh_ + 2*nWx_
      + 1 // beta
      + 2 * nv_ // W0's
      + 4 * nv_; // state & calc vectors
  }
  
  void reset() {h_.clear();}
  unsigned int numParams() { return nParams_; }

  const ublas::vector<double> &activation() const { return h_; }

  void setChronoInit(unsigned int tMax)
  {
    if (tMax < 2)
      {
	bChrono_ = false;
	tMax_ = 1;
      }
    else
      {
	bChrono_ = true;
	tMax_ = tMax-2;
      }
  }

  unsigned int setParams(const std::vector<double> &params, unsigned int i0)
  {
    i0 = vUtil::copyData(Wsh_.data(), params, i0, nWh_);
    i0 = vUtil::copyData(Wsx_.data(), params, i0, nWx_);
    i0 = vUtil::copyData(Wch_.data(), params, i0, nWh_);
    i0 = vUtil::copyData(Wcx_.data(), params, i0, nWx_);
    beta_[0] = params[i0++];
    for (unsigned int i=1; i<nOut_; ++i)
      {
	beta_[i] = beta_[0];
      }
    i0 = vUtil::copyData(Ws0_.data(), params, i0, nv_);
    i0 = vUtil::copyData(Wc0_.data(), params, i0, nv_);

    
    if (bChrono_)
      {
	for (unsigned int i=0; i<Wc0_.size(); ++i)
	  {
	    Wc0_[i] = ::log(1 + tMax_*::tanh(::abs(Wc0_[i])));
	  }
      }
    
    return i0;
  }

  template<class A1>
  void update(const ublas::vector<double, A1> &x)
  {
    ublas::noalias(vTmp1_) = ublas::prod(Wsh_, h_);
    ublas::noalias(vTmp2_) = ublas::prod(Wsx_, x);
    noalias(s_) = Ws0_ + vTmp1_ + vTmp2_;
    noalias(sb_) = s_ - beta_;

    vUtil::func(vUtil::logistic, s_);
    vUtil::func(vUtil::logistic, sb_);
    
    ublas::noalias(vTmp1_) = ublas::prod(Wch_, h_);
    ublas::noalias(vTmp2_) = ublas::prod(Wcx_, x);
    noalias(ctilde_) = Wc0_ + vTmp1_ + vTmp2_;
    vUtil::func(vUtil::tanh, ctilde_);
    
    for (unsigned int i=0; i<s_.size(); ++i)
      {
	s_[i] *= h_[i];
      }
    
    for (unsigned int i=0; i<sb_.size(); ++i)
      {
	sb_[i] = (1 - sb_[i])*ctilde_[i];
      }
    noalias(h_) = s_ + sb_;
    
    vUtil::func(vUtil::tanh, h_);
    
  }
};


class RNNLayerFinal : public RNNLayer
{
  ublas::matrix<double> WPrev_;
  ublas::vector<double> W0_, htmp2_;
  unsigned int nPrev_;

public:
  RNNLayerFinal(unsigned int nOut, unsigned int nIn) :
    WPrev_(nOut, nIn),
    W0_(nOut), htmp2_(nOut),
    nPrev_(WPrev_.size1()*WPrev_.size2())
  {}
  
  void reset() {}
  unsigned int numParams() { return nPrev_ + W0_.size(); }

  void setChronoInit(unsigned int) {}

  
  unsigned int setParams(const std::vector<double> &params, unsigned int i0)
  {
    i0 = vUtil::copyData(WPrev_.data(), params, i0, nPrev_);
    return vUtil::copyData(W0_.data(), params, i0, W0_.size());
  }

  template<class A1, class A2>
  void update(const ublas::vector<double, A1> &x, ublas::vector<double, A2> &yOut)
  {
    ublas::noalias(htmp2_) = ublas::prod(WPrev_, x);
    noalias(yOut) = W0_ + htmp2_;
    vUtil::func(vUtil::tanh, yOut);
  }
};




#endif

