#ifndef __BPBBO__
#define __BPBBO__

#include <iostream>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "rnnLayer.hpp"
#include "commonStoreVector.hpp"
#include "bpUtil.hpp"
#include "bbo.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

template<typename InternalLayer>
class BPBBO
{
  unsigned int numThetas_, numQualities_;
  BBO<InternalLayer, CommonStoreVector::allocator_type> bbo_;
  std::vector<std::shared_ptr<CommonStoreVector>> inThetas_, inQualities_;
  std::shared_ptr<CommonStoreVector> outThetas_;
  std::vector<ublas::vector<double, ublas::shallow_array_adaptor<double>>> uInThetas_, uInQualities_;
  np::ndarray w0_;
  bool bW0Set_;
  
public:
  BPBBO(unsigned int numThetas, unsigned int numQualities, const bp::list &nHidden);

  void resize(unsigned int n);

  unsigned int size() const {return inThetas_.size();}
  
  unsigned int numParams() const {return bbo_.numParams();}

  void setChronoInit(unsigned int tMax) {bbo_.setChronoInit(tMax);}
  
  void setParams(const np::ndarray &);

  np::ndarray getW0() {if (!bW0Set_){bpUtil::copyUToArray(w0_, bbo_.getW0()); bW0Set_=true;}  return w0_;}
  
  np::ndarray getThetas(unsigned int i) {return inThetas_[i]->numpy();}

  np::ndarray getQualities(unsigned int i) {return inQualities_[i]->numpy();}

  np::ndarray getOutThetas() {return outThetas_->numpy();}

  void query(unsigned int nSteps);

  virtual ~BPBBO() {}

};


template<typename InternalLayer>
BPBBO<InternalLayer>::BPBBO(unsigned int numThetas, unsigned int numQualities, const bp::list &nHidden) :
  numThetas_(numThetas)
  , numQualities_(numQualities)
  , bbo_(BBO<InternalLayer, CommonStoreVector::allocator_type>(numThetas, numQualities, bpUtil::listToVec(nHidden)))
  , outThetas_(std::make_shared<CommonStoreVector>(numThetas))
  , w0_(np::zeros(bp::make_tuple(numThetas), np::dtype::get_builtin<double>()))
  , bW0Set_(false)
{
  // std::cout << "BP: " << numThetas << " " << numQualities << " " << bp::extract<unsigned int>(nHidden[0]) << std::endl;
}

template<typename InternalLayer>
void BPBBO<InternalLayer>::resize(unsigned int n)
{
  while (inThetas_.size() < n)
    {
      inThetas_.emplace_back(std::make_shared<CommonStoreVector>(numThetas_));
      uInThetas_.push_back(inThetas_[inThetas_.size()-1]->ublas());
      inQualities_.emplace_back(std::make_shared<CommonStoreVector>(numQualities_));
      uInQualities_.push_back(inQualities_[inQualities_.size()-1]->ublas());
    }
}

template<typename InternalLayer>
void BPBBO<InternalLayer>::setParams(const np::ndarray &params)
{
  bbo_.setParams(bpUtil::arrayToVector(params));
  bW0Set_ = false;
  //bpUtil::copyUToArray(w0_, bbo_.getW0());
}

template<typename InternalLayer>
void BPBBO<InternalLayer>::query(unsigned int nSteps)
{
  // std::cout << "Q: " << bp::extract<double>(inThetas_[0]->numpy()[1]) << std::endl;
  bbo_.query(nSteps, uInThetas_, uInQualities_, outThetas_->ublas());
}

#endif
