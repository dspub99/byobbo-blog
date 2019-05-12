#define BOOST_TEST_MODULE test_bbo
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <random>

#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;

#include "bbo.hpp"
#include "vUtil.hpp"


BOOST_AUTO_TEST_CASE( test_construct )
{
  unsigned int nTh = 3, nQ = 2;
  std::vector<unsigned int> nh{4,5};
  
  BBO<> bbo(nTh, nQ, nh);

  std::vector<ublas::vector<double>> thetas;
  std::vector <ublas::vector<double>> qualities;
  ublas::vector<double> outThetas(nTh);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<double> normal_dist(0, 1);
  unsigned int nParams = bbo.numParams();
  std::vector<double> params(nParams);
  for (unsigned int i=0; i<nParams; ++i)
    {
      params[i] = normal_dist(rng);
    }
  bbo.setParams(params);
  
  ublas::vector<double> q(nQ);
  bbo.query(thetas.size(), thetas, qualities, outThetas);

  vUtil::randNorm01(outThetas);
  thetas.push_back(outThetas);
  vUtil::randNorm01(q);
  qualities.push_back(q);

  bbo.query(thetas.size(), thetas, qualities, outThetas);

  vUtil::randNorm01(outThetas);
  thetas.push_back(outThetas);
  vUtil::randNorm01(q);
  qualities.push_back(q);
  bbo.query(thetas.size(), thetas, qualities, outThetas);
  double x = outThetas[1];
  bbo.query(thetas.size(), thetas, qualities, outThetas);
  BOOST_REQUIRE_EQUAL(x, outThetas[1]);
}
