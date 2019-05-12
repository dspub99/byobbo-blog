#define BOOST_TEST_MODULE test_vUtil
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "vUtil.hpp"

BOOST_AUTO_TEST_CASE( test_mkVector )
{
  unsigned int n = 10;
  double *d = new double[n];
  
  auto v = vUtil::backedVector(d, n);
  
  d[3] = 3.14;
  BOOST_REQUIRE_EQUAL(d[3], v(3));
}

BOOST_AUTO_TEST_CASE( test_func )
{
  ublas::vector<double> x(10);
  vUtil::func(::tanh, x);
}

