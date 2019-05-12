#define BOOST_TEST_MODULE test_bpUtil
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <random>

#include "bpUtil.hpp"

BOOST_AUTO_TEST_CASE( test_mkarray )
{
  Py_Initialize();
  np::initialize();

  unsigned int n = 7;
  double *d = new double[n];
  auto a = bpUtil::backedArray(d, n);

  d[3] = 3.14;
  BOOST_REQUIRE_EQUAL(d[3], bp::extract<double>(a[3]));

}
