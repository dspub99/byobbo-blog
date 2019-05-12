#define BOOST_TEST_MODULE test_minnmax
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;

#include "minMax.hpp"

BOOST_AUTO_TEST_CASE( test_minmax )
{
  MinMax mm;
  mm.update(-1);
  mm.update(-2);
  mm.update(0);
  mm.update(37);
  mm.update(.1);

  BOOST_REQUIRE_EQUAL(mm.min(), -2);
  BOOST_REQUIRE_EQUAL(mm.max(), 37);

}
