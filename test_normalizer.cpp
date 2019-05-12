#define BOOST_TEST_MODULE test_normalizer
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;

#include "normalizer.hpp"

BOOST_AUTO_TEST_CASE( test_normalizer )
{
  unsigned int nq = 7;
  Normalizer norm(nq);

  std::vector <ublas::vector<double>> qualities;
  std::vector <ublas::vector<double>> outQualities;

  std::cout << "nSteps = " << qualities.size() << std::endl;
  norm.normalize(qualities.size(), qualities, outQualities);
  BOOST_REQUIRE_EQUAL(outQualities.size(), 0);

  ublas::vector<double> q(nq);
  for (unsigned int i=0; i<q.size(); ++i)
    {
      q(i)=0;
    }

  for (unsigned int i=1; i<10; ++i)
    {
      qualities.push_back(q);
      outQualities.push_back(ublas::vector<double>(nq));
      norm.normalize(qualities.size(), qualities, outQualities);
      BOOST_REQUIRE_EQUAL(outQualities.size(), i);
    }

  q(0) = -1.345;
  q(4) = -3.2;
  qualities.push_back(q);
  outQualities.push_back(ublas::vector<double>(10));
  q(0) = 2.3;
  q(4) = 4.5;
  qualities.push_back(q);
  outQualities.push_back(ublas::vector<double>(10));

  norm.normalize(qualities.size(), qualities, outQualities);
  BOOST_REQUIRE_EQUAL(outQualities.size(), qualities.size());
  
  for (unsigned int i=0; i<qualities.size(); ++i)
    {
      //ublas::vector<double> &qq = qualities[i];
      ublas::vector<double> &oq = outQualities[i];
      if (i==0)
	{
	  BOOST_REQUIRE( fabs(oq(0) - (2*0.368999-1)) < 1e-5);
	  BOOST_REQUIRE( fabs(oq(4) - (2*0.415584-1)) < 1e-5);
	}
      else if (i==qualities.size()-2)
	{
	  BOOST_REQUIRE_EQUAL(oq(0), -1);
	  BOOST_REQUIRE_EQUAL(oq(4), -1);
	}
      else if (i==qualities.size()-1)
	{
	  BOOST_REQUIRE_EQUAL(oq(0), 1);
	  BOOST_REQUIRE_EQUAL(oq(4), 1);
	}
    }
}
