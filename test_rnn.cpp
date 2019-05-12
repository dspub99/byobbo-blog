#define BOOST_TEST_MODULE test_rnn
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <random>

#include "rnn.hpp"

BOOST_AUTO_TEST_CASE( test_construct )
{
  RNN<> rnn(3, 2, {5,6});

  ublas::vector<double> x(3);
  ublas::vector<double> y(2);
}

BOOST_AUTO_TEST_CASE( test_setParams )
{
  RNN<> rnn(3, 2, {6,5,4,3});

  ublas::vector<double> x(3);
  ublas::vector<double> y(2);

  unsigned int np = rnn.numParams();
  std::vector<double> params(np);
  BOOST_REQUIRE_EQUAL(rnn.setParams(params, 0), params.size());
}


BOOST_AUTO_TEST_CASE( test_calc )
{
  RNN<> rnn(3, 2, {3,4});

  unsigned int np = rnn.numParams();
  std::vector<double> params(np);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<double> normal_dist(0, 1);

  for (unsigned int i=0; i<params.size(); ++i)
    {
      params[i] = normal_dist(rng);
    }
  rnn.setParams(params, 0);

  ublas::vector<double> x(3);
  ublas::vector<double> y(2);

  for (unsigned int i=0; i<3; ++i)
    {
      x(i)=i;
    }

  double ycheck=-1e99;
  rnn.reset();
  for (unsigned int i=0; i<10; ++i)
    {
      rnn.calc(x, y);
      for (unsigned int j=0; j<y.size(); ++j)
	{
	  BOOST_REQUIRE(y[j] >= -1 && y[j] <= 1);
	  //std::cout << y[j] << std::endl;
	}
      BOOST_REQUIRE(y[y.size()-1] != ycheck);
      ycheck = y[y.size()-1];
      //std::cout << ycheck << std::endl;
    }

    for (unsigned int i=0; i<3; ++i)
    {
      x(i)=2*i;
    }
    rnn.reset();
  for (unsigned int i=0; i<10; ++i)
    {
      rnn.calc(x, y);
    }
  std::cout << y[y.size()-1] << " " << ycheck << std::endl;
  BOOST_REQUIRE(y[y.size()-1]!=ycheck);
}
