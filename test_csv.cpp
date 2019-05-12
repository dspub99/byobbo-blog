#define BOOST_TEST_MODULE test_csv
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>
#include <random>

#include "commonStoreVector.hpp"

BOOST_AUTO_TEST_CASE( test_construct )
{
  Py_Initialize();
  np::initialize();

  CommonStoreVector csv(10);
  csv.ublas()[3] = 3.14;

  BOOST_REQUIRE_EQUAL(csv.ublas()[3], bp::extract<double>(csv.numpy()[3]));
}

BOOST_AUTO_TEST_CASE( test_many )
{
  Py_Initialize();
  np::initialize();

  std::vector<std::shared_ptr<CommonStoreVector>> csvs;
  for (unsigned int i = 0; i<100; ++i)
    {
      csvs.emplace_back(std::make_shared<CommonStoreVector>(10));
    }
}

