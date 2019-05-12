
#include <iostream>
#include <random>
#include <chrono>

#include "normalizer.hpp"

#include <boost/numeric/ublas/vector.hpp>

namespace ublas = boost::numeric::ublas;

int main(int, char **)
{
  unsigned int n = 32;
  Normalizer normalizer(n);

  std::vector<ublas::vector<double>> v0, v1;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<double> normal_dist(0, 1);

  for (unsigned int i = 0; i<100; ++i)
    {
      v0.emplace_back(n);
      v1.emplace_back(n);
	for (unsigned int j = 0; j<n; ++j)
	  {
	    v0[i](j) = normal_dist(rng);
	  }
    }

  auto t0 = std::chrono::steady_clock::now();
  for (unsigned int i=0; i<100; ++i)
    {
      normalizer.normalize(100, v0, v1);
    }
  auto tf = std::chrono::steady_clock::now();
  std::cout << "expecting 5100us" << std::endl;
  std::cout << "Don't forget -DNDEBUG" << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(tf-t0).count() << std::endl;
}
