
#include <iostream>
#include <random>
#include <chrono>

#include "rnn.hpp"

int main(int, char **)
{
  RNN<> rnn(3, 2, {9,11,9,11});

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

  for (unsigned int i=0; i<x.size(); ++i)
    {
      x(i)=i;
    }

  auto t0 = std::chrono::steady_clock::now();
  for (unsigned int i=0; i<100; ++i)
    {
      rnn.reset();
      rnn.calc(x, y);
    }
  auto tf = std::chrono::steady_clock::now();
  std::cout << "expecting 600us" << std::endl;
  std::cout << "Don't forget -DNDEBUG" << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(tf-t0).count() << std::endl;
}
