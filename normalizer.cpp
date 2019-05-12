
#include <cassert>
#include <vector>

#include "normalizer.hpp"

#include "minMax.hpp"
#include "vUtil.hpp"

Normalizer::Normalizer(unsigned int numQualities)
  : numQualities_(numQualities), dq_(numQualities)
{
  qOnes_ = ublas::vector<double>(numQualities);
  for (unsigned int i=0; i < qOnes_.size(); ++i)
    {
      qOnes_[i] = 1;
    }
}
