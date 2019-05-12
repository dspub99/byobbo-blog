# byobbo-blog
Build your own black box optimizer

## Installation

I'm running Ubuntu 18.04.2 LTS with Python 3.6 in a virtualenv.

### Python
pip install -r requirements.txt

### boost
sudo apt install python3-dev
sudo apt install python3-numpy   # so boost will detect numpy and build libboost_numpy
Get boost_1_68_0.tar.gz from https://www.boost.org/users/download/
./bootstrap.sh --with-python=/usr/bin/python3
./b2 toolset=gcc cxxstd=14
sudo ./b2 toolset=gcc cxxstd=14 install

## Building

(My apologies for including boost and C++ code in a blog post.)

make
make test
make pytest


