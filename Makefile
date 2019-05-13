
# http://scottmcpeak.com/autodepend/autodepend.html

.PHONY: clean all depend test tidy cppclean show

CXX = g++
LD_FLAGS = -L/usr/local/lib   -lboost_unit_test_framework -lboost_system -lboost_python36 -lboost_numpy36 -lpython3.6m -lm
#-lpthread -lssl -lcrypto -lboost_iostreams -lz -ldl
INCLUDES = -I$(HOME)/include -I/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include -I/usr/include/python3.6m
DEFINES = -DBOOST_TEST_DYN_LINK -DBOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR -DNDEBUG
CC_FLAGS = $(INCLUDES) $(DEFINES) -std=c++14 -g3 -pg -rdynamic -W -Wall -Wno-noexcept-type -fmax-errors=3 -MP -MMD -O3 -fpic
CCC = $(CXX) -pg -o $@ $^ $(LD_FLAGS)

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(SRC_FILES:.cpp=.o)

TESTS = test_rnn test_bbo test_janet test_minmax test_normalizer test_bpUtil test_vUtil test_csv
TARGETS = $(TESTS) time_rnn time_normalizer bpbbo.so


all: $(TARGETS)

kill:
	pkill -f byobbo/

show:
	echo $(OBJ_FILES)

-include $(SRC_FILES:.cpp=.d)


pytest:
	./runPyTests && ./runMPITests

test: $(TESTS)
	 $(foreach t,$(TESTS), ./$t;)	

pylint:
	pyflakes *.py;

lint:
	# TODO: includes and defines
	clang-tidy *.hpp *.cpp --



clean:
	rm -rf *.o $(TARGETS) a.out core *~ *.d gmon.out	__pycache__


cppclean:
	cppclean --verbose *.hpp *.cpp

%.o: %.cpp
	$(CXX) $(CC_FLAGS) -c -o $@ $<

bpbbo.cpp: generateInterface.py bpbbo.template
	./generateInterface.py > bpbbo.cpp

test_rnn: test_rnn.o
	$(CCC)

test_bbo: test_bbo.o normalizer.o
	$(CCC)

test_janet: test_janet.o normalizer.o
	$(CCC)

test_bpUtil: test_bpUtil.o
	$(CCC)

test_vUtil: test_vUtil.o
	$(CCC)

test_csv: test_csv.o
	$(CCC)

test_minmax: test_minmax.o
	$(CCC)

test_normalizer: test_normalizer.o normalizer.o
	$(CCC)

time_rnn: time_rnn.o
	$(CCC)

time_normalizer: time_normalizer.o normalizer.o
	$(CCC)

bpbbo.so: bpbbo.o normalizer.o
	$(CXX) -o bpbbo.so -shared bpbbo.o normalizer.o $(LD_FLAGS)


