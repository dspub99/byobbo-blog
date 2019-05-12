
#include <iostream>
#include <vector>

#include <boost/python.hpp>

namespace bp = boost::python;

#include "bpbbo.hpp"
#include "rnnLayer.hpp"

BOOST_PYTHON_MODULE(bpbbo)
{


  bp::class_<BPBBO<RNNLayerPlain>>("BPBBOPlain", bp::init<unsigned int, unsigned int, const bp::list&>())
    .def("numParams", &BPBBO<RNNLayerPlain>::numParams)
    .def("setChronoInit", &BPBBO<RNNLayerPlain>::setChronoInit)
    .def("setParams", &BPBBO<RNNLayerPlain>::setParams)
    .def("getThetas", &BPBBO<RNNLayerPlain>::getThetas)
    .def("getQualities", &BPBBO<RNNLayerPlain>::getQualities)
    .def("getOutThetas", &BPBBO<RNNLayerPlain>::getOutThetas)
    .def("resize", &BPBBO<RNNLayerPlain>::resize)
    .def("size", &BPBBO<RNNLayerPlain>::size)
    .def("query", &BPBBO<RNNLayerPlain>::query)
    .def("getW0", &BPBBO<RNNLayerPlain>::getW0)
    ;


  bp::class_<BPBBO<RNNLayerRes>>("BPBBORes", bp::init<unsigned int, unsigned int, const bp::list&>())
    .def("numParams", &BPBBO<RNNLayerRes>::numParams)
    .def("setChronoInit", &BPBBO<RNNLayerRes>::setChronoInit)
    .def("setParams", &BPBBO<RNNLayerRes>::setParams)
    .def("getThetas", &BPBBO<RNNLayerRes>::getThetas)
    .def("getQualities", &BPBBO<RNNLayerRes>::getQualities)
    .def("getOutThetas", &BPBBO<RNNLayerRes>::getOutThetas)
    .def("resize", &BPBBO<RNNLayerRes>::resize)
    .def("size", &BPBBO<RNNLayerRes>::size)
    .def("query", &BPBBO<RNNLayerRes>::query)
    .def("getW0", &BPBBO<RNNLayerRes>::getW0)
    ;


  bp::class_<BPBBO<RNNLayerJANET>>("BPBBOJANET", bp::init<unsigned int, unsigned int, const bp::list&>())
    .def("numParams", &BPBBO<RNNLayerJANET>::numParams)
    .def("setChronoInit", &BPBBO<RNNLayerJANET>::setChronoInit)
    .def("setParams", &BPBBO<RNNLayerJANET>::setParams)
    .def("getThetas", &BPBBO<RNNLayerJANET>::getThetas)
    .def("getQualities", &BPBBO<RNNLayerJANET>::getQualities)
    .def("getOutThetas", &BPBBO<RNNLayerJANET>::getOutThetas)
    .def("resize", &BPBBO<RNNLayerJANET>::resize)
    .def("size", &BPBBO<RNNLayerJANET>::size)
    .def("query", &BPBBO<RNNLayerJANET>::query)
    .def("getW0", &BPBBO<RNNLayerJANET>::getW0)
    ;


  Py_Initialize();
  np::initialize();
}



