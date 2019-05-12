#!/usr/bin/env python

interfaces = [
    ("Plain", "RNNLayerPlain"),
    ("Res", "RNNLayerRes"),
    ("JANET", "RNNLayerJANET"),
    ]

iface0 = """
  bp::class_<BPBBO<__LAYER__>>("BPBBO__PYNAME__", bp::init<unsigned int, unsigned int, const bp::list&>())
    .def("numParams", &BPBBO<__LAYER__>::numParams)
    .def("setChronoInit", &BPBBO<__LAYER__>::setChronoInit)
    .def("setParams", &BPBBO<__LAYER__>::setParams)
    .def("getThetas", &BPBBO<__LAYER__>::getThetas)
    .def("getQualities", &BPBBO<__LAYER__>::getQualities)
    .def("getOutThetas", &BPBBO<__LAYER__>::getOutThetas)
    .def("resize", &BPBBO<__LAYER__>::resize)
    .def("size", &BPBBO<__LAYER__>::size)
    .def("query", &BPBBO<__LAYER__>::query)
    .def("getW0", &BPBBO<__LAYER__>::getW0)
    ;
"""
#    .def("paramStartW0", &BPBBO<__LAYER__>::paramStartW0)
#    .def("numLayers", &BPBBO<__LAYER__>::numLayers)
#    .def("paramStartLayer", &BPBBO<__LAYER__>::paramStartLayer)

    
template = open("bpbbo.template").read()

ifaces = []
for pyname, layer in interfaces:
    iface = iface0.replace("__PYNAME__", pyname)
    iface = iface.replace("__LAYER__", layer)
    ifaces.append(iface)

print (template.replace("__INTERFACES__", '\n'.join(ifaces)))


