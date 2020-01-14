/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#if defined(FOAM_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "image_proc.hpp"


namespace foam
{

template<typename T, xt::layout_type L>
struct IsImage<xt::pytensor<T, 2, L>> : std::true_type {};

template<typename T, xt::layout_type L>
struct IsImageArray<xt::pytensor<T, 3, L>> : std::true_type {};

} // foam

namespace py = pybind11;


PYBIND11_MODULE(image_proc, m)
{
  xt::import_numpy();

  using namespace foam;

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 3>& src)
    { return nanmeanImageArray(src); },
    py::arg("src").noconvert());
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 3>& src)
    { return nanmeanImageArray(src); },
    py::arg("src").noconvert());

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanImageArray(src, keep); },
    py::arg("src").noconvert(), py::arg("keep"));
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanImageArray(src, keep); },
    py::arg("src").noconvert(), py::arg("keep"));

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 2>& src1, const xt::pytensor<double, 2>& src2)
    { return nanmeanImageArray(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 2>& src1, const xt::pytensor<float, 2>& src2)
    { return nanmeanImageArray(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());

  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<double, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));
  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<float, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));

  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<double, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));
  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<float, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));

  m.def("maskImageData", &maskImageData<xt::pytensor<double, 2>>, py::arg("src").noconvert());
  m.def("maskImageData", &maskImageData<xt::pytensor<float, 2>>, py::arg("src").noconvert());

  m.def("maskImageData", (void (*)(xt::pytensor<double, 2>&, double, double))
                         &maskImageData<xt::pytensor<double, 2>, double>,
                         py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImageData", (void (*)(xt::pytensor<float, 2>&, float, float))
                         &maskImageData<xt::pytensor<float, 2>, float>,
                         py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&))
                         &maskImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&))
                         &maskImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&, double, double))
                         &maskImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>, double>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert(),
                         py::arg("lb"), py::arg("ub"));
  m.def("maskImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&, float, float))
                         &maskImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>, float>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert(),
                         py::arg("lb"), py::arg("ub"));

  m.def("maskImageData", &maskImageData<xt::pytensor<double, 3>, double>,
                         py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImageData", &maskImageData<xt::pytensor<float, 3>, float>,
                         py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&))
                         &maskImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&))
                         &maskImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&, double, double))
                         &maskImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>, double>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&, float, float))
                         &maskImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>, float>,
                         py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImageData", &maskImageData<xt::pytensor<double, 3>>, py::arg("src").noconvert());
  m.def("maskImageData", &maskImageData<xt::pytensor<float, 3>>, py::arg("src").noconvert());

  m.def("correctOffset", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<double, 3>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());
  m.def("correctOffset", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<float, 3>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());

  m.def("correctOffset", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<double, 2>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());
  m.def("correctOffset", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<float, 2>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());

  m.def("correctGain", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                       &correctImageData<GainPolicy, xt::pytensor<double, 3>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());
  m.def("correctGain", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                       &correctImageData<GainPolicy, xt::pytensor<float, 3>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());

  m.def("correctGain", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                       &correctImageData<GainPolicy, xt::pytensor<double, 2>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());
  m.def("correctGain", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                       &correctImageData<GainPolicy, xt::pytensor<float, 2>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());

  m.def("correctGainOffset", (void (*)(xt::pytensor<double, 3>&,
                                       const xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                             &correctImageData<xt::pytensor<double, 3>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
  m.def("correctGainOffset", (void (*)(xt::pytensor<float, 3>&,
                                       const xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                             &correctImageData<xt::pytensor<float, 3>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());

  m.def("correctGainOffset", (void (*)(xt::pytensor<double, 2>&,
                                       const xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                             &correctImageData<xt::pytensor<double, 2>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
  m.def("correctGainOffset", (void (*)(xt::pytensor<float, 2>&,
                                       const xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                             &correctImageData<xt::pytensor<float, 2>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
}
