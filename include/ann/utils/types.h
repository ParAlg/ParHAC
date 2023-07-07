// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TYPES
#define TYPES

#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace ann {

// for a file in .fvecs or .bvecs format, but extendible to other types
template <typename T>
struct Tvec_point {
  int id;
  int cnt;
  float norm;
  parlay::slice<T*, T*> coordinates;
  Tvec_point() : coordinates(parlay::make_slice<T*, T*>(nullptr, nullptr)) {}

//  void initialize_norm() {
//    // Compute the l2_norm
//    efanna2e::DistanceInnerProduct dist_f;
//    float dist = dist_f.compare(coordinates.begin(), coordinates.begin(), coordinates.size());
//    l2_norm = sqrt(dist);
//  }

  parlay::sequence<int> out_nbh = parlay::sequence<int>();
  parlay::sequence<int> new_out_nbh = parlay::sequence<int>();
  parlay::sequence<int> ngh = parlay::sequence<int>();
};

// for an ivec file, which contains the ground truth
// only info needed is the coordinates of the nearest neighbors of each point
struct ivec_point {
  int id;
  parlay::slice<int*, int*> coordinates;
  ivec_point()
      : coordinates(parlay::make_slice<int*, int*>(nullptr, nullptr)) {}
};

}  // namespace ann

#endif
