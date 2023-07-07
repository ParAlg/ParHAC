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

#pragma once

#include <math.h>
#include <algorithm>
#include <random>
#include <set>
#include <unordered_set>
#include <ann/utils/geometry.h>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include <ann/utils/beamSearch.h>

namespace ann {

bool report_stats = true;

template <class Dist>
struct knn_index {
  using T = typename Dist::T;
  size_t maxDeg;
  size_t beamSize;
  // int k;
  double r2_alpha;  // alpha parameter for round 2 of robustPrune
  size_t d;
  parlay::sequence<float> normalized;
  parlay::sequence<Tvec_point<T>>& v;
  std::string metric;

  using tvec_point = Tvec_point<T>;
  using fvec_point = Tvec_point<float>;
  tvec_point* medoid;
  using pid = std::pair<int, float>;
  using slice_tvec = decltype(make_slice(parlay::sequence<tvec_point>()));
  using index_pair = std::pair<int, int>;
  using slice_idx = decltype(make_slice(parlay::sequence<index_pair>()));

  knn_index(int md, int bs, double a, size_t dim, std::string metric, parlay::sequence<Tvec_point<T>>& pts)
      : maxDeg(md), beamSize(bs), r2_alpha(a), d(dim), metric(metric), v(pts) {}

  parlay::sequence<float> centroid_helper(slice_tvec a) {
    if (a.size() == 1) {
      parlay::sequence<float> centroid_coords = parlay::sequence<float>(d);
      for (size_t i = 0; i < d; i++)
        centroid_coords[i] = static_cast<float>((a[0].coordinates)[i]);
      return centroid_coords;
    } else {
      size_t n = a.size();
      parlay::sequence<float> c1;
      parlay::sequence<float> c2;
      parlay::par_do_if(n > 1000,
                        [&]() { c1 = centroid_helper(a.cut(0, n / 2)); },
                        [&]() { c2 = centroid_helper(a.cut(n / 2, n)); });
      parlay::sequence<float> centroid_coords = parlay::sequence<float>(d);
      for (size_t i = 0; i < d; i++) {
        float result = (c1[i] + c2[i]) / 2;
        centroid_coords[i] = result;
      }
      return centroid_coords;
    }
  }

  tvec_point* medoid_helper(fvec_point* centroid, slice_tvec a) {
    if (a.size() == 1) {
      return &(a[0]);
    } else {
      size_t n = a.size();
      tvec_point* a1;
      tvec_point* a2;
      parlay::par_do_if(
          n > 1000, [&]() { a1 = medoid_helper(centroid, a.cut(0, n / 2)); },
          [&]() { a2 = medoid_helper(centroid, a.cut(n / 2, n)); });
      float d1 = Dist::distance(*centroid, *a1, d);
      float d2 = Dist::distance(*centroid, *a2, d);
      if (Dist::compare(d1, d2))
        return &(a1[0]);
      else
        return &(a2[0]);
    }
  }

  // computes the centroid and then assigns the approx medoid as the point in v
  // closest to the centroid
  void find_approx_medoid(parlay::sequence<Tvec_point<T>>& v) {
    parlay::sequence<float> centroid = centroid_helper(parlay::make_slice(v));
    fvec_point centroidp = Tvec_point<float>();
    centroidp.coordinates = parlay::make_slice(centroid);
    medoid = medoid_helper(&centroidp, parlay::make_slice(v));
    std::cout << "Medoid ID: " << medoid->id << std::endl;
  }

  void print_set(std::set<int> myset) {
    std::cout << "[";
    for (std::set<int>::iterator it = myset.begin(); it != myset.end(); ++it) {
      std::cout << *it << ", ";
    }
    std::cout << "]" << std::endl;
  }

  // robustPrune routine as found in DiskANN paper, with the exception that the
  // new candidate set
  // is added to the field new_nbhs instead of directly replacing the out_nbh of
  // p
  void robustPrune(tvec_point* p, parlay::sequence<pid> candidates,
                   double alpha) {
    // make sure the candidate set does not include p (done later)
    // add out neighbors of p to the candidate set
    for (size_t i = 0; i < (p->out_nbh.size()); i++) {
      candidates.push_back(std::make_pair(
          p->out_nbh[i], Dist::distance(v[p->out_nbh[i]], *p, d)));
    }
    // Sort (dissimilarity or similarity)
    Dist::order_candidates(candidates, [](pid a){return a.second;});

    parlay::sequence<int> new_nbhs = parlay::sequence<int>();
    new_nbhs.reserve(maxDeg);

    size_t candidate_idx = 0;
    while (new_nbhs.size() <= maxDeg && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p->id || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          float dist_starprime = Dist::distance(v[p_star], v[p_prime],
              d);
          float dist_pprime = candidates[i].second;
          if (Dist::compare(alpha * dist_starprime, dist_pprime)) {
            candidates[i].first = -1;
          }
        }
      }
    }
    p->new_out_nbh = std::move(new_nbhs);
  }

  void robustPrune(Tvec_point<T>* p, parlay::sequence<int> candidates,
                   double alpha) {
    parlay::sequence<pid> cc;
    cc.reserve(candidates.size() + p->out_nbh.size());
    for (size_t i=0; i<candidates.size(); ++i) {
      cc.push_back(std::make_pair(candidates[i], Dist::distance(v[candidates[i]], *p, d)));
    }
    return robustPrune(p, std::move(cc), alpha);
  }

  void build_index(bool from_empty = true, bool two_pass = true) {
    parlay::internal::timer t("BuildIndex", true);
    clear(v);
    t.next("clear");

    if (metric == "angular") {
      if constexpr (std::is_same<float, T>::value) {
        normalized = parlay::sequence<float>(d * v.size());
        parlay::parallel_for(0, v.size(), [&] (size_t i) {
          float norm = Dist::get_norm(v[i]);
          auto ptr = v[i].coordinates.begin();
          size_t offset = d * i;
          for (size_t j=0; j<d; ++j) {
            normalized[offset + j] = ptr[j] / norm;
          }
          v[i].coordinates = parlay::make_slice(&(normalized[offset]), &(normalized[offset + d]));
        });
      } else {
        std::cout << "Angular with non-float data currently not supported" << std::endl;
        exit(-1);
      }
    }

    if constexpr (Dist::needs_norms) {
      parlay::parallel_for(0, v.size(), [&] (size_t i) {
        Dist::StoreNorm(v[i]);
      });
    }

    // populate with random edges
    if (not from_empty) {
      random_index(v, maxDeg);
      t.next("random_index");
    }
    // find the medoid, which each beamSearch will begin from
    find_approx_medoid(v);
    t.next("find_approx_medoid");
    build_index_inner();
    t.next("build_index_inner (pass 1)");
    if (two_pass) {
      build_index_inner(r2_alpha);
      t.next("build_index_inner (pass 2)");
    }
  }

  void build_index_inner(double alpha = 1.0, bool random_order = false) {
    size_t n = v.size();
    size_t inc = 0;
    parlay::sequence<int> rperm;
    parlay::random rnd;
    if (random_order)
      rperm = parlay::random_permutation<int>(static_cast<int>(n), time(NULL));
    else
      rperm = parlay::tabulate(v.size(), [&](int i) { return i; });
    while (pow(2, inc) < n) {
      rnd = rnd.next();
      parlay::internal::timer rt("Round");
      size_t floor = static_cast<size_t>(pow(2, inc)) - 1;
      size_t ceiling = std::min(static_cast<size_t>(pow(2, inc + 1)), n) - 1;
      // search for each node starting from the medoid, then call
      // robustPrune with the visited list as its candidate set
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        parlay::sequence<pid> visited = ann::beam_search_2<Dist>(&(v[rperm[i]]), v, medoid, beamSize, d);

        auto i_perm = parlay::random_permutation<int>(static_cast<int>(visited.size()), rnd.ith_rand(i));
        auto new_visited = parlay::tabulate(visited.size(), [&] (size_t i) { return visited[i_perm[i]]; });
        robustPrune(&(v[rperm[i]]), std::move(new_visited), alpha);

        if (report_stats) v[rperm[i]].cnt = visited.size();

        //robustPrune(&(v[rperm[i]]), std::move(visited), alpha);
      }, 1);
      rt.next("Search and Robust Prune");

      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      parlay::sequence<parlay::sequence<index_pair>> to_flatten =
          parlay::sequence<parlay::sequence<index_pair>>(ceiling - floor);
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t point_id = rperm[i];
        parlay::sequence<int> new_nbh = std::move(v[point_id].new_out_nbh);
        parlay::sequence<index_pair> edges =
            parlay::sequence<index_pair>(new_nbh.size());
        for (size_t j = 0; j < new_nbh.size(); j++) {
          edges[j] = std::make_pair(new_nbh[j], point_id);
        }
        to_flatten[i - floor] = edges;
        v[point_id].out_nbh = std::move(new_nbh);
        v[point_id].new_out_nbh = parlay::sequence<int>();
      });
      auto new_edges = parlay::flatten(to_flatten);
      auto grouped_by = parlay::group_by_key(new_edges);
      rt.next("Bidirect, groupby");

      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        size_t index = grouped_by[j].first;
        parlay::sequence<int> candidates = std::move(grouped_by[j].second);
        size_t newsize = candidates.size() + (v[index].out_nbh).size();
        if (newsize <= maxDeg) {
          for (size_t k = 0; k < candidates.size();
               k++) {  // try concatenating instead of pushing back
            (v[index].out_nbh).push_back(candidates[k]);
          }
        } else {
          robustPrune(&(v[index]), std::move(candidates), alpha);
          parlay::sequence<int> new_nbh = v[index].new_out_nbh;
          v[index].out_nbh = new_nbh;
          (v[index].new_out_nbh).clear();
        }
      });
      std::cout << "Grouped_by_size = " << grouped_by.size() << ", new_edges.size = " << new_edges.size() << std::endl;
      rt.next("Add bidirectional edges");
      inc += 1;
    }
  }

  void searchNeighbors(parlay::sequence<Tvec_point<T>>& q, int beamSizeQ,
                       int k) {
    searchFromSingle<Dist>(q, v, beamSizeQ, k, d, medoid);
  }

  void searchNeighborsFromPointsInIndex(parlay::sequence<Tvec_point<T>>& q, int beamSizeQ,
                       int k) {
    searchFromPointsInIndex<Dist>(q, v, beamSizeQ, k, d, medoid);
  }

  template <class TrueDist>
  double measureQuality(parlay::sequence<Tvec_point<T>>& q, int k) {
    parlay::sequence<double> recalls(q.size());

    size_t d = v[0].coordinates.size();
    auto get_true_neighbors = [&] (Tvec_point<T>* p) {
      parlay::sequence<std::pair<double, int>> distances(v.size());
      parlay::parallel_for(0, v.size(), [&] (size_t i) {
        distances[i] = std::make_pair(TrueDist::distance(*p, v[i], d), i);
      });
      parlay::sort_inplace(parlay::make_slice(distances), TrueDist::compare_pair);
      parlay::sequence<int> neighbors = parlay::tabulate(k, [&] (size_t i) { return distances[i].second; });
      return neighbors;
    };

    parlay::parallel_for(0, q.size(), [&] (size_t i) {
      parlay::sequence<int> true_neighbors = get_true_neighbors(&q[i]);
      parlay::sequence<int>& neighbors = q[i].ngh;
      std::unordered_set<int> true_nghs(true_neighbors.begin(), true_neighbors.end());
      size_t num_correct = 0;
//      for (size_t j=0; j<q[i].ngh.size(); ++j) {
////        std::cout << "ReportedDist = " << TrueDist::distance(q[i], v[q[i].ngh[j]], d) << std::endl;
//      }
      for (int ngh : neighbors) {
//        std::cout << "TrueDist = " << TrueDist::distance(q[i], v[ngh], d) << std::endl;
        if (true_nghs.find(ngh) != true_nghs.end()) {
          num_correct++;
        }
      }
      recalls[i] = static_cast<double>(num_correct) / k;
    }, 1000000);

    return parlay::reduce(recalls) / q.size();
  }

};

}  // namespace ann