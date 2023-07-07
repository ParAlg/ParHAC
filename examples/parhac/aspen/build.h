#pragma once

#include "macros.h"
#include "utils.h"

#include <cpam/parse_command_line.h>

// #define USE_PAM 1

namespace aspen {

template <class TA, class TB>
size_t intersect(const TA& A, const TB& B) {
  size_t i = 0, j = 0, nA = A.size(), nB = B.size();
  size_t ans = 0;
  while (i < nA && j < nB) {
    if (A[i] ==B[j])
      i++, j++, ans++;
    else if (A[i] < B[j])
      i++;
    else
      j++;
  }
  return ans;
}

// Combines two hash values.
inline uint64_t hash_combine(uint64_t hash_value_1, uint64_t hash_value_2) {
  // This is the same as boost's 32-bit `hash_combine` implementation, but with
  // 2 ^ 64 / (golden ratio) chosen as an arbitrary 64-bit additive magic number
  // rather than 2 ^ 32 / (golden ratio).
  return hash_value_1 ^ (hash_value_2 + 0x9e3779b97f4a7c15 +
                         (hash_value_1 << 6) + (hash_value_1 >> 2));
}

template <class SliceU, class SliceV>
double get_weight(uintE u, uintE v, const std::string& weight_scheme,
                  SliceU& u_nghs, SliceV& v_nghs) {
  if (weight_scheme == "log_deg") {
    return log(u_nghs.size() + v_nghs.size() + 1);
  } else if (weight_scheme == "inv_log_deg") {
    return 1.0 / (1 + log(u_nghs.size() + v_nghs.size() + 1));
  } else if (weight_scheme == "deg") {
    return u_nghs.size() + v_nghs.size();
  } else if (weight_scheme == "rand") {
    size_t hash = parlay::hash64_2(u) ^ parlay::hash64_2(v);
    return hash % 2048 + 1;
  } else if (weight_scheme == "log_rand") {
    size_t hash = parlay::hash64_2(u) ^ parlay::hash64_2(v);
    return 1 + log((hash % 2048) + 1);
  } else if (weight_scheme == "perturbed_log") {
    size_t hash_x = parlay::hash64_2(u);
    size_t hash_y = parlay::hash64_2(v);
    auto min_hash = std::min(hash_x, hash_y);
    auto max_hash = std::max(hash_x, hash_y);

    size_t degree_x = u_nghs.size();
    size_t degree_y = v_nghs.size();

    uint64_t fprint = 1 + hash_combine(min_hash, max_hash) %
                              static_cast<uint64_t>(degree_x + degree_y);
    double low_bits = 1 / static_cast<double>(
                              hash_combine(min_hash, max_hash) % 100000);

    return 1 / (log2(degree_x + degree_y + fprint) + low_bits);
  }
  else if (weight_scheme == "unit") {
    return 1.0;
  } else if (weight_scheme == "triangle") {
    // intersection
    return 1 + intersect(u_nghs, v_nghs);
  } else {
    std::cout << "Unknown scheme: " << weight_scheme << std::endl;
    exit(-1);
  }
}

// Builds an aspen graph from the given static graph.
template <class weight>
auto graph_to_edges(std::tuple<size_t, size_t, uintT*, uintE*>& parsed_graph,
                    cpam::commandLine& P, size_t num_batches = 1) {
  size_t n = std::get<0>(parsed_graph);

  auto offsets = std::get<2>(parsed_graph);
  auto E = std::get<3>(parsed_graph);
  auto degs = parlay::sequence<size_t>::from_function(
      n, [&](size_t i) { return offsets[i + 1] - offsets[i]; });
  size_t sum_degs = parlay::scan_inplace(parlay::make_slice(degs));
  assert(sum_degs == std::get<1>(parsed_graph));

  auto weight_scheme = P.getOptionValue("-weight_scheme", "unit");

#ifdef USE_PAM
  using ngh_and_weight = std::pair<vertex_id, weight>;
#else
  using ngh_and_weight = std::tuple<vertex_id, weight>;
#endif
  using edge = std::pair<vertex_id, ngh_and_weight>;

  auto edges = parlay::sequence<edge>::uninitialized(sum_degs);

  ////  double scaling = log(2*n);
  //  auto get_weight = [&] (uintE u, uintE v) {
  //    return 1.0;
  ////    size_t hash = parlay::hash64_2(u) ^ parlay::hash64_2(v);
  ////    return log(hash % 2048);
  ////    return hash % 2048;
  //
  //    size_t deg_u = degs[u];
  //    size_t deg_v = degs[v];
  //    return static_cast<double>(1) / (log(static_cast<double>(deg_u + deg_v +
  //    1UL)) / log(2));
  //
  ////    size_t rand = (parlay::hash64_2(u) ^ parlay::hash64_2(v)) %
  ///(10UL*(deg_u + deg_v + 1UL));
  ////    double unquantized_weight = static_cast<double>(1) /
  ///log(static_cast<double>(deg_u + deg_v + rand + 1UL));
  ////    return unquantized_weight;
  ////    uintE quantized_weight = std::max((uintE)(unquantized_weight *
  ///scaling), (uintE)1);
  ////    return quantized_weight;
  //  };

  parlay::parallel_for(
      0, n,
      [&](size_t i) {
        size_t k = degs[i];
        auto deg = offsets[i + 1] - offsets[i];
        size_t offset = offsets[i];
        auto nghs_i =
            parlay::delayed_seq<uintE>(deg, [&](size_t j) { return E[offset + j]; });
        parlay::parallel_for(0, deg, [&](size_t j) {
          auto ngh = E[offset + j];
          auto ngh_deg = offsets[ngh + 1] - offsets[ngh];
          auto nghs_j = parlay::delayed_seq<uintE>(
              ngh_deg, [&](size_t k) { return E[offsets[ngh] + k]; });
          edges[k + j] = std::make_pair(
              i, ngh_and_weight(
                     ngh, get_weight(i, ngh, weight_scheme, nghs_i, nghs_j)));
        });
      },
      1);

//  auto sims = parlay::delayed_seq<float>(edges.size(), [&] (size_t i) {
//    return std::get<1>(edges[i].second);
//  });
//  float max_sim = parlay::reduce(sims, parlay::maxm<float>());
//  parlay::parallel_for(0, edges.size(), [&] (size_t i) {
//    std::get<1>(edges[i].second) = std::get<1>(edges[i].second) / max_sim;
//  });

  return edges;
}

// Builds an aspen graph from the given static weighted graph.
template <class weight>
auto graph_to_edges(
    std::tuple<size_t, size_t, uintT*, std::tuple<uintE, float>*>& parsed_graph,
    size_t num_batches = 1) {
  size_t n = std::get<0>(parsed_graph);

  auto offsets = std::get<2>(parsed_graph);
  auto E = std::get<3>(parsed_graph);
  auto degs = parlay::sequence<size_t>::from_function(
      n, [&](size_t i) { return offsets[i + 1] - offsets[i]; });
  size_t sum_degs = parlay::scan_inplace(parlay::make_slice(degs));
  assert(sum_degs == std::get<1>(parsed_graph));

#ifdef USE_PAM
  using ngh_and_weight = std::pair<vertex_id, weight>;
#else
  using ngh_and_weight = std::tuple<vertex_id, weight>;
#endif
  using edge = std::pair<vertex_id, ngh_and_weight>;

  auto edges = parlay::sequence<edge>::uninitialized(sum_degs);

  parlay::parallel_for(0, n,
                       [&](size_t i) {
                         size_t k = degs[i];
                         auto deg = offsets[i + 1] - offsets[i];
                         size_t offset = offsets[i];
                         parlay::parallel_for(0, deg, [&](size_t j) {
                           auto[ngh, wgh] = E[offset + j];
                           edges[k + j] =
                               std::make_pair(i, ngh_and_weight(ngh, wgh));
                         });
                       },
                       1);
  return edges;
}

}  // namespace aspen
