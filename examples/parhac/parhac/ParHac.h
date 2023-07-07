#pragma once

#include "aspen/aspen.h"
#include "aspen/clustered_vertex.h"
#include "aspen/merge_batched.h"
// #include "aspen/merge.h"
#include "parlay/random.h"
#include <type_traits>

namespace aspen {

timer merge_t("MergeTimer");

// At the end of this call, we will have performed merges and ensured that no
// edges exist with weights between [lower_threshold, ...).
//
// Returns the number of clusters merged by this routine.
template <class Graph, class ClusteredGraph, class MergeTarget, class Active,
          class Colors>
size_t ProcessGraphUnweightedAverage(
    Graph& G, ClusteredGraph& CG, double lower_threshold, double max_weight,
    Active& active, Colors& colors, MergeTarget& merge_target,
    parlay::random& rnd, size_t& num_inner_rounds, size_t m, double eps = 0.1) {
  // Identify vertices with edges between [lower_threshold, max_weight)
  size_t n = CG.clusters.size();

  constexpr uint8_t kBlue = 1;
  constexpr uint8_t kRed = 2;

  size_t num_merged = 0;

  using vertex = typename Graph::vertex;
  auto GetVertex = [&](size_t id) { return vertex(id, CG.clusters[id].node); };

  auto GetBestEdgeWeight = [&](size_t i) {
    assert(i < n);
    auto vtx = GetVertex(i);
    double our_size = CG.clusters[i].cluster_size();
    auto map_f = [&](const auto& u, const auto& v,
                     const auto& wgh) -> std::pair<double, size_t> {
      double ngh_size = CG.clusters[v].cluster_size();
      double our_wgh = wgh;
      if (u == v) {
        return std::make_pair(static_cast<double>(0),
                              std::numeric_limits<size_t>::max());
      }
      return std::make_pair(our_wgh / (our_size * ngh_size),
                            static_cast<size_t>(v));
    };
    auto reduce_f = MaxM<double, size_t>();
    return vtx.out_neighbors().map_reduce(map_f, reduce_f);
  };

  parlay::parallel_for(0, n, [&](size_t i) {
    if (CG.clusters[i].is_active()) {
      auto[wgh, id] = GetBestEdgeWeight(i);
      if ((id != std::numeric_limits<size_t>::max()) &&
          (wgh >= lower_threshold)) {
        active[i] = true;
      }
    }
  });

  // TODO: this can be made sparse (lower-pri)
  auto alive = parlay::pack_index(parlay::make_slice(active));

  std::cout << "Thresholds: " << lower_threshold << " and " << max_weight
            << ". Num-Active = " << alive.size() << std::endl;

  if (max_weight < 0.01) {
    return 0;
  }

  double one_plus_eps = 1 + eps;
  size_t rounds = 0;
//  double round_granularity = (max_weight - lower_threshold) / 16384;

  while (alive.size() > 0) {
    timer t("----- ProcessBucket", false);
    //std::cout << std::endl;
    //std::cout << "Starting round: " << rounds << std::endl;

    // Use fine-grained merge sim
//    double merge_weight = max_weight - (round_granularity * rounds);

    parlay::parallel_for(0, alive.size(), [&](size_t i) {
      uintE u = alive[i];
      if (rnd.ith_rand(u) & 1) {
        colors[u] = kRed;
      } else {
        colors[u] = kBlue;
      }
    });

    parlay::parallel_for(
        0, alive.size(),
        [&](size_t i) {
          uintE u = alive[i];
          if (colors[u] == kBlue) {
            // seq selection for now (todo: could run in par for high-degrees)
            double our_size = CG.clusters[u].cluster_size();

//                    auto iter_f = [&](const auto& u, const auto& v, const
//                    auto& wgh) {
//                      double neighbor_size = CG.clusters[v].cluster_size();
//                      double our_wgh = wgh;
//                      double real_weight = our_wgh / (our_size *
//                      neighbor_size);
//                      if (!CG.clusters[v].is_active()) {
//                        std::cout << "Edge to inactive neighbor! " << v << " exiting." << std::endl;
//                        exit(-1);
//                      }
//                      if (real_weight >= lower_threshold || almost_equal(real_weight, lower_threshold, 3)) {
//                        if (colors[v] != kBlue) {  // symmetry break
//                          uintE ngh_cur_size = neighbor_size;
//                          uintE upper_bound = one_plus_eps * ngh_cur_size *
//                          (max_weight / real_weight);
//                          uintE our_cluster_size = our_size;
//                          auto opt =
//                          fetch_and_add_threshold(&(CG.clusters[v].cas_size),
//                                                             our_cluster_size,
//                                                             upper_bound);
//                          if (opt.has_value()) {  // Success in the F&A!
//                            merge_target[u] = std::make_pair(v, real_weight);
//                            return false;  // done.
//                          }
//                        }
//                      }
//                      return true;  // keep going
//                    };
//                    GetVertex(u).out_neighbors().foreach_cond(iter_f);

            auto vtx = GetVertex(u);
            auto out_degree = vtx.out_degree();
            auto neighbors =
                parlay::sequence<std::pair<float, uintE>>::uninitialized(
                    out_degree);
            auto map_f = [&](uintE u_id, uintE v_id, float weight, size_t idx) {
              neighbors[idx] = std::make_pair(weight, v_id);
            };
            vtx.out_neighbors().map_index(map_f);

            parlay::sort_inplace(neighbors);
            for (size_t j = 0; j < neighbors.size(); ++j) {
              auto[wgh, v] = neighbors[j];

              double neighbor_size = CG.clusters[v].cluster_size();
              double real_weight = wgh / (our_size * neighbor_size);
              if (!CG.clusters[v].is_active()) {
                std::cout << "Edge to inactive neighbor! " << v << " exiting."
                          << std::endl;
                exit(-1);
              }
              if (real_weight >= lower_threshold  || almost_equal(real_weight, lower_threshold, 3)) {
                if (colors[v] != kBlue) {  // symmetry break
                  uintE ngh_cur_size = neighbor_size;
                  // uintE upper_bound = one_plus_eps * ngh_cur_size *
                  // (max_weight / real_weight);
                  uintE upper_bound = one_plus_eps * ngh_cur_size;
                  uintE our_cluster_size = our_size;
                  auto opt =
                      fetch_and_add_threshold(&(CG.clusters[v].cas_size),
                                              our_cluster_size, upper_bound);
                  if (opt.has_value()) {  // Success in the F&A!
                    merge_target[u] = std::make_pair(v, real_weight);
                    break;
                  }
                }
              }
            }

          }
        },
        1);

    auto all_merges = parlay::delayed_seq<std::tuple<uintE, uintE, float>>(
        alive.size(), [&](size_t i) {
          uintE u = alive[i];
          return std::make_tuple(merge_target[u].first, u,
                                 merge_target[u].second);
        });
    auto merges = parlay::filter(all_merges, [&](const auto& tup) {
      return std::get<0>(tup) != UINT_E_MAX;
    });
    t.next("Compute Merges");

    // Reset merge targets for live vertices.
    parlay::parallel_for(0, alive.size(), [&](size_t i) {
      uintE u = alive[i];
      merge_target[u] = std::make_pair(UINT_E_MAX, float());
    });
    // std::cout << "Alive.size = " << alive.size()
    //           << " Num merges = " << merges.size() << std::endl;
    num_merged += merges.size();

    merge_t.start();
    UniteMergeBatched(G, merges, CG, GetVertex, m);
    //UniteMerge(G, merges, CG, GetVertex);
    merge_t.stop();
    t.next("Merge Batched");

    rnd = rnd.next();
    // Reset colors.
    // Recompute active and update alive.

    parlay::parallel_for(0, alive.size(),
                         [&](size_t i) {
                           uintE u = alive[i];
                           bool is_active = false;
                           if (CG.clusters[u].is_active()) {
                             auto[wgh, id] = GetBestEdgeWeight(u);
                             is_active =
                                 (id != std::numeric_limits<size_t>::max() &&
                                  (wgh >= lower_threshold));
                           }
                           active[u] = is_active;
                           // Reset color.
                           colors[u] = 0;
                         },
                         1);

    // Update alive vertices.
    alive =
        parlay::filter(make_slice(alive), [&](uintE u) { return active[u]; });
    t.next("Reset Active");

    rounds++;
  }

  std::cout << "Finished bucket. Performed " << num_merged << " many merges."
            << std::endl;
  num_inner_rounds += rounds;
  return num_merged;
}

template <class Graph>
inline auto ParHac(Graph& G, double epsilon = 0.1, bool get_size = false) {
  timer t;
  t.start();

  double one_plus_eps = 1 + epsilon;

  size_t n = G.num_vertices();
  size_t m = G.num_edges();
  std::cout << "ParHac: Graph has " << n << " vertices and " << m << " edges."
            << std::endl;
  parlay::random rnd;

  auto CG = clustered_graph<Graph>(G);

  if (get_size) {
    auto[used, unused] = parlay::internal::get_default_allocator().stats();
    std::cout << "Used: " << used << " unused: " << unused << std::endl;
    parlay::internal::get_default_allocator().print_stats();
    using vertex_gc = typename Graph::vertex_gc;
    using edge_gc = typename Graph::vertex_gc;
    vertex_gc::print_stats();
    edge_gc::print_stats();
    G.get_tree_sizes("graph", "");
  }

  using vertex = typename Graph::vertex;
  auto GetVertex = [&](size_t id) { return vertex(id, CG.clusters[id].node); };

  size_t num_active = n;

  auto colors = parlay::sequence<uint8_t>(n, 0);  // 1 is blue, 2 is red
  auto merge_target = parlay::sequence<std::pair<uintE, float>>(
      n, std::make_pair(UINT_E_MAX, float()));

  size_t num_inner_rounds = 0;
  size_t num_outer_rounds = 0;
  size_t max_inner_rounds = 0;
  timer rt;

  auto active = parlay::sequence<bool>(n, false);
  auto all_active = parlay::tabulate(n, [&](size_t i) -> uintE { return i; });
  auto cur_active = parlay::filter(all_active, [&](uintE u) {
    auto vtx = GetVertex(u);
    return vtx.out_degree() > 0;
  });
  auto merges = parlay::sequence<uintE>(n, UINT_E_MAX);

  auto GetBestEdgeWeight = [&](size_t i) {
    auto vtx = GetVertex(i);
    double our_size = CG.clusters[i].cluster_size();
    auto map_f = [&](const auto& u, const auto& v, const auto& wgh) {
      double ngh_size = CG.clusters[v].cluster_size();
      return std::make_pair(wgh / (our_size * ngh_size), v);
    };
    auto reduce_f = MaxM<double, size_t>();
    return vtx.out_neighbors().map_reduce(map_f, reduce_f);
  };

  auto GetTotalEdges = [&]() {
    auto degrees = parlay::delayed_seq<size_t>(
        n, [&](size_t i) { return GetVertex(i).out_degree(); });
    return parlay::reduce(degrees);
  };

  size_t max_size = 0;
  while (num_active > 1) {
    if (get_size) {
      size_t size_in_bytes = CG.size_in_bytes();
      max_size = std::max(max_size, size_in_bytes);
      std::cout << "Max size is: " << max_size << std::endl;
    }
    size_t cur_m = GetTotalEdges();
    std::cout << "Bucket = " << num_outer_rounds
              << " total_active = " << num_active << " total_edges = " << cur_m
              << std::endl;

    // The current max-weight calculated in this round.
    double max_weight = 0;
    rt.start();
    parlay::parallel_for(0, n, [&](size_t i) {
      if (CG.clusters[i].active) {
        auto[wgh, neighbor_id] = GetBestEdgeWeight(i);
        if (wgh > max_weight) {
          write_max(&max_weight, wgh);
        }
      }
    });
    rt.next("Max weight time");

    std::cout << "Max weight = " << max_weight << std::endl;
    if (max_weight == 0) break;

    num_outer_rounds++;

    size_t inner_rounds_before = num_inner_rounds;
    size_t num_merged = ProcessGraphUnweightedAverage(
        G, CG, max_weight / one_plus_eps, max_weight, active, colors,
        merge_target, rnd, num_inner_rounds, cur_m, epsilon);
    if (num_merged == 0) {
      break;
    }

    rt.next("ProcessGraphUnweightedAverage time");
    max_inner_rounds =
        std::max(max_inner_rounds, num_inner_rounds - inner_rounds_before);
    // auto[used, unused] = parlay::internal::get_default_allocator().stats();
    // std::cout << "Used = " << used << " unused = " << unused << std::endl;
    num_active -= num_merged;
  }

  double total_time = t.stop();
  std::cout << "##### outer_rounds : " << num_outer_rounds << std::endl;
  std::cout << "##### inner_rounds : " << num_inner_rounds << std::endl;
  std::cout << "##### max_inner_rounds : " << max_inner_rounds << std::endl;
  std::cout << "##### size_in_bytes : " << max_size << std::endl;
  std::cout << "##### total_time : " << total_time << std::endl;

  t.next("Total time");

  return CG.get_dendrogram();
}

}  // namespace aspen
