#pragma once

#include "aspen/aspen.h"
#include "aspen/clustered_vertex.h"
#include "aspen/merge_batched.h"
#include "utils.h"

namespace aspen {

template <class Graph, class ClusteredGraph, class Active>
std::pair<size_t, size_t> BestEdgeMerge(Graph& G, ClusteredGraph& CG,
                                        double lower_threshold, Active& active,
                                        bool binarized_dendrogram = false,
                                        bool use_unnormalized_weights = false) {
  using vertex = typename Graph::vertex;
  auto GetVertex = [&](size_t id) { return vertex(id, CG.clusters[id].node); };
  size_t n = CG.clusters.size();

  size_t num_merged = 0;

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
      return (use_unnormalized_weights) ? std::make_pair(our_wgh, static_cast<size_t>(v)) :
        std::make_pair(our_wgh / (our_size * ngh_size),
                        static_cast<size_t>(v));
    };
    auto reduce_f = MaxM<double, size_t>();
    return vtx.out_neighbors().map_reduce(map_f, reduce_f);
  };
  auto GetTotalEdges = [&]() {
    auto degrees = parlay::delayed_seq<size_t>(n, [&](size_t i) {
      if (CG.clusters[i].is_active()) {
        return GetVertex(i).out_degree();
      }
      return (size_t)0;
    });
    return (size_t)parlay::reduce(degrees);
  };

  auto alive = parlay::filter(parlay::make_slice(active), [&](uintE u) {
    auto[wgh, id] = GetBestEdgeWeight(u);
    return CG.clusters[u].is_active() && (wgh >= lower_threshold) &&
           (id != std::numeric_limits<size_t>::max());
  });

  std::cout << "Thresholds: " << lower_threshold
            << ". Num-Active = " << alive.size() << std::endl;

  size_t rounds = 0;

  auto merge_target = parlay::sequence<std::pair<uintE, float>>(
      n, std::make_pair(UINT_E_MAX, float()));
  auto parents =
      parlay::sequence<uintE>::from_function(n, [&](size_t i) { return i; });
  size_t last_merged = 1;

  while (alive.size() > 1 && last_merged > 0) {
    std::cout << std::endl;
    size_t m = GetTotalEdges();
    std::cout << "Starting round: " << rounds
              << " lower_threshold = " << lower_threshold
              << " alive.size() = " << alive.size() << " total edges = " << m
              << std::endl;

    parlay::parallel_for(0, alive.size(), [&](size_t i) {
      uintE u = alive[i];
      auto[wgh, id] = GetBestEdgeWeight(u);
      merge_target[u] = {id, wgh};
      unite_impl(u, id, parents);
    });

    parlay::parallel_for(0, alive.size(), [&](size_t i) {
      uintE u = alive[i];
      uintE parent = find_compress(u, parents);
      merge_target[u].first = parent;
      if (merge_target[u].first == u) {
        merge_target[u].first = UINT_E_MAX;
      }
    });

    auto all_merges = parlay::delayed_seq<std::tuple<uintE, uintE, float>>(
        alive.size(), [&](size_t i) {
          uintE u = alive[i];
          return std::make_tuple(merge_target[u].first, u,
                                 merge_target[u].second);
        });
    auto merges = parlay::filter(all_merges, [&](const auto& tup) {
      return std::get<0>(tup) != UINT_E_MAX;
    });

    // Reset merge targets for live vertices.
    parlay::parallel_for(0, alive.size(), [&](size_t i) {
      uintE u = alive[i];
      merge_target[u] = std::make_pair(UINT_E_MAX, float());
    });
    std::cout << "Num merges = " << merges.size() << std::endl;
    num_merged += merges.size();
    last_merged = merges.size();

    auto[largest, largest_id] = largest_cc(parents);
    std::cout << "Num_cc = " << num_cc(parents) << std::endl;
    std::cout << "Largest cc now has size = " << largest
              << " id = " << largest_id << std::endl;

    UniteMergeBatched(G, merges, CG, GetVertex, m);

    using edge_tree = typename Graph::edge_tree;
    std::cout << "Num used nodes = " << edge_tree::GC::num_used_nodes()
              << std::endl;

    alive = parlay::filter(parlay::make_slice(alive), [&](uintE u) {
      auto[wgh, id] = GetBestEdgeWeight(u);
      return CG.clusters[u].is_active() && (wgh >= lower_threshold) &&
             (id != std::numeric_limits<size_t>::max());
    });
    std::cout << "nactive is now = " << alive.size() << std::endl;

    rounds++;
  }

  std::cout << "Finished bucket. Performed " << num_merged << " many merges."
            << std::endl;
  return {num_merged, rounds};
}

template <class Graph>
inline auto Affinity(Graph& G) {
  timer t;
  t.start();

  size_t n = G.num_vertices();
  auto CG = clustered_graph(G);

  using vertex = typename Graph::vertex;
  auto GetVertex = [&](size_t id) { return vertex(id, CG.clusters[id].node); };

  size_t num_outer_rounds = 0;
  size_t num_inner_rounds = 0;
  timer rt;

  auto GetTotalEdges = [&]() {
    auto degrees = parlay::delayed_seq<size_t>(
        n, [&](size_t i) { return GetVertex(i).out_degree(); });
    return parlay::reduce(degrees);
  };

  auto all_active = parlay::tabulate(n, [&](size_t i) -> uintE { return i; });
  auto active = parlay::filter(all_active, [&](uintE u) {
    auto vtx = GetVertex(u);
    return vtx.out_degree() > 0;
  });

  std::cout << "Num_active = " << active.size() << std::endl;
  num_outer_rounds++;

  double lower_threshold = 0;
  auto[num_merged, inner_rounds] =
      BestEdgeMerge(G, CG, lower_threshold, active);
  num_inner_rounds += inner_rounds;
  rt.next("ProcessGraphUnweightedAverage time");

  std::cout << "##### outer_rounds : " << num_outer_rounds << std::endl;
  std::cout << "##### inner_rounds : " << num_inner_rounds << std::endl;
  t.next("Total time");
  return CG.get_dendrogram();
}

double GetWeightThreshold(size_t iteration, size_t num_iterations,
                          double lower_threshold, double upper_threshold) {
  if (num_iterations == 1) return upper_threshold;
  return upper_threshold *
         std::pow(lower_threshold / upper_threshold,
                  static_cast<double>(iteration) /
                      (static_cast<double>(num_iterations) - 1.0));
}

template <class Graph>
inline auto SCC(Graph& G, double lower_threshold, double upper_threshold, size_t max_iters = 50) {
  timer t;
  t.start();
  std::cout << "Running SCC with lower_threshold = " << lower_threshold << " upper_threshold = " << upper_threshold << " num_iters = " << max_iters << std::endl;

  size_t n = G.num_vertices();
  auto CG = clustered_graph(G);

  using vertex = typename Graph::vertex;
  auto GetVertex = [&](size_t id) { return vertex(id, CG.clusters[id].node); };

  size_t num_outer_rounds = 0;
  size_t num_inner_rounds = 0;
  timer rt;

  auto GetTotalEdges = [&]() {
    auto degrees = parlay::delayed_seq<size_t>(
        n, [&](size_t i) { return GetVertex(i).out_degree(); });
    return parlay::reduce(degrees);
  };

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

  auto GetMinEdgeWeight = [&](size_t i) {
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
    auto reduce_f = MinM<double, size_t>();
    return vtx.out_neighbors().map_reduce(map_f, reduce_f);
  };

  auto best_edges = parlay::delayed_seq<double>(n, [&] (size_t i) {
    return (CG.clusters[i].is_active()) ? GetBestEdgeWeight(i).first : 0.0;
  });
  auto min_edges = parlay::delayed_seq<double>(n, [&] (size_t i) {
    return (CG.clusters[i].is_active()) ? GetBestEdgeWeight(i).first : std::numeric_limits<double>::infinity();
  });

  // lower_threshold should be 0.1 or 0.01
  if (upper_threshold == std::numeric_limits<double>::infinity()) {
    upper_threshold = parlay::reduce(best_edges, parlay::maxm<double>());
    std::cout << "upper_threshold is now: " << upper_threshold << std::endl;
  }
  // double lower_threshold = parlay::reduce(min_edges, parlay::minm<double>());
  std::cout << "max_iters = " << max_iters;

  auto all_active = parlay::tabulate(n, [&](size_t i) -> uintE { return i; });
  auto active = parlay::filter(all_active, [&](uintE u) {
    auto vtx = GetVertex(u);
    return vtx.out_degree() > 0;
  });

  while (active.size() > 1 && num_outer_rounds < max_iters) {
    std::cout << "Num_active = " << active.size() << std::endl;

    //// Instead, compute the max weight.
    // double best_edge = parlay::reduce(best_edges, parlay::maxm<double>());
    ////double threshold = best_edge / (1 + epsilon);
    //lower_threshold = parlay::reduce(min_edges, parlay::minm<double>());

    double threshold = GetWeightThreshold(num_outer_rounds, max_iters,
                                          lower_threshold, upper_threshold);
    // std::cout << "BESTEDGE = " << best_edge << " threshold = " << threshold << std::endl;
    std::cout << "Upper_threshold = " << upper_threshold << " lower_threshold = " << lower_threshold << " threshold this round = " << threshold << std::endl;

    num_outer_rounds++;

    auto[num_merged, inner_rounds] =
        BestEdgeMerge(G, CG, threshold, active);
    num_inner_rounds += inner_rounds;

    rt.next("ProcessGraphUnweightedAverage time");

    active = parlay::filter(parlay::make_slice(active), [&](uintE u) {
      auto vtx = GetVertex(u);
      return CG.clusters[u].is_active() && vtx.out_degree() > 0;
    });

    std::cout << "Bucket = " << num_outer_rounds
              << " total_edges = " << GetTotalEdges() << std::endl;
  }

  std::cout << "##### outer_rounds : " << num_outer_rounds << std::endl;
  std::cout << "##### inner_rounds : " << num_inner_rounds << std::endl;
  t.next("Total time");
  return CG.get_dendrogram();
}

}  // namespacghte aspen
