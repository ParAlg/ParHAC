#pragma once

#include "aspen/aspen.h"
#include <queue>
#include <vector>

namespace aspen {

using Dendrogram = parlay::sequence<std::pair<uintE, double>>;

struct clustered_vertex {
  clustered_vertex() {}
  clustered_vertex(uintE vtx_id, void* _node) {
    cas_size = 1;
    current_id = vtx_id;
    num_in_cluster = 1;
    active = true;
    node = _node;
  }

  bool is_active() { return active; }
  size_t cluster_size() { return num_in_cluster; }

  void merge_to(size_t center_id) {
    active = false;
    current_id = center_id;
  }

  // Initially equal to staleness, but cas'd in parallel rounds.
  uintE cas_size;
  // The "current" id of this cluster, updated upon a merge that keeps this
  // cluster active.
  uintE current_id;
  // Number of vertices contained in this cluster.
  uintE num_in_cluster;

  // Active = false iff this cluster is no longer active.
  bool active;

  void* node;
};

template <class Graph>
struct clustered_graph {

  using edge_tree = typename Graph::edge_tree;

  clustered_graph(Graph& G) : n(G.num_vertices()) {
    clusters = parlay::sequence<clustered_vertex>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      auto vertex = G.get_vertex(i);
      clusters[i] = clustered_vertex(i, vertex.edges);
    });
    merge_order_idx = 0;
    merge_order = parlay::sequence<std::pair<uintE, float>>(n, std::make_pair(UINT_E_MAX, float()));
  }

  size_t size_in_bytes() const {
    auto ith_size = parlay::delayed_seq<size_t>(n, [&] (size_t i) {
      auto& cluster_i = clusters[i];
      edge_tree et;
      if (cluster_i.node != nullptr && cluster_i.active) {
        et.root = cluster_i.node;
        auto noop = [](const auto& q) { return 0; };
        auto sz = et.size_in_bytes(noop);
        et.root = nullptr;
        return sz;
      }
      return (size_t)0;
    });
    return parlay::reduce(ith_size);
  }

  Dendrogram get_dendrogram() const {
    parlay::sequence<std::pair<uintE, double>> dendrogram(2*n-1, std::make_pair(UINT_E_MAX, double()));
    std::cout << "merge order idx = " << merge_order_idx << " n = " << n << std::endl;
    std::cout << "dendrogram length = " << dendrogram.size() << std::endl;

    auto mapping = parlay::sequence<uintE>(n, UINT_E_MAX);

    size_t cluster_id = n;  // Next new cluster id is n.
    for (size_t i=0; i<merge_order_idx; i++) {
      auto [u, wgh] = merge_order[i];
      if (u == UINT_E_MAX) {
        std::cout << "Invalid merge in merge_order!" << std::endl;
        exit(-1);
      }
      assert(u != UINT_E_MAX);

      uintE merge_target = clusters[u].current_id;

      // Get a new id.
      uintE new_id = cluster_id;
      ++cluster_id;

      assert(u < n);
      if (mapping[u] != UINT_E_MAX) {
        //std::cout << "remapped u to: " << mapping[u] << std::endl;
        u = mapping[u];
      }

      assert(merge_target < n);
      uintE old_merge_target = merge_target;
      if (mapping[merge_target] != UINT_E_MAX) {  // Actually a different (new) cluster.
        //std::cout << "remapped merge_target to: " << mapping[merge_target] << std::endl;
        merge_target = mapping[merge_target];
      }
      mapping[old_merge_target] = new_id;  // Subsequent merge to merge_target uses new_id.
      // std::cout << "new cluster id: " << new_id << std::endl;

      dendrogram[u] = std::make_pair(new_id, wgh);
      dendrogram[merge_target] = std::make_pair(new_id, wgh);
    }

    auto all_vertices = parlay::delayed_seq<uintE>(n, [&] (size_t i) {
      return (clusters[i].current_id == i) ? i : UINT_E_MAX; });
    auto bad = parlay::filter(all_vertices, [&] (uintE u) -> bool { return u != UINT_E_MAX; });

    std::queue<uintE> bad_queue;
    for (size_t i=0; i<bad.size(); i++) {
      bad_queue.push(bad[i]);
    }

    std::cout << "Num bad = " << bad_queue.size() << std::endl;

    while (bad_queue.size() > 1) {
      // std::cout << "queue size = " << bad_queue.size() << std::endl;
      uintE fst = bad_queue.front();
      bad_queue.pop();
      uintE snd = bad_queue.front();
      bad_queue.pop();

      if (fst < n && mapping[fst] != UINT_E_MAX) {  // An original vertex.
        fst = mapping[fst];
      }
      if (snd < n && mapping[snd] != UINT_E_MAX) {
        snd = mapping[snd];
      }

      uintE new_id = cluster_id;  // increments next_id
      cluster_id++;
      dendrogram[fst] = {new_id, double(0)};
      dendrogram[snd] = {new_id, double(0)};

      //std::cout << "Merged components for: " << fst << " " << snd << " dend_size = " << dendrogram.size() << std::endl;

      bad_queue.push(new_id);
    }

    std::cout << "Built dendrogram" << std::endl;

    // Check dendrogram.

//    for (size_t i=0; i<(2*n - 2); i++) {
//      std::cout << "Checking i = " << i << std::endl;
//      cluster_id = i;
//      double wgh = std::numeric_limits<double>::max();
//      while (true) {
//        auto parent = dendrogram[cluster_id].first;
//        auto merge_wgh = dendrogram[cluster_id].second;
//        std::cout << "id = " << cluster_id << " parent = " << parent << " wgh = " << merge_wgh << std::endl;
//        assert(wgh >= merge_wgh);
//        wgh = merge_wgh;
//        if (cluster_id == parent || parent == UINT_E_MAX) break;
//        cluster_id = parent;
//      }
//      std::cout << "i = " << i << " is good." << std::endl;
//    }

    return dendrogram;
  }

  size_t n;  // number of vertices
  parlay::sequence<clustered_vertex> clusters;  // the clusters

  size_t merge_order_idx;
  parlay::sequence<std::pair<uintE, float>> merge_order;
};

}  // namespace aspen
