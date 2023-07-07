#pragma once

#include "aspen/aspen.h"
#include "merge.h"

// Needed to make frequent large allocations efficient with standard
// malloc implementation.  Otherwise they are allocated directly from
// vm.
#if !defined __APPLE__ && !defined LOWMEM
#include <malloc.h>
//comment out the following two lines if running out of memory
// static int __ii =  mallopt(M_MMAP_MAX,0);
// static int __jj =  mallopt(M_TRIM_THRESHOLD,-1);
#endif

#ifdef DEBUG
#define log_message(msg) std::cout << msg << std::endl;
#else
#define log_message(msg) while(false);
#endif

namespace aspen {

  template<class T>
  typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
      almost_equal(T x, T y, int ulp)
  {
      // the machine epsilon has to be scaled to the magnitude of the values used
      // and multiplied by the desired precision in ULPs (units in the last place)
      return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
          // unless the result is subnormal
          || std::fabs(x-y) < std::numeric_limits<T>::min();
  }

  template <class Edge>
  void sort_updates(parlay::sequence<Edge>& edges, size_t n) {
    size_t m = edges.size();
    size_t vtx_bits = parlay::log2_up(n);
    auto edge_to_long = [vtx_bits](Edge e) -> size_t {
      return (static_cast<size_t>(std::get<0>(e)) << vtx_bits) +
             static_cast<size_t>(std::get<1>(e));
    };

    // Only apply integer sort if it will be work-efficient
    if (n <= (m * parlay::log2_up(m))) {
      log_message("running integer sort: " << n << " and mm = " << (m
            * parlay::log2_up(m)));
      parlay::integer_sort_inplace(parlay::make_slice(edges),
                                   edge_to_long);
    } else {
      log_message("running sample sort");
      parlay::sort_inplace(parlay::make_slice(edges),
                           std::less<Edge>());
    }
  }

  template <class Graph, class ClusteredGraph>
  void delete_edges_batch_inplace(Graph& G, ClusteredGraph& CG,
      parlay::sequence<std::pair<uintE, uintE>>& deletions) {
    timer t("Delete-Edges", false);
    size_t m = deletions.size();
    if (m == 0) return;
    sort_updates(deletions, G.num_vertices());
    t.next("Sort time");

    auto& E = deletions;
    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
    });
    auto I = parlay::pack_index<size_t>(start_im);
    t.next("Generate starts time");

    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> uintE {
      return std::get<1>(E[i]);
    });

    using edge_tree = typename Graph::edge_tree;
    parlay::parallel_for(0, I.size(), [&] (size_t i) {
      auto start = I[i];
      auto end = (i == I.size()-1) ? m : I[i+1];
      auto incoming = parlay::make_slice(Vals.begin() + start, Vals.begin() + end);

      uintE id = std::get<0>(E[start]);
      assert(CG.clusters[id].is_active());
      edge_tree t;
      t.root = CG.clusters[id].node;
      auto ret = edge_tree::multi_delete_sorted(std::move(t), incoming);
      CG.clusters[id].node = ret.root;
      assert(ret.root == nullptr || ret.ref_cnt() == 1);
      ret.root = nullptr;
    }, 1);
  }

  template <class ClusteredGraph>
  void PruneEdgesBelowThreshold(ClusteredGraph& CG, double threshold) {
    timer t("Delete-Active", false);

    log_message("Starting to filter active vertices.");
    using edge_tree = typename ClusteredGraph::edge_tree;

    parlay::parallel_for(0, CG.clusters.size(), [&] (size_t i) {
      if (CG.clusters[i].is_active()) {
        edge_tree t;
        t.root = CG.clusters[i].node;
        size_t our_size = CG.clusters[i].cluster_size();
        auto filter_f = [&] (const auto& et) {
          uintE ngh = std::get<0>(et);
          size_t ngh_size = CG.clusters[ngh].cluster_size();
          double wgh_sum = std::get<1>(et);
          double real_weight = wgh_sum / (our_size * ngh_size);
          return real_weight >= threshold || almost_equal(real_weight, threshold, 3);
        };
        auto ret = edge_tree::filter(std::move(t), filter_f);
        CG.clusters[i].node = ret.root;
        assert(ret.root == nullptr || ret.ref_cnt() == 1);
        ret.root = nullptr;
      }
    }, 1);
    t.next("Filter active vertices time");
  }

  template <class Graph, class ClusteredGraph>
  void delete_from_active_vertices(Graph& G, ClusteredGraph& CG) {
    timer t("Delete-Active", false);

    log_message("Starting to filter active vertices.");
    using edge_tree = typename Graph::edge_tree;
    parlay::parallel_for(0, CG.clusters.size(), [&] (size_t i) {
      if (CG.clusters[i].is_active()) {
        edge_tree t;
        t.root = CG.clusters[i].node;
        auto filter_f = [&] (const auto& et) {
          uintE ngh = std::get<0>(et);
          return CG.clusters[ngh].is_active();
        };
        auto ret = edge_tree::filter(std::move(t), filter_f);
        CG.clusters[i].node = ret.root;
        assert(ret.root == nullptr || ret.ref_cnt() == 1);
        ret.root = nullptr;
      }
    }, 1);
    t.next("Filter active vertices time");
  }

  template <class Graph, class ClusteredGraph>
  void insert_edges_batch_inplace(Graph& G, ClusteredGraph& CG,
      parlay::sequence<std::tuple<uintE, uintE, float>>& insertions) {
    size_t m = insertions.size();
    if (m == 0) return;
    timer t("Delete-Edges", false);
    sort_updates(insertions, G.num_vertices());
    t.next("Sort time");

    auto& E = insertions;
    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (std::get<0>(E[i]) != std::get<0>(E[i - 1])));
    });
    auto I = parlay::pack_index<size_t>(start_im);
    t.next("Generate starts time");

    using ngh_and_weight = typename Graph::ngh_and_weight;
    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> ngh_and_weight {
      return ngh_and_weight(std::get<1>(E[i]), std::get<2>(E[i]));
    });

    auto replace = [&] (const auto& a, const auto& b) { return a + b; };

    using edge_tree = typename Graph::edge_tree;
    parlay::parallel_for(0, I.size(), [&] (size_t i) {
      auto start = I[i];
      auto end = (i == I.size()-1) ? m : I[i+1];
      auto incoming = parlay::make_slice(Vals.begin() + start, Vals.begin() + end);

      uintE id = std::get<0>(E[start]);
      assert(CG.clusters[id].is_active());
      edge_tree t;
      t.root = CG.clusters[id].node;
      auto ret = edge_tree::multi_insert_sorted(std::move(t), incoming, replace);
      CG.clusters[id].node = ret.root;
      assert(ret.root == nullptr || ret.ref_cnt() == 1);
      ret.root = nullptr;
    }, 1);
    t.next("multi insert time");
  }

  template <class ClusteredGraph>
  void ApplyMergesToDendrogram(ClusteredGraph& CG, const parlay::sequence<uintE>& starts,
      parlay::sequence<std::tuple<uintE, uintE, float>>& merge_seq) {
    // Sort the weights incident to each merged vertex. This ensures that the
    // merge order stores parallel merges into this vertex in descending order
    // of weight (preventing inversions in the dendrogram).
    parlay::parallel_for(0, starts.size(), [&] (size_t i) {
      size_t start = starts[i];
      size_t end = (i == starts.size() - 1) ? merge_seq.size() : starts[i+1];
      auto our_merges = merge_seq.cut(start, end);

      auto comp = [&] (const auto& l, const auto& r) {
        return std::get<2>(l) > std::get<2>(r);  // sort by weights in desc. order
      };
      parlay::sort_inplace(our_merges, comp);
    });

    // Lastly for each merged vertex, save it in the merge_order seq.
    parlay::parallel_for(0, merge_seq.size(), [&] (size_t i) {
      // assert(merge_order[CG.merge_order_idx + i].first == UINT_E_MAX);
      CG.merge_order[CG.merge_order_idx + i] = std::make_pair(std::get<1>(merge_seq[i]), std::get<2>(merge_seq[i]));
    });
    CG.merge_order_idx += merge_seq.size();
    // std::cout << "CG.merge_order_idx is now: " << CG.merge_order_idx << " merge_seq_size = " << merge_seq.size() << std::endl;
  }

  // Given a sequence of (u,v) pairs representing that a satellite v
  // merges with a center u, perform all merges and update the
  // internal clustered graph representation. Note the following
  // requirements:
  // - there can be multiple v's (satellites) that are merging with a
  // single u (center)
  // - the graph induced on the merges consists of a set of stars. So
  //   if the (u,v) pair appears in the input, there can be no (w,u)
  //   merge in the input.
  template <class Graph, class ClusteredGraph, class GetVertex>
  void UniteMergeBatched(Graph& G, parlay::sequence<std::tuple<uintE,
      uintE, float>>& merge_seq, ClusteredGraph& CG, GetVertex&
      get_vertex, size_t m) {

    using EdgeTriple = std::tuple<uintE, uintE, float>;
    using edge_tree = typename Graph::edge_tree;
    timer t("UniteMergeBatched", false);
    timer tt("UniteMergeBatched-Overall", false);

    //size_t max_batch_size = 500000000;
    size_t max_batch_size = 5000000000;
    //size_t max_batch_size = 100000000;
    //size_t full_deletion_threshold = m/30;
    size_t full_deletion_threshold = m;

    // Sort merges based on the centers (semi-sort also ok).
    parlay::sort_inplace(make_slice(merge_seq));
    t.next("Sort Merges");

    // Identify the start of each center's merges.
    auto all_starts =
        parlay::delayed_seq<uintE>(merge_seq.size(), [&](size_t i) {
          if ((i == 0) ||
              std::get<0>(merge_seq[i]) != std::get<0>(merge_seq[i - 1])) {
            return (uintE)i;
          }
          return UINT_E_MAX;
        });
    // The start of every center's list of satellites
    auto starts =
        parlay::filter(all_starts, [&](uintE v) { return v != UINT_E_MAX; });
    auto edge_sizes = parlay::sequence<size_t>::uninitialized(merge_seq.size());

    // In parallel over every component:
    //   1. Make the merge center the cluster with the largest number of
    //   out-edges.
    //   2. Logically perform all of the merges: updates the clusters
    //   for all satellites, setting their parent to be the center,
    //   and then deactivate the satellite cluster.
    parlay::parallel_for(0, starts.size(), [&](size_t i) {
      size_t start = starts[i];
      size_t end = (i == starts.size() - 1) ? merge_seq.size() : starts[i + 1];

      uintE center_id = std::get<0>(merge_seq[start]);

      // Write the neighbor size before scan.
      // - Update the current_id for each (being merged) neighbor.
      // - Set the active flag for each such neighbor to false ( Deactivate ).
      size_t total_size = 0;
      for (size_t j = start; j < end; j++) {
        uintE ngh_id = std::get<1>(merge_seq[j]);

        edge_sizes[j] = get_vertex(ngh_id).out_degree();

        // Was an active cluster before.
        assert(CG.clusters[ngh_id].current_id == ngh_id);
        assert(CG.clusters[ngh_id].is_active());

        // Update total_size with the size of the cluster being merged
        total_size += CG.clusters[ngh_id].cluster_size();

        // Update id to point to the merge target, and deactivate.
        CG.clusters[ngh_id].merge_to(center_id);
      }

      // Update this center's cluster size.
      CG.clusters[center_id].num_in_cluster += total_size;
      // Update the CAS size for the next round.
      CG.clusters[center_id].cas_size = CG.clusters[center_id].num_in_cluster;
    }, 1);
    t.next("Preprocess Merges");

    // Scan to compute #edges we need to merge.
    size_t total_edges = parlay::scan_inplace(make_slice(edge_sizes));
    std::cout << "Total edges = " << total_edges << std::endl;
    if (total_edges == 0) {
      ApplyMergesToDendrogram(CG, starts, merge_seq);
      return;
    }

    if (total_edges < max_batch_size) {

#ifdef DEBUG
      auto [used, unused] = parlay::internal::get_default_allocator().stats();
      std::cout << "Used = " << used << " unused = " << unused << std::endl;
#endif

      auto edges = parlay::sequence<EdgeTriple>::uninitialized(2*total_edges);
      auto all_deletions = parlay::sequence<std::pair<uintE, uintE>>::uninitialized(total_edges);

      // Copy edges from trees to edges and deletions.
      parlay::parallel_for(0, merge_seq.size(), [&] (size_t i) {
        uintE center_id = std::get<0>(merge_seq[i]);
        uintE satellite_id = std::get<1>(merge_seq[i]);
        size_t deletion_offset = edge_sizes[i];
        size_t offset = 2 * deletion_offset;
        // Map over every edge to a neighbor v, incident to the satellite.
        auto map_f = [&] (const uintE& u, const uintE& v, const float& wgh, size_t k) {
          bool v_active = CG.clusters[v].active;
          uintE merged_id = CG.clusters[v].current_id;
          // (1) v is active (itself a center) and it is not this center
          // (2) symmetry break to prevent sending a (u,v) edge between
          // two deactivated vertices twice to the activated targets.
          if ((merged_id != center_id) && (v_active || (satellite_id < v))) {
            edges[offset + 2 * k] = {center_id, merged_id, wgh};
            edges[offset + 2 * k + 1] = {merged_id, center_id, wgh};
          } else {
            edges[offset + 2 * k] = {UINT_E_MAX, UINT_E_MAX, 0};
            edges[offset + 2 * k + 1] = {UINT_E_MAX, UINT_E_MAX, 0};
          }

          // Handle deletions.
          if (v_active) {
            all_deletions[deletion_offset + k] = std::make_pair(v, u);
          } else {
            all_deletions[deletion_offset + k] = {UINT_E_MAX, UINT_E_MAX};
          }
        };
        get_vertex(satellite_id).out_neighbors().map_index(map_f);
      }, 1);
      t.next("Copy Edges Time");

      auto deletions = parlay::filter(parlay::make_slice(all_deletions), [&] (const auto& e) {
        return e.first != UINT_E_MAX;
      });

      // Clear deleted vertices.
      parlay::parallel_for(0, merge_seq.size(), [&] (size_t i) {
        auto id = std::get<1>(merge_seq[i]);
        auto& cluster_i = CG.clusters[id];
        edge_tree et;
        et.root = cluster_i.node;
        assert(et.ref_cnt() == 1);
        cluster_i.node = nullptr;
      });
      t.next("Clear deleted vertices");

      // (1) First perform the deletions. This makes life easier later when we
      // perform the insertions.
      delete_edges_batch_inplace(G, CG, deletions);
      t.next("Delete edges");

      // No references to deactivated vertices in any neighborlist of an
      // active vertex at this point. Next, we simply perform all
      // insertions (concurrently) in parallel. An alternate approach
      // that we used with PAM is to perform some semisorts + scans to
      // merge "same" edges, and then perform the insertions as a bulk
      // step. Need to understand which approach is faster.
      // Filter out empty edge triples.
      auto pred = [&](const EdgeTriple& e) {
        return (std::get<0>(e) != UINT_E_MAX) && (std::get<0>(e) != std::get<1>(e));
      };
      auto orig_filtered_edges = parlay::filter(parlay::make_slice(edges), pred);

      // Sort triples lexicographically.
      //parlay::sort_inplace(parlay::make_slice(filtered_edges));
      auto filtered_edges = parlay::stable_sort(parlay::make_slice(orig_filtered_edges));

      // Scan over the triples, merging edges going to the same neighbor
      // with (+).
      auto scan_f = [&](const EdgeTriple& l, const EdgeTriple& r) -> EdgeTriple {
        auto[l_u, l_v, l_wgh] = l;
        auto[r_u, r_v, r_wgh] = r;
        if (l_u != l_v || r_u != r_v) return {r_u, r_v, r_wgh};
        return {r_u, r_v, l_wgh + r_wgh};
      };
      EdgeTriple id = {UINT_E_MAX, UINT_E_MAX, 0};
      auto scan_mon = parlay::make_monoid(scan_f, id);
      // After the scan, the last occurence of each ngh has the
      // aggregated weight.
      parlay::scan_inclusive_inplace(parlay::make_slice(filtered_edges),
                                     scan_mon);

      size_t filtered_edges_size = filtered_edges.size();
      std::cout << "filtered edges size = " << filtered_edges_size << std::endl;
      // Apply filter index to extract the last occurence of each edge.
      auto idx_f = [&](const EdgeTriple& e, size_t idx) {
        const auto & [ u, v, wgh ] = e;
        if (u == UINT_E_MAX) return false;
        if (idx < (filtered_edges_size - 1)) {
          const auto & [ next_u, next_v, next_wgh ] = filtered_edges[idx + 1];
          // Next edge is not the same as this one
          return u != next_u || v != next_v;
        }
        return true;
      };
      auto inserts =
          parlay::filter_index(parlay::make_slice(filtered_edges), idx_f);
      t.next("Preprocess insertions");

      // Add the edge weights.
      std::cout << "Num insertions: " << inserts.size() << std::endl;
      insert_edges_batch_inplace(G, CG, inserts);
      t.next("Insert insertions");

      ApplyMergesToDendrogram(CG, starts, merge_seq);
      return;
    }

    // First perform deletions.

    if (total_edges >= full_deletion_threshold) {
      delete_from_active_vertices(G, CG);
      t.next("Filter all active vertices");
    }

    size_t deletions_finished = 0;
    size_t last_vtx_offset = 0;
    size_t total_deletions = total_edges;
    while (deletions_finished < total_deletions) {
      size_t next_threshold = std::min(deletions_finished + max_batch_size, total_deletions);
      size_t offset = parlay::internal::binary_search(parlay::make_slice(edge_sizes), next_threshold, std::less<size_t>());
      size_t vtx_bs = offset - last_vtx_offset;

      size_t num_edges = ((offset == edge_sizes.size()) ? total_edges : edge_sizes[offset]) - deletions_finished;
      auto edges = parlay::sequence<EdgeTriple>::uninitialized(2*num_edges);
      parlay::sequence<std::pair<uintE, uintE>> all_deletions;
      if (total_edges < full_deletion_threshold) {
        all_deletions = parlay::sequence<std::pair<uintE, uintE>>::uninitialized(num_edges);
      }

      std::cout << "Finished " << deletions_finished << " deletions. Next vtx_bs = " << vtx_bs << " num_edges = " << num_edges << ". Total_deletions = " << total_deletions << std::endl;

      // Copy edges from trees to edges and deletions.
      parlay::parallel_for(0, vtx_bs, [&] (size_t idx) {
        size_t i = idx + last_vtx_offset;
        uintE center_id = std::get<0>(merge_seq[i]);
        uintE satellite_id = std::get<1>(merge_seq[i]);
        size_t deletion_offset = edge_sizes[i] - deletions_finished;
        size_t edges_offset = 2*deletion_offset;
        // Map over every edge to a neighbor v, incident to the satellite.
        auto map_f = [&] (const uintE& u, const uintE& v, const float& wgh, size_t k) {
          bool v_active = CG.clusters[v].active;
          uintE merged_id = CG.clusters[v].current_id;
          // (1) v is active (itself a center) and it is not this center
          // (2) symmetry break to prevent sending a (u,v) edge between
          // two deactivated vertices twice to the activated targets.
          if ((merged_id != center_id) && (v_active || (satellite_id < v))) {
            edges[edges_offset + 2 * k] = {center_id, merged_id, wgh};
            edges[edges_offset + 2 * k + 1] = {merged_id, center_id, wgh};
          } else {
            edges[edges_offset + 2 * k] = {UINT_E_MAX, UINT_E_MAX, 0};
            edges[edges_offset + 2 * k + 1] = {UINT_E_MAX, UINT_E_MAX, 0};
          }

          if (total_edges < full_deletion_threshold) {
            if (v_active) {
              all_deletions[deletion_offset + k] = std::make_pair(v, u);
            } else {
              all_deletions[deletion_offset + k] = {UINT_E_MAX, UINT_E_MAX};
            }
          }
        };
        get_vertex(satellite_id).out_neighbors().map_index(map_f);
      });
      t.next("Copy edges time");

      // Clear deleted vertices.
      parlay::parallel_for(0, vtx_bs, [&] (size_t i) {
        auto id = std::get<1>(merge_seq[last_vtx_offset + i]);
        auto& cluster_i = CG.clusters[id];
        edge_tree et;
        et.root = cluster_i.node;
        assert(et.ref_cnt() == 1);
        cluster_i.node = nullptr;
      }, 1);
      t.next("Clear deleted time");

      last_vtx_offset += vtx_bs;
      deletions_finished += num_edges;

      if (total_edges < full_deletion_threshold) {
        auto deletions = parlay::filter(parlay::make_slice(all_deletions), [&] (const auto& e) {
          return e.first != UINT_E_MAX;
        });
        // (1) First perform the deletions. This makes life easier later when we
        // performing the insertions.
        delete_edges_batch_inplace(G, CG, deletions);
        deletions.clear();
        t.next("Delete edges");
      }

      // No references to deactivated vertices in any neighborlist of an
      // active vertex at this point. Next, we simply perform all
      // insertions (concurrently) in parallel. An alternate approach
      // that we used with PAM is to perform some semisorts + scans to
      // merge "same" edges, and then perform the insertions as a bulk
      // step. Need to understand which approach is faster.
      // Filter out empty edge triples.
      auto pred = [&](const EdgeTriple& e) {
        return (std::get<0>(e) != UINT_E_MAX) && (std::get<0>(e) != std::get<1>(e));
      };
      auto orig_filtered_edges = parlay::filter(parlay::make_slice(edges), pred);
      edges.clear();
      t.next("Filter insertions");

      // Sort triples lexicographically.
      //parlay::sort_inplace(parlay::make_slice(filtered_edges));
      auto filtered_edges = parlay::stable_sort(parlay::make_slice(orig_filtered_edges));
      orig_filtered_edges.clear();
      t.next("Sort time");
      std::cout << "filtered_edges size = " << filtered_edges.size() << std::endl;

      // Scan over the triples, merging edges going to the same neighbor
      // with (+).
      auto scan_f = [&](const EdgeTriple& l, const EdgeTriple& r) -> EdgeTriple {
        const auto& [l_u, l_v, l_wgh] = l;
        const auto& [r_u, r_v, r_wgh] = r;
        if (l_u != l_v || r_u != r_v) return {r_u, r_v, r_wgh};
        return {r_u, r_v, l_wgh + r_wgh};
      };
      EdgeTriple id = {UINT_E_MAX, UINT_E_MAX, 0};
      auto scan_mon = parlay::make_monoid(scan_f, id);
      // After the scan, the last occurence of each ngh has the
      // aggregated weight.
      parlay::scan_inclusive_inplace(parlay::make_slice(filtered_edges),
                                     scan_mon);
      t.next("Scan time");

      size_t filtered_edges_size = filtered_edges.size();
      // Apply filter index to extract the last occurence of each edge.
      auto idx_f = [&](const EdgeTriple& e, size_t idx) {
        const auto & [ u, v, wgh ] = e;
        if (u == UINT_E_MAX) return false;
        if (idx < (filtered_edges_size - 1)) {
          const auto & [ next_u, next_v, next_wgh ] = filtered_edges[idx + 1];
          // Next edge is not the same as this one
          return (u != next_u) || (v != next_v);
        }
        return true;
      };
      auto inserts =
          parlay::filter_index(parlay::make_slice(filtered_edges), idx_f);
      t.next("Preprocess edges time");

      // Add the edge weights.
      insert_edges_batch_inplace(G, CG, inserts);
      t.next("Insert edges time");

      ApplyMergesToDendrogram(CG, starts, merge_seq);
    }

    tt.next("Overall time");
  }

}  // namespace aspen
