#pragma once

#include "aspen/aspen.h"

namespace parlay {

  constexpr const size_t _log_block_size = 10;
  constexpr const size_t _block_size = (1 << _log_block_size);
  
  inline size_t num_blocks(size_t n, size_t block_size) {
    if (n == 0)
      return 0;
    else
      return (1 + ((n)-1) / (block_size));
  }

  template <class In_Seq, class F>
  auto filter_index(In_Seq const& In, F f, flags fl = no_flag)
      -> sequence<typename In_Seq::value_type> {
    using T = typename In_Seq::value_type;
    size_t n = In.size();
    size_t l = num_blocks(n, _block_size);
    sequence<size_t> Sums(l);
    sequence<bool> Fl(n);
    parlay::internal::sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
      size_t r = 0;
      for (size_t j = s; j < e; j++) r += (Fl[j] = f(In[j], j));
      Sums[i] = r;
    });
    size_t m = parlay::scan_inplace(make_slice(Sums));
    sequence<T> Out = sequence<T>::uninitialized(m);
    parlay::internal::sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
        parlay::internal::pack_serial_at(
          make_slice(In).cut(s, e), make_slice(Fl).cut(s, e),
          make_slice(Out).cut(Sums[i], (i == l - 1) ? m : Sums[i + 1]));
    });
    return Out;
  }
}

namespace aspen {

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
  void UniteMerge(Graph& G, parlay::sequence<std::tuple<uintE, uintE,
      float>>& merge_seq, ClusteredGraph& CG, GetVertex& get_vertex) {

    using EdgeTriple = std::tuple<uintE, uintE, float>;

    // Sort merges based on the centers (semi-sort also ok).
    parlay::sort_inplace(make_slice(merge_seq));

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
    std::cout << "Number of merge targets = " << starts.size() << std::endl;
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
        // We will emit edges for both endpoints of the edge which is
        // why we emit 2*neighbor_size() edges
        edge_sizes[j] = 2 * get_vertex(ngh_id).out_degree();

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
    });
    std::cout << "Before deletions" << std::endl;

    // Scan to compute #edges we need to merge.
    size_t total_edges = parlay::scan_inplace(make_slice(edge_sizes));
    std::cout << "Total edges = " << total_edges << std::endl;
    auto edges = parlay::sequence<EdgeTriple>::uninitialized(total_edges);

    // How to perform edge deletions?

    auto all_deletions = parlay::sequence<std::pair<uintE, uintE>>::uninitialized(total_edges / 2);

    // Copy edges from trees to edges and deletions.
    parlay::parallel_for(0, merge_seq.size(), [&] (size_t i) {
      uintE center_id = std::get<0>(merge_seq[i]);
      uintE satellite_id = std::get<1>(merge_seq[i]);
      size_t k = 0;
      size_t offset = edge_sizes[i];
      size_t deletion_offset = offset / 2;
      // Map over every edge to a neighbor v, incident to the satellite.
      auto map_f = [&] (const uintE& u, const uintE& v, const float& wgh, size_t k) {
        bool v_active = CG.clusters[v].active;
        uintE merged_id = CG.clusters[v].current_id;
        // (1) v is active (itself a center) and it is not this center
        // (2) symmetry break to prevent sending a (u,v) edge between
        // two deactivated vertices twice to the activated targets.
        if ((merged_id != center_id) && ((v_active || (satellite_id < v)))) {
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
    });
    
    std::cout << "Deletions.size = " << all_deletions.size() << std::endl;

    auto deletions = parlay::filter(parlay::make_slice(all_deletions), [&] (const auto& e) {
      return e.first != UINT_E_MAX;
    });

    std::cout << "Deletions.size after filter = " << deletions.size() << std::endl;

    auto vertex_deletions = parlay::sequence<uintE>::from_function(merge_seq.size(), [&] (size_t i) {
      return std::get<1>(merge_seq[i]);
    });

    G.delete_vertices_inplace(vertex_deletions.size(), vertex_deletions.begin());

    // (1) First perform the deletions. This makes life easier later when we
    // perform the insertions.
    std::cout << "Num edges before = " << G.num_edges() << std::endl;
    G.delete_edges_batch_inplace(deletions.size(), deletions.begin());
    std::cout << "Num edges after = " << G.num_edges() << std::endl;

    // No references to deactivated vertices in any neighborlist of an
    // active vertex at this point. Next, we simply perform all
    // insertions (concurrently) in parallel. An alternate approach
    // that we used with PAM is to perform some semisorts + scans to
    // merge "same" edges, and then perform the insertions as a bulk
    // step. Need to understand which approach is faster.

    // Filter out empty edge triples.
    auto pred = [&](const EdgeTriple& e) {
      return std::get<0>(e) != UINT_E_MAX;
    };
    auto filtered_edges = parlay::filter(parlay::make_slice(edges), pred);

    // Sort triples lexicographically.
    parlay::sort_inplace(parlay::make_slice(filtered_edges));

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
    std::cout << "num inserts = " << inserts.size() << std::endl;

    // Add the edge weights.
    auto replace = [&] (const auto& a, const auto& b) { return a + b; };
    G.insert_edges_batch_inplace(inserts.size(), inserts.begin(), replace, /*sorted=*/ true);
    std::cout << "Num edges after inserts = " << G.num_edges() << std::endl;
  }

}  // namespace aspen
