#include "ann/io.h"
#include "ann/vamana/neighbors.h"

namespace ann {

template <class Dist>
auto BuildFromComplete(parlay::sequence<ann::Tvec_point<typename Dist::T>>& pts,
    size_t k, cpam::commandLine& P) {
  size_t n = pts.size();
  size_t d = pts[0].coordinates.size();

  auto print_pt = [] (ann::Tvec_point<typename Dist::T>& pt) {
    for (size_t i=0; i<pt.coordinates.size(); ++i) {
      std::cout << pt.coordinates[i] << " ";
    }
    std::cout << std::endl;
  };

  if (k > pts.size()) {
    std::cerr << "k must be <= n: k=" << k << ", n=" << pts.size() << std::endl;
    exit(0);
  }

  auto distances = parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>>::from_function(n, [&] (size_t i) {
    return parlay::sequence<std::pair<float, uint32_t>>::uninitialized(n);
  });
  parlay::parallel_for(0, n, [&] (size_t i) {
    auto& pt = pts[i];
    for (size_t j=0; j<n; ++j) {
      if (i != j) {
        distances[i][j] = std::make_pair(sqrt(Dist::distance(pt, pts[j], d)), j);
        if (distances[i][j].first > 100000000000.0) {
          std::cout << "dist = " << distances[i][j].first << std::endl;
        }
      } else {
        distances[i][j] = std::make_pair(std::numeric_limits<float>::max(), i);
      }
    }
    parlay::sort_inplace(distances[i]);
  });

//  auto all_distances = parlay::tabulate(n*n, [&] (size_t i) {
//    auto vtx = i / n;
//    auto ngh = i % n;
//    return distances[vtx][ngh].first;
//  });
//  parlay::sort_inplace(all_distances);

  using vertex_id = uint32_t;
#ifdef USE_PAM
  using ngh_and_weight = std::pair<vertex_id, float>;
#else
  using ngh_and_weight = std::tuple<vertex_id, float>;
#endif
  using edge = std::pair<vertex_id, ngh_and_weight>;
  parlay::sequence<edge> edges(2*k*n);
  parlay::parallel_for(0, n, [&] (size_t i) {
    for (size_t j=0; j<k; ++j) {
      auto [wgh, ngh_id] = distances[i][j];
      edges[2*k*i + 2*j] = std::make_pair(i, std::make_tuple(ngh_id, wgh));
      edges[2*k*i + 2*j+1] = std::make_pair(ngh_id, std::make_tuple(i, wgh));
    }
  });

  parlay::sort_inplace(edges);
  auto index_seq = parlay::delayed_seq<std::pair<edge, size_t>>(edges.size(), [&] (size_t i) {
    return std::make_pair(edges[i], i);
  });

  auto removed_dups = parlay::filter(index_seq, [&] (auto e_idx) {
    const auto& [e, idx] = e_idx;
    const auto& [id, ngh_and_wgh] = e;
    const auto& [ngh, wgh] = ngh_and_wgh;
    if (id == edges.size()-1) return true;
    const auto& [next_id, next_ngh_and_wgh] = edges[idx+1];
    const auto& [next_ngh, next_wgh] = next_ngh_and_wgh;
    return (id == next_id) ? (ngh != next_ngh) : true;
  });

  // Awful, fix with filter_index in a bit.
  auto good_edges = parlay::tabulate(removed_dups.size(), [&] (size_t i) {
    return removed_dups[i].first;
  });

  // Perform conversion from distances -> similarities
  auto dists = parlay::delayed_seq<float>(good_edges.size(), [&] (size_t i) {
    return std::get<1>(good_edges[i].second);
  });
  float max_dist = parlay::reduce(dists, parlay::maxm<float>());
  std::cout << "Max overall dist = " << max_dist << std::endl;

  float min_dist = parlay::reduce(dists, parlay::minm<float>());
  float max_sim = 1.0 / (1.0 + min_dist);
  std::cout << "min_dist = " << min_dist << std::endl;

  parlay::parallel_for(0, good_edges.size(), [&] (size_t i) {
      float wgh = std::get<1>(good_edges[i].second);
      float sim = 1.0 / (1.0 + wgh);
//      std::get<1>(good_edges[i].second) = sim / max_sim;  // normalize to [0, 1]
//      float sim = max_dist - wgh + 1e-6;
      std::get<1>(good_edges[i].second) = sim / max_sim;  // normalize to [0, 1]
  });

  return good_edges;
}

template <class Dist>
auto BuildKNNGraph_Vamana(parlay::sequence<ann::Tvec_point<typename Dist::T>>& pts, size_t k,
                            cpam::commandLine P) {
  size_t max_deg = P.getOptionDoubleValue("-R", 75);
  size_t beam_size = P.getOptionDoubleValue("-L", 100);
  size_t beam_size_q = P.getOptionDoubleValue("-Q", std::max(k, beam_size));
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  bool do_check = P.getOptionValue("-check");
  size_t check_size = P.getOptionLongValue("-m", 10000);
  string metric = P.getOptionValue("-dist", "euclidean");

  parlay::internal::timer t("ANN", true);
  t.start();

  using T = typename Dist::T;
  using findex = ann::knn_index<Dist>;
  size_t d = pts[0].coordinates.size();
  findex I(max_deg, beam_size, alpha, d, metric, pts);
  I.build_index(true);
  t.next("Built index");

  if (k > pts.size()) {
    std::cerr << "k must be <= n: k=" << k << ", n=" << pts.size() << std::endl;
    exit(0);
  }

  auto permuted_ids =
      parlay::random_permutation<int>(static_cast<int>(pts.size()), time(NULL));

  // Optionally, measure k@k recall for various k.
  if (do_check) {
    std::vector<int> k_values = {1, 2, 4, 8, 16, 32, 64};
    k_values.push_back(k);

    auto q = parlay::sequence<ann::Tvec_point<T>>(check_size);
    parlay::parallel_for(0, check_size,
                         [&](size_t i) { q[i] = pts[permuted_ids[i]]; });

    for (auto k : k_values) {
      // Search
      I.searchNeighbors(q, beam_size_q, k);
      // Measure quality. For each of the searched point, compute the
      // true k-NN and measure k@k.
      double recall = I.template measureQuality<Dist>(q, k);
      std::cout << k << "@" << k << ": " << recall << std::endl;
    }
  }

  auto q = parlay::sequence<ann::Tvec_point<T>>(pts.size());
  parlay::parallel_for(0, q.size(), [&] (size_t i) {
    q[i].id = i;
  });
  //beam_size_q = std::max(2*k, beam_size_q);
  std::cout << "Searching for neighbors" << std::endl;
  I.searchNeighborsFromPointsInIndex(q, beam_size_q, k);
  std::cout << "Done." << std::endl;

//  auto min_dist = [&] (size_t i) {
//    auto& pt = q[i];
//    auto imap = parlay::delayed_seq<float>(pt.ngh.size(), [&] (size_t j) {
//      return distance(pt.coordinates.begin(), pts[pt.ngh[j]].coordinates.begin(), d);
//    });
//    return parlay::reduce(imap, parlay::minm<float>());
//  };
//  auto dist_imap =

  parlay::sequence<size_t> degs(pts.size());
  parlay::parallel_for(0, pts.size(), [&] (size_t i) { degs[i] = q[i].ngh.size(); });
  size_t total_edges = parlay::scan_inplace(parlay::make_slice(degs));
  std::cout << "Total edges = " << total_edges << std::endl;

  using vertex_id = uint32_t;
#ifdef USE_PAM
  using ngh_and_weight = std::pair<vertex_id, float>;
#else
  using ngh_and_weight = std::tuple<vertex_id, float>;
#endif
  using edge = std::pair<vertex_id, ngh_and_weight>;

  auto edges = parlay::sequence<edge>::uninitialized(2*total_edges);
  parlay::parallel_for(0, pts.size(), [&] (size_t i) {
    size_t offset = 2*degs[i];
    for (size_t j=0; j<q[i].ngh.size(); ++j) {
      int ngh = q[i].ngh[j];
      float dist = sqrt(Dist::distance(pts[i], pts[ngh], d));
      edges[offset + 2*j] = std::make_pair(i, ngh_and_weight(ngh, dist));
      edges[offset + 2*j+1] = std::make_pair(ngh, ngh_and_weight(i, dist));
    }
  });

  parlay::sort_inplace(edges);
  auto index_seq = parlay::delayed_seq<std::pair<edge, size_t>>(edges.size(), [&] (size_t i) {
    return std::make_pair(edges[i], i);
  });

  auto removed_dups = parlay::filter(index_seq, [&] (auto e_idx) {
    const auto& [e, idx] = e_idx;
    const auto& [id, ngh_and_wgh] = e;
    const auto& [ngh, wgh] = ngh_and_wgh;
    if (id == edges.size()-1) return true;
    const auto& [next_id, next_ngh_and_wgh] = edges[idx+1];
    const auto& [next_ngh, next_wgh] = next_ngh_and_wgh;
    return (id == next_id) ? (ngh != next_ngh) : true;
  });

  // Awful, fix with filter_index in a bit.
  auto good_edges = parlay::tabulate(removed_dups.size(), [&] (size_t i) {
    return removed_dups[i].first;
  });

  // Perform conversion from distances -> similarities
  auto dists = parlay::delayed_seq<float>(good_edges.size(), [&] (size_t i) {
      return std::get<1>(good_edges[i].second);
//    long idx = i;
//    auto our_id = good_edges[i].first;
//    bool ret = true;
//    int ct = 0;
//    while (idx >= 0) {
//      if (good_edges[idx].first != our_id) break;
//      if (ct >= 5) break;
//      ++ct;
//    }
//    return (ret) ? std::get<1>(good_edges[i].second) : -1.0;
  });
  float max_dist = parlay::reduce(dists, parlay::maxm<float>());
  float min_dist = parlay::reduce(dists, parlay::minm<float>());
  std::cout << "Max_dist = " << max_dist << std::endl;
  std::cout << "Min_dist = " << min_dist << std::endl;
  float max_sim = 1.0 / (1.0 + min_dist);
  parlay::parallel_for(0, good_edges.size(), [&] (size_t i) {
      float wgh = std::get<1>(good_edges[i].second);
      float sim = 1.0 / (1.0 + wgh);
      std::get<1>(good_edges[i].second) = sim / max_sim;  // normalize to [0, 1]
      //std::get<1>(good_edges[i].second) = 1 - (wgh/max_dist);
      // std::get<1>(good_edges[i].second) = 1.0 / (1.0 + log(1 + wgh));
      //std::get<1>(good_edges[i].second) = exp(-1 * (wgh/max_dist));
  });

  float new_max_sim = parlay::reduce(dists, parlay::maxm<float>());
  float min_sim = parlay::reduce(dists, parlay::minm<float>());
  std::cout << "Max_sim = " << new_max_sim << " old_max_sim = " << max_sim << std::endl;
  std::cout << "Min_sim = " << min_sim << std::endl;

//  for (size_t i=0; i<500; ++i) {
//    std::cout << good_edges[i].first << " " << std::get<0>(good_edges[i].second) << " " << std::get<1>(good_edges[i].second) << std::endl;
//  }
  std::cout << pts.size() << std::endl;

  return good_edges;
}

}  // namespace ann
