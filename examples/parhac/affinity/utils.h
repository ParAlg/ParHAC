#pragma once

namespace aspen {

template <class W>
struct Affinity_F {
  uintE* Parents;
  Affinity_F(uintE* _Parents) : Parents(_Parents) {}
  inline bool update(const uintE& s, const uintE& d, const W& w) const {
    Parents[d] = s;
    return 1;
  }
  inline bool updateAtomic(const uintE& s, const uintE& d, const W& w) const {
    return (cpam::utils::atomic_compare_and_swap(&Parents[d], UINT_E_MAX, s));
  }
  inline bool cond(const uintE& d) const { return (Parents[d] == UINT_E_MAX); }
};

inline uintE find_compress(uintE i, parlay::sequence<uintE>& parents) {
  uintE pathlen = 1;
  uintE j = i;
  if (parents[j] == j) return j;
  do {
    j = parents[j];
    pathlen++;
  } while (parents[j] != j);
  uintE tmp;
  while ((tmp=parents[i])>j) {
    parents[i] = j; i=tmp;
  }
  return j;
}

inline void unite_impl(uintE u_orig, uintE v_orig, parlay::sequence<uintE>& parents) {
  uintE u = u_orig;
  uintE v = v_orig;
  while(1) {
    u = find_compress(u,parents);
    v = find_compress(v,parents);
    if(u == v) break;
    else if (u > v && parents[u] == u && atomic_compare_and_swap(&parents[u],u,v)) {
      break;
    }
    else if (v > u && parents[v] == v && atomic_compare_and_swap(&parents[v],v,u)) {
      break;
    }
  }
}

template <class Seq>
inline std::pair<size_t, size_t> largest_cc(Seq& labels) {
  size_t n = labels.size();
  parlay::parallel_for(0, n, [&] (size_t i) {
    find_compress(i, labels);
  });
  // could histogram to do this in parallel.
  auto flags = parlay::sequence<uintE>::from_function(n + 1, [&](size_t i) { return 0; });
  for (size_t i = 0; i < n; i++) {
    flags[labels[i]] += 1;
  }
  size_t sz = parlay::reduce(parlay::make_slice(flags), parlay::maxm<uintE>());
  size_t id = UINT_E_MAX;
  for (size_t i=0; i<n; i++) {
    if (flags[labels[i]] == sz) {
      id = i;
      break;
    }
  }
  std::cout << "# largest_cc has size: " << sz << "\n";
  return {sz, id};
}

template <class Seq>
inline size_t num_cc(Seq& labels) {
  size_t n = labels.size();
  auto flags =
      parlay::sequence<uintE>::from_function(n + 1, [&](size_t i) { return 0; });
  parlay::parallel_for(0, n, [&](size_t i) {
    if (!flags[labels[i]]) {
      flags[labels[i]] = 1;
    }
  });
  parlay::scan_inplace(flags);
  std::cout << "# n_cc = " << flags[n] << "\n";
  return flags[n];
}

}  // namespace aspen
