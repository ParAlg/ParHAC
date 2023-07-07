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

#ifndef BEAMSEARCH
#define BEAMSEARCH

#include <ann/utils/geometry.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <queue>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace ann {

extern bool report_stats;

using pid = std::pair<int, float>;

// returns true if F \setminus V = emptyset
bool intersect_nonempty(parlay::sequence<pid>& V, parlay::sequence<pid>& F) {
  for (size_t i = 0; i < F.size(); i++) {
    auto pred = [&](pid a) { return F[i].first == a.first; };
    if (parlay::find_if(V, pred) == V.end()) return true;
  }
  return false;
}

// will only be used when there is an element in F that is not in V
// hence the ``return 0" line will never be called
pid id_next(parlay::sequence<pid>& V, parlay::sequence<pid>& F) {
  for (size_t i = 0; i < F.size(); i++) {
    auto pred = [&](pid a) { return F[i].first == a.first; };
    if (parlay::find_if(V, pred) == V.end()) return F[i];
  }
  return std::make_pair(0, 0);
}

// for debugging
void print_seq(parlay::sequence<int> seq) {
  int fsize = seq.size();
  std::cout << "[";
  for (int i = 0; i < fsize; i++) {
    std::cout << seq[i] << ", ";
  }
  std::cout << "]" << std::endl;
}

// takes in two sorted sequences and returns a sorted union
parlay::sequence<pid> seq_union(parlay::sequence<pid>& P,
                                parlay::sequence<pid>& Q) {
  auto less = [&](pid a, pid b) { return a.second < b.second; };
  pid* first1 = P.begin();
  pid* last1 = P.end();
  pid* first2 = Q.begin();
  pid* last2 = Q.end();
  parlay::sequence<pid> result = parlay::sequence<pid>();
  while (true) {
    if (first1 == last1) {
      while (first2 != last2) {
        result.push_back(*first2);
        ++first2;
      }
      return result;
    } else if (first2 == last2) {
      while (first1 != last1) {
        result.push_back(*first1);
        ++first1;
      }
      return result;
    }
    if (less(*first1, *first2)) {
      result.push_back(*first1);
      ++first1;
    } else if (less(*first2, *first1)) {
      result.push_back(*first2);
      ++first2;
    } else {
      if (first1->first == first2->first) {
        result.push_back(*first1);
        ++first1;
        ++first2;
      } else {
        result.push_back(*first1);
        result.push_back(*first2);
        ++first1;
        ++first2;
      }
    }
  }
  return result;
}

template <class Dist>
parlay::sequence<pid> beam_search_2(Tvec_point<typename Dist::T>* p,
                                    parlay::sequence<Tvec_point<typename Dist::T>>& v,
                                    Tvec_point<typename Dist::T>* medoid, size_t beamSize,
                                    size_t d) {
  using T = typename Dist::T;
  // initialize data structures
  std::unordered_set<int> visited;
  auto q_comp = [] (const pid& l, const pid& r) {
    if (l.second < r.second) return true;
    if (l.second == r.second) return l.first < r.first;
    return false;
  };
  std::set<pid, decltype(q_comp)> frontier(q_comp);
  std::set<pid, decltype(q_comp)> beam(q_comp);

  auto ret = parlay::sequence<pid>();
  ret.reserve(beamSize);

  // the frontier starts with the medoid
  auto medoid_pair = std::make_pair(
      medoid->id,
      Dist::distance(*medoid, *p, d));
  frontier.insert(medoid_pair);
  beam.insert(medoid_pair);

  // terminate beam search when the entire frontier has been visited
  float last_dist = std::numeric_limits<float>::max();
  while (!frontier.empty()) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    pid currentPid = *(frontier.begin());
    frontier.erase(frontier.begin());

    ret.push_back(currentPid);
    visited.insert(currentPid.first);

    Tvec_point<T>* current = &(v[currentPid.first]);

    for (const int& ngh : current->out_nbh) {
      if (visited.find(ngh) == visited.end()) {
        float dist = Dist::distance(v[ngh], *p, d);
        // insert it into the beam
        if (dist < last_dist) {
          beam.insert(std::make_pair(ngh, dist));
          frontier.insert(std::make_pair(ngh, dist));
          visited.insert(ngh);
        }
      }
    }
    if (beam.size() > beamSize) {
      while (beam.size() > beamSize) {
        auto it = std::prev(beam.end());
        pid point = *it;
        beam.erase(it);
        auto f_it = frontier.find(point);
        if (f_it != frontier.end()) {
          frontier.erase(f_it);
        }
      }
      auto it = std::prev(beam.end());
      pid point = *it;
      last_dist = it->second;
    }
  }
  return ret;
}

template <typename Dist>
std::pair<parlay::sequence<pid>, parlay::sequence<pid>> beam_search(
    Tvec_point<typename Dist::T>* p, parlay::sequence<Tvec_point<typename Dist::T>>& v, Tvec_point<typename Dist::T>* medoid,
    size_t beamSize, size_t d) {
  using T = typename Dist::T;
  // initialize data structures
  parlay::sequence<pid> visited = parlay::sequence<pid>();
  parlay::sequence<pid> frontier = parlay::sequence<pid>();
  auto less = [&](pid a, pid b) { return a.second < b.second; };
  // the frontier starts with the medoid
  frontier.push_back(std::make_pair(
      medoid->id,
      Dist::distance(*medoid, *p, d)));
  // terminate beam search when the entire frontier has been visited
  while (intersect_nonempty(visited, frontier)) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    pid currentPid = (id_next(visited, frontier));
    Tvec_point<T>* current = &(v[currentPid.first]);
    auto g = [&](int a) {
      auto pred = [&](pid p) { return p.first == a; };
      if (parlay::find_if(visited, pred) == visited.end()) return true;
      return false;
    };
    auto candidates = parlay::filter(current->out_nbh, g);
    parlay::sequence<pid> pairCandidates =
        parlay::sequence<pid>(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++)
      pairCandidates[i] = std::make_pair(
          candidates[i], Dist::distance(v[candidates[i]],
                                  *p, d));
    auto sortedCandidates = parlay::sort(pairCandidates, less);
    frontier = seq_union(frontier, sortedCandidates);
    if (frontier.size() > beamSize)
      frontier.erase(frontier.begin() + beamSize, frontier.end());
    // add the node to the visited list
    visited.push_back(currentPid);
  }
  return std::make_pair(frontier, visited);
}

// searches every element in q starting from a randomly selected point
template <typename Dist>
void beamSearchRandom(parlay::sequence<Tvec_point<typename Dist::T>>& q,
                      parlay::sequence<Tvec_point<typename Dist::T>>& v, size_t beamSizeQ, size_t k,
                      size_t d) {
  using T = typename Dist::T;
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  // use a random shuffle to generate random starting points for each query
  size_t n = v.size();
  auto indices =
      parlay::random_permutation<int>(static_cast<int>(n), time(NULL));
  parlay::parallel_for(0, q.size(), [&](size_t i) {
    parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    Tvec_point<T>* start = &(v[indices[i]]);
    parlay::sequence<pid> beamElts;
    parlay::sequence<pid> visitedElts;
    std::pair<parlay::sequence<pid>, parlay::sequence<pid>> pairElts;
    pairElts = beam_search<Dist>(&(q[i]), v, start, beamSizeQ, d);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    // the first element of the frontier may be the point itself
    // if this occurs, do not report it as a neighbor
    if (beamElts[0].first == i) {
      for (int j = 0; j < k; j++) {
        neighbors[j] = beamElts[j + 1].first;
      }
    } else {
      for (int j = 0; j < k; j++) {
        neighbors[j] = beamElts[j].first;
      }
    }
    q[i].ngh = neighbors;
    if (report_stats) q[i].cnt = visitedElts.size();
  });
}

template <typename Dist>
void searchFromSingle(parlay::sequence<Tvec_point<typename Dist::T>>& q,
                      parlay::sequence<Tvec_point<typename Dist::T>>& v, size_t beamSizeQ, size_t k,
                      size_t d, Tvec_point<typename Dist::T>* medoid) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  parlay::parallel_for(0, q.size(), [&] (int i) {
    parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    parlay::sequence<pid> beamElts;
    parlay::sequence<pid> visitedElts;
    std::pair<parlay::sequence<pid>, parlay::sequence<pid>> pairElts;
    pairElts = beam_search<Dist>(&(q[i]), v, medoid, beamSizeQ, d);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    // the first element of the frontier may be the point itself
    // if this occurs, do not report it as a neighbor
    if (beamElts[0].first == i) {
      for (size_t j = 0; j < k; j++) {
        neighbors[j] = beamElts[j + 1].first;
      }
    } else {
      for (size_t j = 0; j < k; j++) {
        neighbors[j] = beamElts[j].first;
      }
    }
    q[i].ngh = neighbors;
    if (report_stats) q[i].cnt = visitedElts.size();
  });
}

template <typename Dist>
void searchFromPointsInIndex(parlay::sequence<Tvec_point<typename Dist::T>>& q,
                      parlay::sequence<Tvec_point<typename Dist::T>>& v, size_t beamSizeQ, size_t k,
                      size_t d, Tvec_point<typename Dist::T>* medoid) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  parlay::parallel_for(0, q.size(), [&] (int i) {
    parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    parlay::sequence<pid> beamElts;
    parlay::sequence<pid> visitedElts;
    std::pair<parlay::sequence<pid>, parlay::sequence<pid>> pairElts;
    pairElts = ann::beam_search<Dist>(&(v[q[i].id]), v, &(v[q[i].id]), beamSizeQ, d);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    if (beamElts.size() < k) {
      pairElts = beam_search<Dist>(&(v[q[i].id]), v, medoid, beamSizeQ, d);
      beamElts = pairElts.first;
      visitedElts = pairElts.second;
    }
    // the first element of the frontier may be the point itself
    // if this occurs, do not report it as a neighbor
    if (beamElts[0].first == i) {
      for (size_t j = 0; j < k; j++) {
        neighbors[j] = beamElts[j + 1].first;
      }
    } else {
      for (size_t j = 0; j < k; j++) {
        neighbors[j] = beamElts[j].first;
      }
    }
    q[i].ngh = neighbors;
    for (size_t j=0; j<q[i].ngh.size(); ++j) {
      assert(q[i].ngh[j] < v.size());
    }
    if (report_stats) q[i].cnt = visitedElts.size();
  });
}

}  // namespace ann

#endif