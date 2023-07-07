#pragma once

#include "parlay/primitives.h"

namespace aspen {

struct empty {};

using vertex_id = uint32_t;
using edge_id = size_t;

// Writes the list of indices `i` where `Fl[i] == true` to range `Out`.
template <class Bool_Seq, class Out_Seq>
size_t pack_index_out(Bool_Seq const& Fl, Out_Seq&& Out) {
  using Idx_Type = typename std::remove_reference<Out_Seq>::type::value_type;
  auto identity = [](size_t i) { return (Idx_Type)i; };
  return parlay::pack_into(parlay::delayed_seq<Idx_Type>(Fl.size(), identity),
                           Fl, std::forward<Out_Seq>(Out));
}

template <class Idx_Type, class D, class F>
inline parlay::sequence<std::tuple<Idx_Type, D> > pack_index_and_data(
    F& f, size_t size) {
  auto id_seq =
      parlay::delayed_seq<std::tuple<Idx_Type, D> >(size, [&](size_t i) {
        return std::make_tuple((Idx_Type)i, std::get<1>(f[i]));
      });
  auto flgs_seq = parlay::delayed_seq<bool>(
      size, [&](size_t i) { return std::get<0>(f[i]); });

  return parlay::pack(id_seq, flgs_seq);
}

template <typename ET>
inline bool atomic_compare_and_swap(ET* a, ET oldval, ET newval) {
  if
    constexpr(sizeof(ET) == 1) {
      uint8_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint8_t*>(a), r_oval,
                                          r_nval);
    }
  else if
    constexpr(sizeof(ET) == 4) {
      uint32_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>(a),
                                          r_oval, r_nval);
    }
  else if
    constexpr(sizeof(ET) == 8) {
      uint64_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>(a),
                                          r_oval, r_nval);
    }
  else if
    constexpr(sizeof(ET) == 16) {
      __int128 r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap_16(reinterpret_cast<__int128*>(a),
                                             r_oval, r_nval);
    }
  else {
    std::cout << "Bad CAS Length" << sizeof(ET) << std::endl;
    exit(0);
  }
}

template <typename ET, typename F>
inline bool write_max(ET* a, ET b, F less) {
  ET c;
  bool r = 0;
  do
    c = *a;
  while (less(c, b) && !(r = atomic_compare_and_swap(a, c, b)));
  return r;
}

template <typename ET>
inline bool write_max(ET* a, ET b) {
  return write_max<ET>(a, b, std::less<ET>());
}

template <typename E, typename EV>
inline std::optional<E> fetch_and_add_threshold(E* a, EV b, EV max_v) {
  volatile E newV, oldV;
  oldV = *a;
  newV = oldV + b;
  while (oldV <= max_v) {
    if (atomic_compare_and_swap(a, oldV, newV)) return oldV;
    oldV = *a;
    newV = oldV + b;
  }
  return std::nullopt;
}

template <typename E, typename EV, typename F>
inline std::optional<E> fetch_and_add_threshold(E* a, EV b, EV max_v, F f) {
  volatile E newV, oldV;
  oldV = *a;
  newV = oldV + b;
  while (oldV <= max_v && f(oldV)) {
    if (atomic_compare_and_swap(a, oldV, newV)) return oldV;
    oldV = *a;
    newV = oldV + b;
  }
  return std::nullopt;
}

template <typename ET, typename F>
inline bool write_min(ET* a, ET b, F less) {
  ET c;
  bool r = 0;
  do
    c = *a;
  while (less(b, c) && !(r = atomic_compare_and_swap(a, c, b)));
  return r;
}

template <typename ET, typename F>
inline bool write_min(volatile ET* a, ET b, F less) {
  ET c;
  bool r = 0;
  do
    c = *a;
  while (less(b, c) && !(r = atomic_compare_and_swap(a, c, b)));
  return r;
}

template <typename ET, typename F>
inline bool write_min(std::atomic<ET>* a, ET b, F less) {
  ET c;
  bool r = 0;
  do
    c = a->load();
  while (less(b, c) && !(r = std::atomic_compare_exchange_strong(a, &c, b)));
  return r;
}

template <class _T, class Id>
struct MaxM {
  using T = std::pair<_T, Id>;
  static T identity() { return std::make_pair(0, 0); }
  static T add(T a, T b) {
    if (a.first > b.first) return a;
    return b;
  }
};

template <class _T, class Id>
struct MinM {
  using T = std::pair<_T, Id>;
  static T identity() { return std::make_pair(0, 0); }
  static T add(T a, T b) {
    if (a.first < b.first) return a;
    return b;
  }
};

}  // namespace aspen
