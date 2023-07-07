#pragma once

#include "build.h"
#include "integer_weighted_edge_encoder.h"
#include "vertex_subset.h"

#include <type_traits>
#include <cpam/parse_command_line.h>

namespace aspen {

template <class weight>
struct symmetric_graph {
  struct edge_entry {
    using key_t = vertex_id;  // a vertex_id
    using val_t = weight;     // placeholder
    static inline bool comp(key_t a, key_t b) { return a < b; }
    using entry_t = std::tuple<key_t, val_t>;
  };

#ifdef USE_PAM
  using edge_tree = pam_map<edge_entry>;
#else
// map can either be uncompressed or compressed (using difference encoding, or
// another suitable compression scheme).
#ifdef USE_DIFF_ENCODING
  using edge_tree = typename std::conditional<std::is_same<uintE, weight>::value,
        cpam::pam_map<edge_entry, 128, cpam::integer_weighted_edge_encoder>,
        cpam::diff_encoded_map<edge_entry, 128>>::type;
#else
  using edge_tree = cpam::pam_map<edge_entry, 128>;
#endif
#endif

  using edge_node = typename edge_tree::node;
  using edge_gc = typename edge_tree::GC;

  struct vertex_entry {
    using key_t = vertex_id;
    using val_t = edge_node*;
    using aug_t = edge_id;
    static inline bool comp(key_t a, key_t b) { return a < b; }
    static aug_t get_empty() { return 0; }
    static aug_t from_entry(const key_t& k, const val_t& v) {
      return edge_tree::Tree::size(v);
    }
    static aug_t combine(aug_t a, aug_t b) { return a + b; }
    using entry_t = std::tuple<key_t, val_t>;
  };
#ifdef USE_PAM_UPPER
  using vertex_tree = aug_map<vertex_entry>;
#else
#ifdef USE_DIFF_ENCODING
//  using vertex_tree = cpam::diff_encoded_aug_map<vertex_entry, 32>;
  using vertex_tree = cpam::aug_map<vertex_entry>;
#else
  using vertex_tree = cpam::aug_map<vertex_entry>;
#endif
#endif
  using vertex_node = typename vertex_tree::node;
  using vertex_gc = typename vertex_tree::GC;

#ifdef USE_PAM
  using ngh_and_weight = std::pair<vertex_id, weight>;
#else
  using ngh_and_weight = std::tuple<vertex_id, weight>;
#endif
  using edge = std::pair<vertex_id, ngh_and_weight>;

  struct neighbors {
    vertex_id id;
    edge_node* edges;

    neighbors(vertex_id id, edge_node* edges) : id(id), edges(edges) {}

    template <class F, class G>
    void copy(size_t offset, F& f, G& g) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        auto val = f(id, ngh, wgh);
        g(ngh, offset + i, val);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class F>
    void map_index(F& f) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        f(id, ngh, wgh, i);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class F>
    void map(F& f) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        f(id, ngh, wgh);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class _T>
    struct Add {
      using T = _T;
      static T identity() { return 0; }
      static T add(T a, T b) { return a + b; }
    };

    // Count the number of neighbors satisfying the predicate p.
    template <class P>
    size_t count(P& p) {
      edge_tree tree;
      tree.root = edges;
      auto map_f = [&](const auto& et) -> size_t {
        auto[ngh, wgh] = et;
        return p(id, ngh, wgh);
      };
      auto addm = Add<size_t>();
      auto ct = edge_tree::map_reduce(tree, map_f, addm);
      tree.root = nullptr;
      return ct;
    }

    // Count the number of neighbors satisfying the predicate p.
    template <class M, class R>
    typename R::T map_reduce(M& m, R& r) {
      using T = typename R::T;
      edge_tree tree;
      tree.root = edges;
      auto map_f = [&](const auto& et) -> T {
        auto [ngh, wgh] = et;
        return m(id, ngh, wgh);
      };
      auto ret = edge_tree::map_reduce(tree, map_f, r);
      tree.root = nullptr;
      return ret;
    }

    template <class F, class C>
    void map_cond(F& f, C& c) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        return f(id, ngh, wgh);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_cond_par(tree, map_f, c);
      tree.root = nullptr;
    }

    template <class F>
    void foreach_cond(F& f) {
      auto map_f = [&](const auto& et) -> bool {
        return f(id, std::get<0>(et), std::get<1>(et));
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_cond(tree, map_f);
      tree.root = nullptr;
    }
  };

  struct vertex {
    vertex_id id;
    edge_node* edges;
    size_t out_degree() {
      return edge_tree::size(edges);
    }
    size_t in_degree() { return out_degree(); }
    size_t ref_cnt() {
      edge_tree tree;
      tree.root = edges;
      auto sz = tree.ref_cnt();
      tree.root = nullptr;
      return sz;
    }
    auto out_neighbors() const { return neighbors(id, edges); }
    auto in_neighbors() const { return neighbors(id, edges); }
    vertex(vertex_id id, edge_node* edges) : id(id), edges(edges) {}
    vertex() : id(std::numeric_limits<vertex_id>::max()), edges(nullptr) {}
    // todo: map, etc, necessary for edgeMap.
  };

  using maybe_vertex = std::optional<vertex>;
  using weight_type = weight;
  using SymGraph = symmetric_graph<weight>;

  vertex_tree V;
  size_t n;  // num vertices

  // Build from a static graph.
  symmetric_graph(std::tuple<size_t, size_t, uintT*, uintE*>& parsed_graph, cpam::commandLine& P) {
    auto edges = graph_to_edges<weight_type>(parsed_graph, P);
    auto filtered_edges = parlay::filter(parlay::make_slice(edges), [&] (const auto& et) {
      return et.first != std::get<0>(et.second);
    });
    edges.clear();
    std::cout << "Edges.size = " << filtered_edges.size() << std::endl;
    SymGraph::reserve(std::get<0>(parsed_graph), std::get<1>(parsed_graph));
    V = from_edges(filtered_edges);
    n = std::get<0>(parsed_graph);
    size_t m = std::get<1>(parsed_graph);
    // if not binary
    free_array(std::get<2>(parsed_graph), n);
    free_array(std::get<3>(parsed_graph), m);
  }

  // Build from a static weighted graph.
  symmetric_graph(std::tuple<size_t, size_t, uintT*, std::tuple<uintE, float>*>& parsed_graph) {
    auto edges = graph_to_edges<weight_type>(parsed_graph);
    auto filtered_edges = parlay::filter(parlay::make_slice(edges), [&] (const auto& et) {
      return et.first != std::get<0>(et.second);
    });
    edges.clear();
    std::cout << "Edges.size = " << filtered_edges.size() << std::endl;
    SymGraph::reserve(std::get<0>(parsed_graph), std::get<1>(parsed_graph));
    V = from_edges(filtered_edges);
    n = std::get<0>(parsed_graph);
    size_t m = std::get<1>(parsed_graph);
    // if not binary
    free_array(std::get<2>(parsed_graph), n);
    free_array(std::get<3>(parsed_graph), m);
  }

  // Build from weighted edge list. Assumes that edges have already
  // been filtered.
  symmetric_graph(size_t _n, parlay::sequence<edge>& edges, cpam::commandLine& P) {
    std::cout << "Edges.size = " << edges.size() << std::endl;
    SymGraph::reserve(_n, edges.size());
    V = from_edges(edges);
    n = _n;
  }

  // Set from a provided root (no ref-ct bump)
  symmetric_graph(vertex_node* root) {
    set_root(root);
  }

  // Build from a sequence of edges.
  symmetric_graph(parlay::sequence<edge>& edges) { V = from_edges(edges); }

  symmetric_graph(vertex_tree&& _V) : V(std::move(_V)) {}

  symmetric_graph() { V.root = nullptr; }

  vertex_tree& get_vertices() { return V; }

  // TODO: cleanup. clear_root not needed?
  // need to store "n"? Can store inside of a special "graph_root" node, with an extra 4-bytes. Then storage for a symmetric graph is just a single pointer.
  void clear_root() { V.root = nullptr; }

  // Definitely bad. Fix.
  vertex_node* get_root() { return V.root; }

  size_t ref_cnt() {
    return V.ref_cnt();
  }

  void set_root(vertex_node* root) {
    V.root = root;
    //std::cout << "Set root to = " << root << " ref_cnt = " << ref_cnt() << std::endl;
  }

  // Note that it's important to use n and not V.size() here.
  // TODO(laxmand): should add empty nodes in vtx-tree for 0-degree vertices?
  size_t num_vertices() { return n; }
  //size_t num_vertices() { return V.size(); }

  void set_vertices(size_t _n) { n = _n; }

  size_t num_edges() { return V.aug_val(); }

  vertex get_vertex(vertex_id v) {
    auto opt = V.find(v);
    if (opt.has_value()) {
      const auto& in_opt = *opt;
#ifdef DEBUG
      auto ref_cnt = edge_tree::Tree::ref_cnt(in_opt);
      assert(ref_cnt == 1);
#endif
      return vertex(v, in_opt);
    }
    return vertex(v, nullptr);
  }

  template <class F>
  void map_vertices(const F& f) {
    using entry_t = typename vertex_entry::entry_t;
    auto map_f = [&](const entry_t& vtx_entry, size_t i) {
      const vertex_id& v = std::get<0>(vtx_entry);
      auto vtx = vertex(v, std::get<1>(vtx_entry));
      f(vtx);
    };
    vertex_tree::foreach_index(V, map_f, 0, 1);
  }

  static vertex_tree from_edges(parlay::sequence<edge>& edges) {

    for (size_t i=0; i<std::min(edges.size(), (size_t)100); ++i) {
      std::cout << edges[i].first << " " << std::get<0>(edges[i].second) << " " << std::get<1>(edges[i].second) << std::endl;
    }
    auto reduce = [&](parlay::slice<ngh_and_weight*, ngh_and_weight*> R) {
      auto tree = edge_tree(R.begin(), R.begin() + R.size());
      auto root = tree.root;
      tree.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(root) == 1);
      return root;
    };
    vertex_tree vertices;
    return vertex_tree::multi_insert_reduce(vertices, edges, reduce);
  }

  // Reserve space for n vertices and m edges.
  static void reserve(size_t n, size_t m) {
    vertex_tree::reserve(n);
    edge_tree::reserve(m);
  }

  //  parlay::sequence<edge_tree> fetch_all_vertices() {
  //    timer t; t.start();
  //    size_t n = num_vertices();
  //    auto vtxs = parlay::sequence<edge_tree>(n);
  //    auto map_f = [&] (const vertex_entry& entry, size_t ind) {
  //      const auto& v = std::get<0>(entry);
  ////      const auto& el = std::get<1>(entry);
  ////      vtxs[v] = el;
  //    };
  //    map_vertices(map_f);
  //    t.next("fetch time");
  ////    cout << "fetched" << endl;
  //    return vtxs;
  //  }

  struct AddFourTup {
    using T = std::tuple<size_t, size_t, size_t, size_t>;
    static T identity() { return {0, 0, 0, 0}; }
    static T add(T a, T b) {
      return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b),
              std::get<2>(a) + std::get<2>(b), std::get<3>(a) + std::get<3>(b)};
    }
  };

  inline auto get_tree_sizes(const std::string& graphname, const std::string& mode) {
    auto noop = [](const auto& q) { return 0; };
    size_t vertex_tree_bytes = V.size_in_bytes(noop);
    auto[outer_internal, outer_leafs, outer_leaf_sizes] = V.node_stats();
    std::cout << "Num vertex_tree outer_nodes = " << outer_internal
              << " Num vertex_tree inner_nodes = " << outer_leafs
              << " Total vertex tree leaf sizes = " << outer_leaf_sizes
              << std::endl;

    auto map_f =
        [&](const auto& et) -> std::tuple<size_t, size_t, size_t, size_t> {
      auto[key, root] = et;
      if (root != nullptr) {
        edge_tree tree;
        tree.root = root;
        auto sz = tree.size_in_bytes(noop);
        auto[internal, leafs, leaf_sizes] = tree.node_stats();
        tree.root = nullptr;
        return {sz, internal, leafs, leaf_sizes};
      }
      return {0, 0, 0, 0};
    };

    auto addm = AddFourTup();
    auto[edge_tree_bytes, inner_internal, inner_leaf, inner_sizes] =
        vertex_tree::map_reduce(V, map_f, addm);

    std::cout << "Num edge_trees outer_nodes = " << inner_internal
              << " Num edge_trees inner_nodes = " << inner_leaf
              << " Total edge_trees leaf sizes = " << inner_sizes << std::endl;

    std::cout << "Edge trees size in bytes = " << edge_tree_bytes << std::endl;
    std::cout << "Vertex tree size in bytes = " << vertex_tree_bytes
              << std::endl;

    size_t total_bytes = edge_tree_bytes + vertex_tree_bytes;
    size_t m = num_edges();

    std::cout << "csv: " << graphname << "," << n << "," << m << "," << mode
              << "," << total_bytes << "," << vertex_tree_bytes << ","
              << edge_tree_bytes << std::endl;
    return std::make_tuple(total_bytes, vertex_tree_bytes, edge_tree_bytes);
  }

  void print_stats() {
#ifndef USE_PAM
    size_t sz = 0;
    size_t edges_bytes = 0;
    auto f = [&](const auto& et) {
      const auto& incident = std::get<1>(et);
      auto noop = [](const auto& q) { return 0; };
      size_t edges_size = incident.size();
      edges_bytes += incident.size_in_bytes(noop);
      //      if (edges_size < 2*cpam::utils::B) {
      //	assert(incident.root_is_compressed());
      //      }
    };
    vertex_tree::foreach_seq(V, f);
    std::cout << "num_edges = " << sz << std::endl;
    std::cout << "edges_size = " << edges_bytes << std::endl;
#else

#endif
  }

  /* ============= Update Operations ================ */

  template <class Edge>
  void sort_updates(Edge* edges, size_t m) const {
    size_t vtx_bits = parlay::log2_up(n);
    auto edge_to_long = [vtx_bits](Edge e) -> size_t {
      return (static_cast<size_t>(std::get<0>(e)) << vtx_bits) +
             static_cast<size_t>(std::get<1>(e));
    };

    // Only apply integer sort if it will be work-efficient
    if (n <= (m * parlay::log2_up(m))) {
      // cout << "running integer sort: " << n
      //      << " and mm = " << (m * parlay::log2_up(m)) << endl;
      parlay::integer_sort_inplace(parlay::make_slice(edges, edges + m),
                                   edge_to_long);
    } else {
      // cout << "running sample sort" << endl;
      parlay::sort_inplace(parlay::make_slice(edges, edges + m),
                           std::less<Edge>());
    }
  }

  template <class VertexId>
  void sort_vertex_ids(VertexId* vertices, size_t m) const {
    // Only apply integer sort if it will be work-efficient
    if (n <= (m * parlay::log2_up(m))) {
      parlay::integer_sort_inplace(parlay::make_slice(vertices, vertices + m));
    } else {
      // cout << "running sample sort" << endl;
      parlay::sort_inplace(parlay::make_slice(vertices, vertices + m));
    }
  }

  // Trying different versions of batch insert:
  // (1) insert_edges_batch_1:
  //

  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  // assuming that an edge already shows up in both directions
  // Let the caller delete edges.
  template <class Edge>
  void insert_edges_batch_1(size_t m, Edge* edges, bool sorted = false,
                            bool remove_dups = false,
                            size_t nn = std::numeric_limits<size_t>::max(),
                            bool run_seq = false) {
    // sort edges by their endpoint
    timer t;
    t.start();
    auto E_orig = parlay::make_slice(edges, edges + m);
    parlay::sequence<Edge> E_alloc;
    if (!sorted) {
      sort_updates(edges, m);
      t.next("sort time");
    }

    if (remove_dups) {
      // can perform combining here if desired
      auto bool_seq = parlay::delayed_seq<bool>(E_orig.size(), [&](size_t i) {
        return (i == 0 || E_orig[i] != E_orig[i - 1]);
      });
      E_alloc = parlay::pack(E_orig, bool_seq);
      m = E_alloc.size();
      std::cout << "Remove dups: orig size = " << E_orig.size()
                << " new size = " << E_alloc.size() << std::endl;
      // t.next("remove dups time");
    }

    auto E = (remove_dups) ? parlay::make_slice(E_alloc) : E_orig;

    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
    });
    auto starts = parlay::pack_index<size_t>(start_im);
    size_t num_starts = starts.size();

// build new vertices for each start
#ifdef USE_PAM_UPPER
    using KV = std::pair<uintE, edge_node*>;
#else
    using KV = std::tuple<uintE, edge_node*>;
#endif

    constexpr const size_t stack_size = 20;
    KV kv_stack[stack_size];
    KV* new_verts = kv_stack;
    if (num_starts > stack_size) {
      new_verts = cpam::utils::new_array_no_init<KV>(num_starts);
    }

    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> ngh_and_weight {
      return ngh_and_weight(std::get<1>(E[i]), weight());
    });
    t.next("prepare time");

    // std::cout << "Num starts = " << num_starts << " E.size = " << E.size() <<
    // std::endl;
    parlay::parallel_for(
        0, num_starts,
        [&](size_t i) {
          size_t off = starts[i];
          size_t deg = ((i == (num_starts - 1)) ? m : starts[i + 1]) - off;
          uintE v = std::get<0>(E[starts[i]]);

          auto tree = edge_tree(Vals.begin() + off, Vals.begin() + off + deg);
          new_verts[i] = {v, tree.root};
          tree.root = nullptr;
        },
        1);
    t.next("build vertices time");

    auto replace = [&](edge_node* prev_tree, edge_node* new_tree) {
      auto t1 = edge_tree();
      t1.root = prev_tree;
      auto t2 = edge_tree();
      t2.root = new_tree;
      if (t1.root) assert(t1.ref_cnt() == 1);
      if (t2.root) assert(t2.ref_cnt() == 1);
      auto t3 = edge_tree::map_union(std::move(t1), std::move(t2));
      assert(t3.ref_cnt() == 1);
      edge_node* ret = t3.root;
      t3.root = nullptr;
      return ret;
    };

    auto new_verts_seq = parlay::make_slice(new_verts, new_verts + num_starts);
    V = vertex_tree::multi_insert_sorted(std::move(V), new_verts_seq, replace);
    t.next("multiinsert time");
  }

  template <class Func>
  void iterate_seq(const Func& f) {
    V.iterate_seq(f);
  }


  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  // assuming that an edge already shows up in both directions
  // Let the caller delete edges.
  template <class Edge>
  SymGraph insert_edges_batch_2(size_t m, Edge* edges,
                            bool sorted = false, bool remove_dups = false,
                            size_t nn = std::numeric_limits<size_t>::max(),
                            bool run_seq = false) {
    // std::cout << "Start of insert, root = : " << V.root << " ref_cnt = " << V.ref_cnt() << std::endl;
    // sort edges by their endpoint
    timer pt("Insert", false);
    timer t("Insert", false);
    auto E_orig = parlay::make_slice(edges, edges + m);
    parlay::sequence<Edge> E_alloc;

    sort_updates(edges, m);
    t.next("insert: sort time");
    auto E = E_orig;

    if (remove_dups) {
      // can perform combining here if desired
      auto bool_seq = parlay::delayed_seq<bool>(E_orig.size(), [&](size_t i) {
        return (i == 0 || E_orig[i] != E_orig[i - 1]);
      });
      E_alloc = parlay::pack(E_orig, bool_seq);
      m = E_alloc.size();
      // std::cout << "Remove dups: orig size = " << E_orig.size()
      //           << " new size = " << E_alloc.size() << std::endl;
      t.next("insert: remove dups time");
      E = parlay::make_slice(E_alloc);
    }

    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
    });
    auto I = parlay::pack_index<size_t>(start_im);
    t.next("insert: Generate starts");

    // At this point have the (key, slice<uintE>) pairs ready to go.
    // VE = slice<uintE>
    // apply multi_update_sorted

    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> ngh_and_weight {
      return ngh_and_weight(std::get<1>(E[i]), weight());
    });
    t.next("insert: Generate vals");

    using key_type = uintE;
    using value_type = parlay::slice<ngh_and_weight*, ngh_and_weight*>;
    using KV = std::pair<key_type, value_type>;
    auto elts = parlay::sequence<KV>::from_function(I.size(), [&] (size_t i) {
      auto start = I[i];
      auto end = (i == I.size()-1) ? m : I[i+1];
      return KV(get<0>(E[start]), parlay::make_slice(Vals.begin() + start, Vals.begin() + end));
    });
    t.next("insert: Generate KV-pairs");

    auto replace = [&] (const auto& a, const auto& b) { return b; };
    auto combine_op = [&] (edge_node* cur, value_type incoming) {
      edge_tree t;
      t.root = cur;
      size_t rc = t.ref_cnt();
//      if (rc != 1) { //  currently always true! (when using std::move(t) below). need to fix.
//        std::cout << "rc = " << rc << std::endl;
//        exit(0);
//      }
//      auto ret = edge_tree::multi_insert_sorted(std::move(t), incoming, replace);
      auto ret = edge_tree::multi_insert_sorted(t, incoming, replace);
      t.root = nullptr;

      auto r = ret.root;
      ret.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(r) == 1);
      return r;
    };
    auto map_op = [] (value_type incoming) {
      auto tree = edge_tree(incoming.begin(), incoming.begin() + incoming.size());
      auto root = tree.root;
      tree.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(root) == 1);
      return root;
    };
    // std::cout << "Before multi_insert_sorted. " << std::endl;
    // std::cout << "Inserting into root: " << V.root << " ref_cnt = " << V.ref_cnt() << std::endl;
    //V = vertex_tree::multi_insert_sorted_map(std::move(V), elts, combine_op, map_op);
    // std::cout << "Intermediate time = " << tt << std::endl;
    auto new_V = vertex_tree::multi_insert_sorted_map(V, elts, combine_op, map_op);

    return SymGraph(std::move(new_V));
  }

  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  // assuming that an edge already shows up in both directions
  // Let the caller delete edges.
  template <class Edge, class Replace>
  void insert_edges_batch_3(size_t m, Edge* edges, Replace& replace,
                            bool sorted = false, bool remove_dups = false,
                            size_t nn = std::numeric_limits<size_t>::max(),
                            bool run_seq = false) {

    if (m == 0) return;

    // std::cout << "Start of insert, root = : " << V.root << " ref_cnt = " << V.ref_cnt() << std::endl;
    // sort edges by their endpoint
    timer pt("Insert", false);
    timer t("Insert", false);
    auto E_orig = parlay::make_slice(edges, edges + m);
    parlay::sequence<Edge> E_alloc;

    sort_updates(edges, m);
    t.next("insert: sort time");
    auto E = E_orig;

    if (remove_dups) {
      // can perform combining here if desired
      auto bool_seq = parlay::delayed_seq<bool>(E_orig.size(), [&](size_t i) {
        return (i == 0 || E_orig[i] != E_orig[i - 1]);
      });
      E_alloc = parlay::pack(E_orig, bool_seq);
      m = E_alloc.size();
      std::cout << "Remove dups: orig size = " << E_orig.size()
                << " new size = " << E_alloc.size() << std::endl;
      t.next("insert: remove dups time");
      E = parlay::make_slice(E_alloc);
    }

    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
    });
    auto I = parlay::pack_index<size_t>(start_im);
    t.next("insert: Generate starts");

    // At this point have the (key, slice<uintE>) pairs ready to go.
    // VE = slice<uintE>
    // apply multi_update_sorted

    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> ngh_and_weight {
      return ngh_and_weight(std::get<1>(E[i]), std::get<2>(E[i]));
    });
    t.next("insert: Generate vals");

    using key_type = uintE;
    using value_type = parlay::slice<ngh_and_weight*, ngh_and_weight*>;
    using KV = std::pair<key_type, value_type>;
    auto elts = parlay::sequence<KV>::from_function(I.size(), [&] (size_t i) {
      auto start = I[i];
      auto end = (i == I.size()-1) ? m : I[i+1];
      return KV(get<0>(E[start]), parlay::make_slice(Vals.begin() + start, Vals.begin() + end));
    });
    t.next("insert: Generate KV-pairs");

    //auto replace = [&] (const auto& a, const auto& b) { return b; };
    auto combine_op = [&] (edge_node* cur, value_type incoming) {
      edge_tree t;
      t.root = cur;
      auto ret = edge_tree::multi_insert_sorted(std::move(t), incoming, replace);
      auto r = ret.root;
      ret.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(r) == 1);
      return r;
    };
    auto map_op = [] (value_type incoming) {
      auto tree = edge_tree(incoming.begin(), incoming.begin() + incoming.size());
      auto root = tree.root;
      tree.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(root) == 1);
      return root;
    };
    // std::cout << "Before multi_insert_sorted. " << std::endl;
    // std::cout << "Inserting into root: " << V.root << " ref_cnt = " << V.ref_cnt() << std::endl;
    //V = vertex_tree::multi_insert_sorted_map(std::move(V), elts, combine_op, map_op);
    // std::cout << "Intermediate time = " << tt << std::endl;
    V = vertex_tree::multi_insert_sorted_map(std::move(V), elts, combine_op, map_op);
  }

  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  // assuming that an edge already shows up in both directions
  // Let the caller delete edges.
  template <class VertexId>
  void delete_vertices_inplace(size_t m, VertexId* vertices,
                            bool sorted = false, bool remove_dups = false) {
    if (m == 0) return;
    // Sort if needed
    timer t("DeleteVertices", false);
    auto V_orig = parlay::make_slice(vertices, vertices + m);
    parlay::sequence<VertexId> E_alloc;

    sort_vertex_ids(vertices, m);
    t.next("delete: sort time");

    if (remove_dups) {
      // can perform combining here if desired
      auto bool_seq = parlay::delayed_seq<bool>(V_orig.size(), [&](size_t i) {
        return (i == 0 || V_orig[i] != V_orig[i - 1]);
      });
      E_alloc = parlay::pack(V_orig, bool_seq);
      m = E_alloc.size();
      // std::cout << "Remove dups: orig size = " << V_orig.size()
      //           << " new size = " << E_alloc.size() << std::endl;
      t.next("delete: remove dups time");
    }
    auto vtxs = (remove_dups) ? parlay::make_slice(E_alloc) : V_orig;

    auto found_values = vertex_tree::multi_find_sorted(V, vtxs);
    t.next("vtx delete: multifind-sorted time");

    V = vertex_tree::multi_delete_sorted(std::move(V), vtxs);
    t.next("vtx delete: multidelete-sorted time");

    parlay::parallel_for(0, found_values.size(), [&] (size_t i) {
      edge_node* e = found_values[i];
      assert(e);
      if (e) {
        edge_tree tree;
        assert(tree.ref_cnt() == 1);
        tree.root = e;  // let this call clear().
      }
    });
    std::cout << "Finished deleting found nodes." << std::endl;
    t.next("vtx delete: clear edge trees time");
  }


  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  // assuming that an edge already shows up in both directions
  // Let the caller delete edges.
  template <class Edge>
  void delete_edges_batch_3(size_t m, Edge* edges,
                            bool sorted = false, bool remove_dups = false,
                            size_t nn = std::numeric_limits<size_t>::max(),
                            bool run_seq = false) {
    if (m == 0) return;

    // sort edges by their endpoint
    timer t("Delete-Edges", true);
    auto E_orig = parlay::make_slice(edges, edges + m);
    parlay::sequence<Edge> E_alloc;

    sort_updates(edges, m);
    t.next("delete: sort time");

    if (remove_dups) {
      // can perform combining here if desired
      auto bool_seq = parlay::delayed_seq<bool>(E_orig.size(), [&](size_t i) {
        return (i == 0 || E_orig[i] != E_orig[i - 1]);
      });
      E_alloc = parlay::pack(E_orig, bool_seq);
      m = E_alloc.size();
      // std::cout << "Remove dups: orig size = " << E_orig.size()
      //           << " new size = " << E_alloc.size() << std::endl;
      t.next("delete: remove dups time");
    }
    auto E = (remove_dups) ? parlay::make_slice(E_alloc) : E_orig;

    // pack starts
    auto start_im = parlay::delayed_seq<size_t>(m, [&](size_t i) {
      return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
    });
    auto I = parlay::pack_index<size_t>(start_im);
    t.next("delete: Generate starts");

    // At this point have the (key, slice<uintE>) pairs ready to go.
    // VE = slice<uintE>
    // apply multi_update_sorted

    auto Vals = parlay::tabulate(E.size(), [&](size_t i) -> uintE {
      return std::get<1>(E[i]);
    });
    t.next("delete: Generate vals");

    using key_type = uintE;
    using value_type = parlay::slice<uintE*, uintE*>;
    using KV = std::pair<key_type, value_type>;
    auto elts = parlay::sequence<KV>::from_function(I.size(), [&] (size_t i) {
      auto start = I[i];
      auto end = (i == I.size()-1) ? m : I[i+1];
      return KV(get<0>(E[start]), parlay::make_slice(Vals.begin() + start, Vals.begin() + end));
    });
    t.next("delete: Generate KV-pairs");
    // std::cout << "Num vertices updated: " << I.size() << std::endl;

    auto combine_op = [&] (edge_node* cur, value_type incoming) {
      assert(edge_tree::Tree::ref_cnt(cur) == 1);
      edge_tree t;
      t.root = cur;
      auto ret = edge_tree::multi_delete_sorted(std::move(t), incoming);
      auto r = ret.root;
      ret.root = nullptr;
      // TODO: need to handle keys w/o any neighbors (fully deleted)
      assert((r == nullptr) || (edge_tree::Tree::ref_cnt(r) == 1));
      return r;
    };
    std::cout << "Calling multi_update" << std::endl;
    vertex_tree::multi_update_sorted_inplace(V, elts, combine_op);
    std::cout << "Finished multi_update" << std::endl;

    t.next("delete: multiinsert time");
  }



  void insert_vertex_block(size_t vn, size_t vm, uintT* offsets, ngh_and_weight* edges, size_t vertex_offset) {

    using key_type = uintE;
    using value_type = uintE;
    using KV = std::pair<key_type, value_type>;

    auto get_deg = [&] (size_t i) {
      return ((i == (vn-1)) ? vm : offsets[i+1]) - offsets[i];
    };

    auto elts_unfiltered = parlay::sequence<KV>::from_function(vn, [&] (size_t i) -> KV {
      size_t deg = get_deg(i);
      KV ret = std::make_pair(vertex_offset + i, i);
      return (deg == 0) ? std::make_pair(UINT_E_MAX, UINT_E_MAX) : ret;
    });
    auto elts = parlay::filter(parlay::make_slice(elts_unfiltered), [&] (auto e) {
      return e.first != UINT_E_MAX;
    });

    auto combine_op = [&] (edge_node* cur, value_type incoming) {
      std::cout << "Unexpected call to combine, quitting." << std::endl;
      exit(-1);
      return nullptr;
    };
    auto map_op = [&] (value_type i) {
      size_t off = offsets[i];
      size_t deg = get_deg(i);

      for (size_t j=0; j<deg; j++) {
        auto our_id = vertex_offset + i;
        if (std::get<0>(*(edges + off + j)) == our_id) {
          std::cout << "self loop!: " << our_id << std::endl;
          exit(0);
        }
      }

      auto tree = edge_tree(edges + off, edges + off + deg);
      auto root = tree.root;
      tree.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(root) == 1);
      return root;
    };

    V = vertex_tree::multi_insert_sorted_map(std::move(V), elts, combine_op, map_op);
  }


};

}  // namespace aspen
