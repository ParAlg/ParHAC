// Usage:
// numactl -i all ./Affinity -src 10012 -s -m -rounds 3 twitter_SJ
// flags:
//   required:
//     -src: the source to compute the Affinity from
//   optional:
//     -rounds : the number of times to run the algorithm
//     -c : indicate that the graph is compressed
//     -m : indicate that the graph should be mmap'd
//     -s : indicate that the graph is symmetric

#include "Affinity.h"
#include "aspen/aspen.h"

#include <cpam/parse_command_line.h>
#include <fstream>

namespace aspen {

template <class Dendrogram>
void WriteAvgDendrogramToDisk(Dendrogram& dendrogram, const std::string& of) {
  std::ofstream out;
  out.open(of);
  size_t wrote = 0;
  out.precision(std::numeric_limits<double>::max_digits10);
  for (size_t i = 0; i < dendrogram.size(); i++) {
    if (dendrogram[i].first != i) {
      if (dendrogram[i].first != UINT_E_MAX) {
        out << i << " " << dendrogram[i].first << " "
            << dendrogram[i].second << std::endl;
      }
      wrote++;
    }
  }
  std::cout << "Wrote " << wrote << " parent-pointers." << std::endl;
}

template <class Graph>
double Affinity_runner(Graph& G, cpam::commandLine P) {
  std::cout << "### Application: Affinity" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << parlay::num_workers() << std::endl;
  std::cout << "### n: " << G.num_vertices() << std::endl;
  std::cout << "### m: " << G.num_edges() << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  bool scc = P.getOptionValue("-scc");
  auto of = P.getOptionValue("-of", "");

  Dendrogram dendrogram;
  timer t; t.start();
  if (scc) {
    double lower_thresh = P.getOptionDoubleValue("-epsilon", 0.001);
    double upper_thresh = P.getOptionDoubleValue("-upper", std::numeric_limits<double>::infinity());
    size_t iters = P.getOptionLongValue("-iters", 50);
    SCC(G, lower_thresh, upper_thresh, iters);
  } else {
    dendrogram = Affinity(G);
  }
  double tt = t.stop();

  if (of != "") {
    // write merges
    std::cout << "Writing dendrogram" << std::endl;
    WriteAvgDendrogramToDisk(dendrogram, of);
  }

  std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}

}  // namespace aspen

generate_symmetric_aspen_main(aspen::Affinity_runner);
