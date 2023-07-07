// Usage: TODO

#include "ParHac.h"
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
double ParHac_runner(Graph& G, cpam::commandLine P) {

  double epsilon = P.getOptionDoubleValue("-epsilon", 0.1);
  auto of = P.getOptionValue("-of", "");
  bool get_size = P.getOptionValue("-get_size");

  std::cout << "### Application: ParHac" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << parlay::num_workers() << std::endl;
  std::cout << "### n: " << G.num_vertices() << std::endl;
  std::cout << "### m: " << G.num_edges() << std::endl;
  std::cout << "### Params: epsilon = " << epsilon << " out_file (dendrogram) = " << of << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  timer t; t.start();
  auto dendrogram = ParHac(G, epsilon, get_size);
  merge_t.reportTotal("merge time");
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

generate_symmetric_aspen_main(aspen::ParHac_runner);
