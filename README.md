## Repository Overview

### Experiments on graphs

Our ParHAC code (and code for our other implementations of clustering
algorithms) can be found in code/examples/parhac/.

The code can be compiled by going into one of these directories, e.g.,
```
cd code/examples/parhac/parhac/
```
and then running `make -j`.

The code can then be run on input graphs in the PBBS/GBBS adjacency
list format (described below).

```
numactl -i all ./ParHac-CPAM-CPAM-Diff -epsilon 0.01 -s -m -rounds 1 ~/inputs/soc-LiveJournal1_sym.adj
```

Please see below for instructions on the graph format that we use.


### End-to-end experiments on pointsets

For each algorithm, we have both a version that takes as input a
graph, and a version that takes as input a pointset. The pointset
versions can be found in the directories suffixed `_end_to_end`, e.g.

```
cd code/examples/parhac/parhac_end_to_end
make -j
```

The code can then be run on inputs in the ANN-benchmarks format
(`.fvecs`).

```
numactl -i all ./ParHac -k 50 -ftype fvecs /ssd0/ANN/sift1M/sift_base.fvecs
```

This command builds a similarity graph using a parallel ANN algorithm
that is part of concurrent ongoing work by the anonymized authors. The
algorithm uses a graph-based construction which is competitive with
state-of-the-art methods on the ANN-benchmarks leaderboard. The value
of k used in the construction can be supplied using the `-k` flag.




### Command-Line Flags

The applications take the input graph as input as well as an optional
flag `-s` to indicate a symmetric graph.  Symmetric graphs should be
called with the `-s` flag for better performance.


On NUMA machines, adding the command "numactl -i all " when running
the program may improve performance for large graphs. For example:

```
> numactl -i all ./ParallelHAC [...]
```

When processing large compressed graphs, using the `-m` command-line flag can
help speed-up data loading if the input file is already in the page cache, since the
compressed graph data can be mmap'd.


### Input Formats
We support the adjacency graph format used by the [Problem Based Benchmark
suite](http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html)
and [Ligra](https://github.com/jshun/ligra).

The adjacency graph format starts with a sequence of offsets one for each
vertex, followed by a sequence of directed edges ordered by their source vertex.
The offset for a vertex i refers to the location of the start of a contiguous
block of out edges for vertex i in the sequence of edges. The block continues
until the offset of the next vertex, or the end if i is the last vertex. All
vertices and offsets are 0 based and represented in decimal. The specific format
is as follows:

```
AdjacencyGraph
<n>
<m>
<o0>
<o1>
...
<o(n-1)>
<e0>
<e1>
...
<e(m-1)>
```

This file is represented as plain text.

Weighted graphs are represented in the weighted adjacency graph format. The file
should start with the string "WeightedAdjacencyGraph". The m edge weights
should be stored after all of the edge targets in the .adj file.

### Using SNAP graphs

Graphs from the [SNAP dataset
collection](https://snap.stanford.edu/data/index.html) are commonly used for
graph algorithm benchmarks. We provide a tool that converts the most common SNAP
graph format to the adjacency graph format that GBBS accepts. Usage example:
```sh
# Download a graph from the SNAP collection.
wget https://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip --decompress ${PWD}/wiki-Vote.txt.gz
# Run the SNAP-to-adjacency-graph converter.
# Run with Bazel:
bazel run //utils:snap_converter -- -s -i ${PWD}/wiki-Vote.txt -o <output file>
# Or run with Make:
#   cd utils
#   make snap_converter
#   ./snap_converter -s -i <input file> -o <output file>
```

