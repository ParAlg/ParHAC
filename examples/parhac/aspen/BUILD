package(
  default_visibility = ["//visibility:public"],
)

cc_library(
  name = "api",
  hdrs = ["api.h"],
  deps = [
  ":graph_io",
  ":immutable_graph",
  ":traversable_graph",
  ":benchmark",
  ],
)

cc_library(
  name = "aspen",
  hdrs = ["aspen.h"],
  deps = [
  ":utils",
  ":build",
  ":immutable_graph",
  ":traversable_graph",
  ":api",
  "//pam/include/pam:pam",
  "//include/cpam:cpam",
  ],
  copts = ["-Iinclude", "-Iparlaylib/include"],
)

cc_library(
  name = "build",
  hdrs = ["build.h"],
  deps = [
  ":utils",
  ],
)

cc_library(
  name = "immutable_graph",
  hdrs = ["immutable_graph.h"],
  deps = [
  ":build",
  ],
)

cc_library(
  name = "traversable_graph",
  hdrs = ["traversable_graph.h"],
  deps = [
  ":flags",
  ":edge_map_utils",
  ":vertex_subset",
  ],
)

cc_library(
  name = "utils",
  hdrs = ["utils.h"],
  deps = [],
)
