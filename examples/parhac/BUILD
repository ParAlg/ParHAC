package(
  default_visibility = ["//visibility:public"],
)

cc_library(
  name = "index",
  hdrs = ["index.h"],
  srcs = ["index.cpp"],
  deps = [
  "//include/cpam:cpam",
  "//include/cpam:get_time",
  "//include/cpam:parse_command_line",
  "//parlaylib/include/parlay:io",
  "//parlaylib/include/parlay:primitives",
  "//parlaylib/include/parlay:random",
  ],
  copts = ["-Iinclude", "-Iparlaylib/include"],
)

