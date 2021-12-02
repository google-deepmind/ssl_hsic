load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

copybara_config_test(
    name = "copybara_test",
    config = "copy.bara.sky",
    deps = [
        "//learning/deepmind/devtools/opensourcing:leakr_deps",
        "//learning/deepmind/opensource/staging:copybara_common",
    ],
)

license(name = "license")
