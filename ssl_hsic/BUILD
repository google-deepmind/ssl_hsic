# Experiment for SSL-HSIC.
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//learning/deepmind/python/fragmented_python:build_defs.bzl", "fragmented_py_binary_mpms")
load("//learning/deepmind/jax:build_defs.bzl", "jax_py_binary")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

exports_files(["LICENSE"])

MPM_PREFIX = "test/third_party/deepmind/ssl_hsic/ssl_hsic"

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "//learning/deepmind/research/jax/pipeline:base_config",
        "//third_party/deepmind/deepmind_research/byol:utils",
        "//third_party/py/jaxline:base_config",
        "//third_party/py/ml_collections/config_dict",
    ],
)

pytype_strict_library(
    name = "kernels",
    srcs = ["kernels.py"],
    deps = [
        "//third_party/py/haiku",
        "//third_party/py/jax",
        "//third_party/py/mpmath",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "kernels_test",
    srcs = ["kernels_test.py"],
    deps = [
        ":kernels",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = ["//third_party/py/jax"],
)

pytype_strict_library(
    name = "eval_config",
    srcs = ["eval_config.py"],
    deps = [
        "//learning/deepmind/research/jax/pipeline:base_config",
        "//third_party/deepmind/deepmind_research/byol:utils",
        "//third_party/py/jaxline:base_config",
        "//third_party/py/ml_collections/config_dict",
    ],
)

jax_py_binary(
    name = "experiment",
    srcs = ["experiment.py"],
    data = [
        ":config",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":helpers",
        ":kernels",
        "//learning/deepmind/research/jax/pipeline:dm_platform",
        "//learning/deepmind/research/jax/pipeline:experiment",
        "//learning/deepmind/research/jax/pipeline:utils",
        "//third_party/deepmind/deepmind_research/byol:utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/acme/jax:utils",
        "//third_party/py/haiku",
        "//third_party/py/jax",
        "//third_party/py/jaxline:experiment",
        "//third_party/py/jaxline:platform",
        "//third_party/py/jaxline:utils",
        "//third_party/py/numpy",
        "//third_party/py/optax",
        "//third_party/py/tensorflow",
    ],
)

jax_py_binary(
    name = "eval_experiment",
    srcs = ["eval_experiment.py"],
    data = [":eval_config"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/deepmind/research/jax/pipeline:dm_platform",
        "//learning/deepmind/research/jax/pipeline:experiment",
        "//third_party/deepmind/deepmind_research/byol:utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/acme/jax:utils",
        "//third_party/py/haiku",
        "//third_party/py/jax",
        "//third_party/py/jaxline:experiment",
        "//third_party/py/jaxline:platform",
        "//third_party/py/numpy",
        "//third_party/py/optax",
        "//third_party/py/tensorflow",
    ],
)

mpm_fragments = [
    "//",
]

fragmented_py_binary_mpms(
    name = "tpu_mpms",
    binary = ":experiment_tpu",
    fragments = mpm_fragments,
    package_name_prefix = MPM_PREFIX,
)

fragmented_py_binary_mpms(
    name = "eval_tpu_mpms",
    binary = ":eval_experiment_tpu",
    fragments = mpm_fragments,
    package_name_prefix = MPM_PREFIX,
)
