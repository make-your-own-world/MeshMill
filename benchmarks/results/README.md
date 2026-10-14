# Benchmark results

This directory contains the recorded JSON measurements and visual comparisons used by the
Depth + Density Adaptive QEM validation scripts. The benchmark scripts in the parent directory
recreate each run from `samples/original-scan.stl` and report matched triangle-count comparisons.

Included material:

- JSON reports for the parameter search, validation runs, and production comparisons;
- density and wireframe comparison images for the 250,000-triangle reference run.

The derived STL outputs from individual benchmark runs are intentionally not versioned here. They
total more than 400 MiB, duplicate geometry generated from the published reference sample, and can
be recreated with the corresponding benchmark script. The original 4,126,315-triangle source
fixture is available through Git LFS at `samples/original-scan.stl`.

Measurements describe the tested fixtures and settings. They are reproducibility records, not a
guarantee of the same timing or GPU behavior on every system.
