# AGENTS.md

Project structure, tech stack, and build instructions live in
[`README.md`](README.md) - read it first.

## Working in this repo

- HIP-only. There is no Vulkan path, no build-time backend toggle - if a
  change needs conditional compilation for "no HIP available," that's a
  sign it doesn't belong in this repo.
- Comments are short or absent. Explain *why* only where the code cannot,
  and never restate what the code already says. Design rationale belongs in
  `docs/`.
- Code never points at documentation. No `see docs/x.md` in a comment, a
  docstring, or a diagnostic string - the code is the primary source, and
  those references rot the moment a doc is renamed.
- Before touching either kernel, re-read the "Gotchas" section in
  `README.md` - the `CMAKE_BUILD_TYPE` one in particular is a real
  correctness bug, not a style preference.
- Both correctness gates (`tests/hip_forward_gate.hip`,
  `hip_forward_int8qk_gate.hip`) are standalone `hipcc` builds, not wired
  into CMake. Run them after any kernel change - see README.md for the
  exact commands. They compare against a from-scratch fp64 CPU oracle, not
  against each other or any prior kernel version.
- Several constants are measured, not derived - GQA `G=2`, the BR=128 tile
  choice, `kTileVgprCap`. Re-benchmark before "correcting" one.
