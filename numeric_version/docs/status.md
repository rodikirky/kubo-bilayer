# Project status

## Current focus
- [ ] Bulk FFT + kz coverage metric + wrap-around checks
- [ ] Gluing correctness + unit tests
- [ ] Kubo/Středa integration pipeline and convergence tests

## What is stable
- Preset-driven diagnostics entry points
- Bulk FFT transform interface (subject to minor refactors)

## What may change soon
- ModelConfig fields and registry dispatch
- Gluing interfaces and integration module structure

## Next milestones
1. Orbitronic bulk diagnostics plots (kz overlay + coverage scalar)
2. Interface gluing regression tests
3. End-to-end toy pipeline (bulk → glue → simple Kubo sanity)

## Known issues / gotchas
- FFT kz coverage and real-space wrap-around must both be checked (see diagnostics.md)
