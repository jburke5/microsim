# Test Guidelines

## Debugging Test Failures

If a test fails because of an issue associated with the dataframe (e.g., missing data, unexpected values, weights summing to zero), it is likely that the test is correct and the dataframe has been corrupted. Always check with the user before modifying a test in this situation.

## Stochastic Tests

Every `Person` gets `_rng = np.random.default_rng()` seeded from OS entropy at construction, so
nothing that draws through a `Person` is reproducible across runs. A test that rebuilds a
population and compares against a target is therefore comparing two independent random
realizations, and its tolerance has to cover that spread rather than the modelling error alone.

`test_calibrate_prevalence` is the worked example. `calibrate_prevalence` solves against one
frozen set of per-person RNG states, while the test measures on a freshly built population, so
the two differ by about one binomial standard error in each direction. At the ~1392 persons of
the 65+ scope that is ~0.011, against a tolerance that was 0.01 — the test failed roughly a
third of the time regardless of whether the calibration was correct. It now averages the
measurement over 5 rebuilds and asserts `delta=0.02`; the whole-population assertions keep
`delta=0.01`, where the larger N leaves 3+ standard errors of headroom.

Before widening a tolerance, work out the standard error at that sample size — if the tolerance
is under ~2 standard errors the test is a coin flip, and if it is far above, the test is not
asserting anything.
