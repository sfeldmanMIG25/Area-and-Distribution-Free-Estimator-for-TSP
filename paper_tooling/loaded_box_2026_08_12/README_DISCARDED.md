# Discarded timing runs, 2026-08-12 morning

Nothing in this directory may be quoted. It is kept for provenance only.

These arms were measured while the box was running a 12-worker and then a
24-worker Concorde corpus-generation job (`data_pipeline/coverage_gen.py`,
plans `main`, `poly`, `topup`) inside WSL2. `vmmemWSL` held 5.7 to 9.6 of the
20 logical processors for the whole window, and the drift control taken in it
put GART 2.0 on TSPLIB EUC\_2D at 12.46 ms against the published 6.122 ms — a
factor of 2.04.

Two of the arms are worse than merely inflated. A background wrapper was
stopped without its grandchildren, so a driver shell outlived the job it
belonged to, started the next arm, and then — after the driver script was
rewritten under it — forked three more copies of itself. For roughly
forty-five minutes two timing arms measured each other. `2d_vj_ckpt` and
`2d_polyak` from that window were deleted rather than archived, because a
co-measured 1-tree timing is not a timing of anything.

The published pass hit the same failure once before; see
`hk1tree_solo_cost.py -> co_measurement_incident`. It cannot recur silently
now: `hk1tree_cost_allbench.assert_solo()` refuses to start when another copy
of the harness is running, and prints the check it passed.

The surviving measurement is the session recorded in
`paper_tooling/cost_session.log` and analysed into
`paper_tooling/hk1tree_cost_frontier_bank.json`.
