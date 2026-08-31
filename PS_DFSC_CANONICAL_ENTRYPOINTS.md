# Final canonical PS-DFSC entry points

Use these commands for the formal experiment. They supersede all earlier
`final`, `publication`, and lower-numbered `canonical` wrappers retained for
audit history.

- Identity commitment/cost preparation:
  `python -m repro_scripts.prepare_ps_dfsc_publication_strict_v3`
- Candidate train, calibrate, safety-gate, and exact-evaluate:
  `python -m repro_scripts.run_ps_dfsc_canonical_v8`
- Candidate proxy:
  `python -m repro_scripts.evaluate_ps_dfsc_proxy`
- Paired exact validation summary:
  `python -m repro_scripts.summarize_ps_dfsc_exact_pair`
- Candidate record:
  `python -m repro_scripts.build_ps_dfsc_candidate_record_publication`
- Candidate selection:
  `python -m repro_scripts.select_ps_dfsc_candidate_publication`
- Publication lock:
  `python -m repro_scripts.lock_ps_dfsc_publication`
- Locked confirmation:
  `python -m repro_scripts.run_ps_dfsc_confirmation_canonical_v7`
- Final three-outer report:
  `python -m repro_scripts.ps_dfsc_final_report_publication`

The exact solver uses a 600-second total budget and adaptively tightens the raw
HiGHS gap when objective-constant removal makes the reported operating-cost gap
larger than 0.1%. A binary incumbent that exhausts the budget is retained with
`success=False`, its termination reason, adjusted dual bound, actual gap, node
count, and runtime.

Do not invoke the locked confirmation runner until
`lock_ps_dfsc_publication` has created and verified the outer's v2 manifest.
