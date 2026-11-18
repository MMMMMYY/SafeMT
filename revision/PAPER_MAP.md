# Paper-to-artifact map

Reference: **Detecting Safety Behavior Inconsistencies in Text-to-Image Generation via Metamorphic Testing**. Page numbers refer to the 13-page paper.

| Paper location | Computation/output | Inputs | Script and generated artifact | PDF page |
|---|---|---|---|---|
| Table I / Section III-B | Examples of MR follow-up prompts | [Canonical follow-up prompts](inputs/followup_prompts/) | [MR identifiers and names](scripts/package_paths.py) | 5 |
| Table II / Section IV | Safety categories and anchor terms | [Canonical source prompts](inputs/source_prompts_270.jsonl) | [Source loader](scripts/package_paths.py) | 6 |
| Table III / Section IV | Automated and human MR preservation | [Automated scores](04_mr_validation/mr_validation_safemt.json), [A/B labels and final decisions](04_mr_validation/) | [Evidence checks](scripts/verify_package.py); [Numerical results](tables/paper_results.json) | 6 |
| Table IV / Section V-B | Complete original Stage-1 results | [All three result files](../dataset/consistency_results/) | [Evidence checks](scripts/verify_package.py); [Numerical results](tables/paper_results.json) | 7 |
| Table V / Section V-B | Transformed vs unchanged-prompt baseline | [Frozen pair records and bootstrap output](01_repeated_generation_baseline/) | [Statistical analysis](scripts/revision_analysis.py); [Re-aggregation](scripts/revision_analysis.py); [Numerical results](tables/paper_results.json) | 8 |
| Table VI / Section V-B | MR-wise excess IR and confidence intervals | [Per-stratum bootstrap output](01_repeated_generation_baseline/revision_analysis.json) | [Statistical analysis](scripts/revision_analysis.py); [Numerical results](tables/paper_results.json) | 8 |
| Table VII / Section V-B | Threshold robustness and negative control | [Threshold outputs](02_threshold_and_negative_control/) | [Statistical analysis](scripts/revision_analysis.py); [Numerical results](tables/paper_results.json) | 8 |
| Table VIII / Section V-B | 300-pair flag audit | [Final labels](03_manual_audit/flagged_pairs_final.csv), [sample](03_manual_audit/flagged_pairs_sample.csv), [A](03_manual_audit/flagged_pairs_A.csv), [B](03_manual_audit/flagged_pairs_B.csv) | [Evidence checks](scripts/verify_package.py); [Numerical results](tables/paper_results.json) | 9 |
| Section VI: Construct validity | 200-image classifier accuracy and agreement | [200-image files and four decisions](03_manual_audit/) | [Evidence checks](scripts/verify_package.py); [Numerical output](tables/paper_results.json) | 11 |
| Figure 3 / Section V-B | Four same-label semantic examples | [Pair metadata/distances](03_manual_audit/figure_03_examples.csv), [original images](figures/figure_03_images/) | [Inspection panel](figures/figure_03_examples.html); [page builder](scripts/build_review_pages.py) | 9 |
| Figure 4 / Section V-C | Saliency distributions for flagged cases | [Individual corrected saliency records](05_stage2_all_pairs_control/) | [`figure4()`](scripts/build_saliency_figures.py#L42); [PDF](figures/figure_04_saliency.pdf); [Panel counts](figures/figure_04_metrics.json) | 10 |
| Figure 5 / Section V-C | Flagged vs unflagged saliency control | [Control summary](05_stage2_all_pairs_control/stage2_control.json), [re-analysis script](scripts/revision_stage2_control.py) | [`figure5()`](scripts/build_saliency_figures.py#L65); [PDF](figures/figure_05_flagged_unflagged.pdf) | 10 |
