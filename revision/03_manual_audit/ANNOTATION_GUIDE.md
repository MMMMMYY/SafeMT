# Audit criteria and released files

See [annotation schemas](../ANNOTATION_SCHEMAS.md) for the current trigger-specific
labels and [sampling counts](../README.md#300-pair-flagged-case-audit) for the fixed sample.

Assessors inspect the prompts and available images. For safety judgments, the
reference documents are the OpenAI image-generation safety/usage policies and
Google's Generative AI Prohibited Use Policy:

- https://deploymentsafety.openai.com/4o-native-image-generation
- https://openai.com/policies/usage-policies/
- https://policies.google.com/terms/generative-ai/use-policy?hl=en

Original A/B exports are retained separately. Final labels are supplied result
records. Kappa and raw agreement are computed before final resolution. Semantic
content differences are reported separately from classifier and refusal outcomes.

Inspect [300 flagged pairs](flagged_pairs_review.html) or
[200 classifier images](classifier_images_review.html) offline.
