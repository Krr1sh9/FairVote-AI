# Privacy boundary

FairVote-AI is a research prototype for privacy-preserving polling. It is not production election software and does not provide secure voter authentication.

## Boundary guaranteed by the implementation

1. The browser applies k-ary Randomized Response before submission.
2. The network payload contains `perturbed_answer` and validated demographics only.
3. The server recursively rejects raw-answer fields such as `true_choice`, `selected_answer`, `raw_answer`, and similar variants before storage.
4. Stored records are append-only JSONL records containing perturbed answer, demographics, and reduced-precision timestamp unless timestamp storage is disabled.
5. Individual record export is disabled unless an analyst token is configured.
6. Even with a token, individual export is blocked by default when demographic cells fail the k-anonymity threshold. Analysts should use aggregate `/api/results` for normal analysis.

## Boundary not guaranteed

- LDP protects the answer value, not every demographic combination.
- Demographics are sent as-is and can create re-identification risk when cells are rare.
- The client-side localStorage duplicate guard is convenience-only and is not voter authentication.
- The system does not claim anonymity against browser compromise, malicious clients, server compromise, or side-channel observation.

## Production debug mode removed

Earlier prototypes exposed selected/perturbed answer indices locally via a browser audit mode. That was removed from the production client because it created avoidable leakage risk in screenshots, shared devices, and demos. The current client keeps mechanism explanations but does not render per-response raw debug values.

## Dashboard truth-column rule

Truth columns such as `true_choice` are accepted only in explicit synthetic-evaluation mode. Real respondent exports should not contain truth columns. The dashboard records `synthetic_evaluation_mode` in export metadata.
