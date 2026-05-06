# Ethics, Privacy, and Sustainability

> Canonical privacy-boundary note: the detailed implementation-level privacy boundary is documented in [`privacy_boundary.md`](privacy_boundary.md). This file keeps the broader ethics/privacy discussion.

This document explains the privacy, ethical, and sustainability considerations of FairVote-AI. It deliberately separates the privacy mechanism from the AI inference model.

## 1. Privacy and data protection

### 1.1 Local Differential Privacy

FairVote-AI uses Local Differential Privacy (LDP) through k-ary Randomized Response as its primary collection-time privacy mechanism.

Randomized Response is not AI. It is a probabilistic privacy mechanism. For privacy parameter epsilon, the submitted answer is drawn from a distribution that depends on the respondent's true answer but does not reveal it deterministically. Under the k-ary RR mechanism, the probability ratio between outputs generated from any two possible true answers is bounded by `exp(epsilon)`.

Important design properties:

- The respondent browser applies RR before submission.
- The Flask server receives `perturbed_answer`, not the true answer.
- Requests containing raw-answer fields such as `true_answer`, `true_choice`, `selected_answer`, `selectedOption`, `raw_vote`, or `raw_answer` are rejected recursively, including nested objects/lists.
- Stored records contain privatized reports, validated demographics, and at most a reduced-precision timestamp. Individual-level exports require an analyst bearer token; aggregate `/api/results` remains aggregate-only.

### 1.2 AI and privacy compatibility

The AI component is the RR-aware Neural MRP inference model. It is not part of the respondent collection protocol, and it does not replace Randomized Response.

The neural model is privacy-compatible because it trains on the same data available to the analyst:

```text
demographic features + privatized reported answers
```

It models a latent true vote distribution, then passes that distribution through the known Randomized Response observation channel:

```text
P(reported = r | x)
  = sum_t P_theta(true = t | x) P_RR(reported = r | true = t)
```

The training objective is the marginal likelihood of the privatized reports. The model therefore does not require the server to receive or store true votes.

In synthetic experiments, true labels exist because the simulator generated them. They are used after fitting to score aggregate and subgroup errors. They are not the training labels for the neural model.

### 1.3 GDPR and data-minimisation considerations

- **Data minimisation**: The respondent server stores only the privatized answer, configured demographic labels, and a reduced-precision timestamp by default. It does not intentionally store names, emails, or account identifiers.
- **Purpose limitation**: The collected data is intended for aggregate statistical estimation and method evaluation.
- **Deletion**: The JSONL format supports deletion of individual records where a record can be identified.
- **Transparency**: The respondent client exposes the randomization mechanism and privacy parameter to the user.

A real deployment would still require a context-specific data protection review. Demographic fields and operational metadata can still be personal data even if the answer itself is locally privatized.

### 1.4 Limitations of LDP

LDP is useful but limited:

- **LDP is not anonymity**: It protects the submitted answer value. It does not hide the respondent's identity, IP address, timing, browser metadata, or demographic uniqueness.
- **Demographics are not randomized**: Age group, region, education, and similar fields are transmitted as entered unless the deployment adds extra protection. Small demographic cells can still create disclosure and reliability risks even when the answer value is locally randomized.
- **Epsilon matters**: Very large epsilon values provide weak privacy. Very small epsilon values can destroy too much signal for reliable estimation.
- **Side channels remain**: Deployment logs, hosting infrastructure, timestamps, IP addresses, and network metadata can undermine privacy if handled badly. The demo reduces timestamp precision and protects individual-level exports, but a real deployment still needs operational controls.

## 2. Fairness and social impact

Privacy noise can disproportionately harm small subgroups. A subgroup with few respondents receives the same mechanism-level privacy protection but usually has less statistical information for estimation.

FairVote-AI audits this issue with:

- worst-group L1 error,
- weighted group L1 error,
- p90 group error,
- error-ratio metrics,
- dashboard subgroup views where available.

These metrics do not guarantee fairness. They expose disparities so that an analyst can report uncertainty, reject unreliable configurations, or collect more data.

## 3. Misuse risks

### Cherry-picking

An analyst could run many configurations and report only favourable outputs. The experiment scripts reduce this risk by saving config snapshots and machine-readable outputs, but responsible reporting still matters.

### Over-confidence

Synthetic validation is not proof of real election accuracy. It tests behaviour under the simulator's assumptions. Real electorates, turnout, parties, geography, and response behaviour may differ.

### Misleading AI claims

The neural model may help when nonlinear demographic structure exists, but it can also overfit, underperform, or be slower than RR-aware linear poststratification/MRP. It should be described as an evaluated inference method, not as automatically superior.

The final-evidence protocol reinforces this point by comparing RR-aware Neural MRP against simpler baselines rather than assuming it is superior. It would be misleading to present the AI component as a guaranteed improvement; claims should be based on the current generated `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv` and `runtime_profile.csv`.

### Misleading privacy claims

Running the system with a weak privacy parameter while advertising strong privacy would be deceptive. Reports should state epsilon, the RR mechanism, and any non-randomized fields collected.

## 4. Sustainability

- The non-neural components are lightweight and can run on a laptop.
- The neural model is small compared with large deep-learning systems, but it still adds runtime and dependency cost through PyTorch.
- The full experiment presets can be computationally heavier because they vary epsilon, sample size, scenario, and random seed.
- No cloud deployment is required for the project experiments, though users may choose to run larger sweeps on external hardware.

The added neural complexity is justified only if experiments show useful improvements in overall or subgroup estimation. If it does not, the correct conclusion is that the simpler baseline is preferable for that setting.

## 5. Regulatory context

- **UK Data Protection Act 2018**: LDP is a technical privacy measure, but compliance depends on the whole deployment context.
- **GDPR Article 25**: The project follows a data-protection-by-design approach by applying RR before collection.
- **GDPR Article 35**: A real deployment would still need a Data Protection Impact Assessment, especially because demographics and operational metadata may be personal data.

This documentation is not legal advice.
