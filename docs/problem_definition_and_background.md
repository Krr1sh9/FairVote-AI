# Problem Definition and Academic Background

## Project title

**FairVote-AI: A reproducible research prototype and benchmark for comparing Randomized-Response-aware polling estimators under Local Differential Privacy and sampling/misreporting bias.**

## Concise aim

The aim of this project is to design, implement, and evaluate a privacy-preserving polling prototype that collects only locally privatized responses and benchmarks whether RR-aware Neural MRP improves over simpler RR-aware linear/poststratification baselines under controlled synthetic conditions involving Local Differential Privacy, sampling bias, nonlinear demographic structure, and misreporting.

## Measurable objectives

| ID | Objective | Measurement of completion |
|---|---|---|
| O1 | Implement a browser-side k-ary Randomized Response respondent flow that submits privatized reports rather than raw answers. | Respondent app sends `perturbed_answer`; server rejects forbidden raw-answer fields; privacy-boundary tests pass. |
| O2 | Implement canonical RR utilities for transition probabilities, privatization, and debiasing. | One canonical Python RR channel is used by estimators, dashboard, and experiments; property/statistical tests validate invariants. |
| O3 | Implement and document a canonical RR-aware linear poststratification/MRP-style estimator with poststratification. | Single public import path, validated design matrices, diagnostics, metadata export, and regression tests. |
| O4 | Implement an RR-aware Neural MRP estimator that trains through the known RR observation channel using privatized reported labels. | Training does not require true labels; validation reported-label NLL, early stopping, loss history, calibration helpers, and tests exist. |
| O5 | Build a reproducible experiment pipeline comparing RR debiasing, RR-aware linear poststratification/MRP, RR-aware Neural MRP, oracle baselines, and ablations. | Presets generate `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `config.json`, and `manifest.json`. |
| O6 | Evaluate when neural modelling is useful, harmful, or unnecessary under privacy noise, nonresponse, nonlinear interaction, and misreporting scenarios. | Paired neural-minus-linear deltas, confidence intervals, win rates, runtime profiles, and documented interpretation rules support a conditional conclusion. |

## Formal research questions

**RQ1.** Under Local Differential Privacy implemented with k-ary Randomized Response, how much utility is lost as epsilon decreases, and how does that loss differ for aggregate and subgroup estimates?

**RQ2.** Does poststratification improve estimates from biased or nonrepresentative samples when only privatized reports are available?

**RQ3.** When the true preference function is approximately additive or linear in demographic features, is a simpler RR-aware linear/poststratification estimator sufficient compared with RR-aware Neural MRP?

**RQ4.** When the true preference function contains nonlinear demographic interactions or sparse subgroup effects, does RR-aware Neural MRP improve aggregate L1 error, worst-group L1 error, or calibration enough to justify its extra complexity and runtime?

**RQ5.** How do misreporting and shy-voter-style response mechanisms interact with Local Differential Privacy, and what do oracle or misreport-aware baselines reveal about the limits of estimators that assume standard RR only?

## Problem context

Polling systems usually need individual-level responses to estimate aggregate public opinion, subgroup preferences, and uncertainty. That creates a tension: raw responses can be sensitive, but the analyst still wants useful estimates. This tension becomes sharper when the survey concerns political choices, controversial topics, small demographic groups, or populations where trust in the collector is low.

Classic survey problems also remain. Samples are rarely perfectly representative; response rates can differ across demographic groups; and people may misreport sensitive preferences. Nonresponse bias is not determined by the response rate alone, but by whether response propensity is related to the statistic being estimated (Groves and Peytcheva, 2008). Online and nonprobability samples can also produce uneven error patterns across groups, so aggregate accuracy can hide subgroup failure (Kennedy et al., 2016).

FairVote-AI addresses this setting as a research prototype: collect locally privatized answers, then compare estimators that attempt to recover useful aggregate and subgroup estimates from noisy reports and biased samples.

## Why privacy-preserving polling matters

A polling system that stores raw answers must be trusted not to misuse, leak, subpoena, or re-identify those answers. Even if the final published result is aggregate, the collection system may still hold sensitive individual records. Privacy-preserving collection reduces that trust burden by changing what the server receives.

This project uses privacy-preserving polling as a concrete Computer Science problem because it combines:

- privacy mechanism design;
- client/server boundary design;
- statistical inference under noisy observations;
- sampling and subgroup bias;
- evaluation and reproducibility;
- honest discussion of ethical and deployment limits.

The project does **not** claim to provide secure election infrastructure, anonymous voting, voter authentication, coercion resistance, eligibility verification, end-to-end verifiability, or protection from hosting/traffic metadata. It studies a narrower question: what can be estimated when the answer value is locally privatized before reaching the server?

## Why Local Differential Privacy and Randomized Response are relevant

Differential privacy formalises limits on how much the presence or value of one individual record can influence an output (Dwork et al., 2006). In the local model, the perturbation happens before the data reaches the collector, so the server receives a randomized report rather than the raw value. This is relevant for polling because the respondent does not need to trust the server with the unperturbed answer.

Randomized Response is a classic survey method for sensitive questions (Warner, 1965). In its k-ary form, the respondent keeps the true category with probability determined by epsilon and otherwise reports another category according to the mechanism. This gives a clear privacy/utility trade-off:

- low epsilon gives stronger privacy but noisier reports;
- high epsilon gives weaker privacy but more accurate reports;
- analytical inversion can debias aggregate counts but increases variance, especially with small samples or many categories.

Modern Local Differential Privacy work provides the theoretical context for this trade-off. Duchi, Jordan and Wainwright study statistical estimation under Local Differential Privacy constraints, including multinomial probability estimation (Duchi, Jordan and Wainwright, 2013). RAPPOR demonstrates how randomized-response-style mechanisms can be used for client-side collection at scale (Erlingsson, Pihur and Korolova, 2014). FairVote-AI is smaller and educational, but the same central tension applies: privacy is obtained by intentionally degrading the observation channel.

## Why MRP and poststratification are relevant to polling bias

MRP combines a model of responses by demographic/geographic cells with poststratification to population cell counts. It is widely used for estimating opinion from samples that are not directly representative of the target population (Gelman and Little, 1997; Lax and Phillips, 2009). The key idea is that the model predicts cell-level preferences, and poststratification weights those predictions by the population distribution rather than by the sample distribution.

This is relevant to FairVote-AI because Local Differential Privacy noise does not remove sampling bias. If one group is underrepresented in the sample, simply debiasing Randomized Response counts may still estimate the sample distribution rather than the population distribution. Poststratification gives the project a principled way to separate two problems:

1. **observation noise**, caused by Randomized Response;
2. **sample composition bias**, caused by nonresponse or unequal sampling.

The implementation is deliberately named as an **MRP-style regularised regression plus poststratification estimator**, not a full hierarchical Bayesian MRP sampler. This naming matters: the project uses the MRP idea of cell modelling and poststratification, but it does not claim posterior inference from a full Bayesian multilevel model.

## Why RR-aware Neural MRP may or may not help

A neural model has a plausible role only when the relationship between demographics and preferences is nonlinear or involves interactions that a simpler additive model cannot represent well. Examples include a region × age interaction, education × urbanicity interaction, or a sparse subgroup with a different preference curve. Prior MRP work has shown the importance of deeply interacted subgroup structure in electoral settings (Ghitza and Gelman, 2013).

However, neural modelling is not automatically better. In this project it may fail or be unnecessary when:

- the true signal is simple and approximately additive;
- the sample size is too small for a flexible model;
- low epsilon makes the RR channel too noisy;
- subgroup data are sparse;
- validation reported-label NLL does not improve;
- runtime cost rises without a stable accuracy gain.

Therefore the neural component is tested as a conditional research hypothesis, not treated as an assumed improvement. The intended conclusion may be mixed or negative. That is acceptable if the evidence is rigorous.

## Gap addressed by this project

Existing work separately motivates Randomized Response, Local Differential Privacy, MRP, nonresponse adjustment, and flexible modelling. The project gap is practical and evaluative:

> **A reproducible research prototype and benchmark for comparing RR-aware polling estimators under Local Differential Privacy and sampling/misreporting bias.**

The contribution is not a new privacy theorem or a production polling product. It is an integrated benchmark and prototype that makes the following comparison auditable:

- raw reported distribution;
- analytical RR debiasing;
- RR-aware linear regression with and without poststratification;
- RR-aware Neural MRP;
- neural ablations that ignore the RR channel;
- oracle true-label and known-misreport baselines for synthetic diagnostics.

The central research value is the paired, repeated-trial comparison of methods across simple, nonlinear, nonresponse, sparse, privacy-noise, and misreporting scenarios.

## Scope

In scope:

- browser-side k-ary Randomized Response for categorical polling answers;
- a Flask respondent prototype that stores perturbed answers and validated demographics;
- aggregate and subgroup estimation from privatized reports;
- RR-aware linear poststratification/MRP-style estimation;
- RR-aware Neural MRP with validation diagnostics;
- synthetic populations and controlled bias scenarios;
- oracle baselines for synthetic-only interpretation;
- experiment presets with confidence intervals, paired comparisons, ablations, runtime profiles, and manifests;
- documentation of privacy limitations, testing, and reproducibility.

## Non-goals

Out of scope:

- production election security;
- legal voting, voter registration, eligibility checks, or one-person-one-vote enforcement;
- coercion resistance or end-to-end verifiable voting;
- anonymity against network logs, hosting logs, IP addresses, browser fingerprinting, or operator metadata;
- local privatization of demographic fields;
- proof that RR-aware Neural MRP is universally superior;
- proof that synthetic polling results transfer directly to real elections;
- a full Bayesian hierarchical MRP implementation with posterior credible intervals.

## Literature review scaffold

This scaffold lists the academic areas the final report should discuss. The current project documentation uses these sources to ground the problem and to avoid presenting the repository as merely a broad implementation.

### Randomized Response

Warner (1965) introduced Randomized Response as a way to reduce evasive-answer bias for sensitive survey questions. The relevance to this project is direct: the respondent intentionally randomizes the reported answer so the collector cannot treat one submitted value as a raw answer. The project generalises this idea to k categorical choices and uses the known transition matrix for estimation.

### Local Differential Privacy

Dwork et al. (2006) provide the central formal foundation for differential privacy. Duchi, Jordan and Wainwright (2013) analyse statistical estimation under Local Differential Privacy constraints, including probability estimation. Erlingsson, Pihur and Korolova (2014) show how randomized-response-style mechanisms can be deployed for client-side data collection in RAPPOR. These sources support the project’s privacy/utility framing: Local Differential Privacy reduces trust in the server but increases statistical noise.

### Differential privacy in surveys

In surveys, Local Differential Privacy is attractive because the server does not receive raw answers, but the utility cost is paid at the individual response level. The report should distinguish Local Differential Privacy from central differential privacy and should explain that FairVote-AI protects the answer value but not demographics, identity, metadata, or endpoint logs.

### MRP and poststratification

Gelman and Little (1997) motivate poststratification into many categories using hierarchical logistic regression. Lax and Phillips (2009) apply MRP to state-level public opinion estimation. Downes et al. (2018) discuss MRP for estimating population quantities from highly selected samples. These sources motivate using population cell counts to correct sample-composition bias after modelling cell-level response probabilities.

### Polling bias and nonresponse

Groves and Peytcheva (2008) show that nonresponse bias depends on the relationship between response propensity and survey variables, not simply on the response rate. Kennedy et al. (2016) document errors in online nonprobability survey estimates, including group-level concerns. These motivate the project’s nonresponse and subgroup-error scenarios.

### Subgroup/fairness error metrics

The project reports overall L1 error, worst-group L1, weighted group L1, and related summaries because aggregate performance can hide high error for small or underrepresented groups. In the final report this should be framed as an error-auditing approach rather than a guarantee of fairness.

### Neural/ML-based poststratification

Ghitza and Gelman (2013) show the importance of deeply interacted subgroups in MRP-style political estimation. This motivates testing flexible nonlinear models, but it does not prove that neural models will help under Local Differential Privacy. FairVote-AI therefore evaluates RR-aware Neural MRP against simpler baselines in both scenarios where nonlinear modelling is expected to help and scenarios where it should not.

## Reference list

Brier, G.W. (1950) 'Verification of forecasts expressed in terms of probability', *Monthly Weather Review*, 78(1), pp. 1-3.

Downes, M., Gurrin, L.C., English, D.R., Pirkis, J., Currier, D., Spittal, M.J. and Carlin, J.B. (2018) 'Multilevel regression and poststratification: a modeling approach to estimating population quantities from highly selected survey samples', *American Journal of Epidemiology*, 187(8), pp. 1780-1790.

Duchi, J.C., Jordan, M.I. and Wainwright, M.J. (2013) 'Local privacy and minimax bounds: sharp rates for probability estimation', *Advances in Neural Information Processing Systems*, 26.

Dwork, C., McSherry, F., Nissim, K. and Smith, A. (2006) 'Calibrating noise to sensitivity in private data analysis', in *Theory of Cryptography Conference*. Berlin: Springer, pp. 265-284. https://doi.org/10.1007/11681878_14

Erlingsson, U., Pihur, V. and Korolova, A. (2014) 'RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response', *Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security*, pp. 1054-1067.

Gelman, A. and Little, T.C. (1997) 'Poststratification into many categories using hierarchical logistic regression', *Survey Methodology*, 23(2), pp. 127-135.

Ghitza, Y. and Gelman, A. (2013) 'Deep interactions with MRP: election turnout and voting patterns among small electoral subgroups', *American Journal of Political Science*, 57(3), pp. 762-776.

Groves, R.M. and Peytcheva, E. (2008) 'The impact of nonresponse rates on nonresponse bias: a meta-analysis', *Public Opinion Quarterly*, 72(2), pp. 167-189. https://doi.org/10.1093/poq/nfn011

Kennedy, C., Mercer, A., Keeter, S., Hatley, N., McGeeney, K. and Gimenez, A. (2016) *Evaluating online nonprobability surveys*. Washington, DC: Pew Research Center.

Lax, J.R. and Phillips, J.H. (2009) 'How should we estimate public opinion in the states?', *American Journal of Political Science*, 53(1), pp. 107-121. https://doi.org/10.1111/j.1540-5907.2008.00360.x

Warner, S.L. (1965) 'Randomized response: a survey technique for eliminating evasive answer bias', *Journal of the American Statistical Association*, 60(309), pp. 63-69.
