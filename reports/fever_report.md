# FEVER Reproduction Results

## Experimental Setup

This report analyzes the seven completed 500-example FEVER runs under `outputs/fever/`. The selection procedure rejects incomplete or internally inconsistent directories, prefers the largest compatible run for each task and method, and uses the newest timestamp only as a tie-breaker. All seven selected directories contain 500 unique prediction records, end with a metrics write in `run.log`, and use the same ordered list of sampled example identifiers. No three-example smoke run enters the analysis.

The runs use the `paper_dev` subset and `dev` split of `ysymyth/ReAct`, pinned to revision `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`. They use Qwen/Qwen2.5-7B-Instruct at model revision `a09a35458c702b33eeacc393d103063234e8bc28`, code version 0.11.0, seed 42, and prompt version `fever-appendix-c2-v1`. Normal generations use temperature 0.0, top-p 1.0, and 256 new tokens. The FEVER agent budget is five steps. CoT-SC and the hybrid methods use 21 samples at temperature 0.7 and the paper threshold of 10.5 votes; ReAct best-effort finalization is disabled. Batch size varies by method from 8 to 32, as recorded in each configuration.

FEVER assigns each claim one of three labels. `SUPPORTS` means the evidence entails the claim, `REFUTES` means it contradicts the claim, and `NOT ENOUGH INFO` (NEI) means the available evidence is insufficient for either judgment. The selected sample contains 174 SUPPORTS, 142 REFUTES, and 184 NEI claims.

## Evaluated Methods

Standard, CoT, and CoT-SC make closed-book verification decisions. Act and ReAct can issue Wikipedia `Search` and `Lookup` actions before returning `Finish[label]`; only the environment supplies observations. ReAct → CoT-SC uses CoT-SC when ReAct fails to finish, whereas CoT-SC → ReAct invokes ReAct only when the 21-sample vote does not exceed the `n/2` confidence threshold. These directions are confirmed by both the implementation and the stored execution-path metadata.

## Main Results

Accuracy is the primary FEVER metric. `Invalid / unparsed` counts examples without a valid canonical label; for an agent, this can result from budget exhaustion as well as malformed output.

| Method | Correct | Accuracy (%) | Invalid / unparsed | Avg. steps | Avg. tool calls | Runtime (min) |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 258 / 500 | 51.6 | 2 | 1.000 | 0.000 | 1.31 |
| CoT | 280 / 500 | 56.0 | 5 | 1.000 | 0.000 | 2.93 |
| CoT-SC | 286 / 500 | 57.2 | 0 | 21.000 | 0.000 | 62.93 |
| Act | 271 / 500 | 54.2 | 39 | 2.806 | 1.802 | 27.98 |
| ReAct | 164 / 500 | 32.8 | 258 | 4.058 | 3.432 | 79.42 |
| ReAct → CoT-SC | 308 / 500 | 61.6 | 0 | 15.064 | 3.446 | 170.95 |
| CoT-SC → ReAct | 275 / 500 | 55.0 | 15 | 21.216 | 0.190 | 118.03 |

ReAct → CoT-SC achieved the highest accuracy, 61.6%, and eliminated invalid final predictions through its CoT-SC fallback. CoT-SC was the strongest non-hybrid method at 57.2%, followed by CoT at 56.0%. Standalone ReAct was substantially lower because more than half of its examples ended without a label. The table therefore reflects both decision quality and completion reliability.

## Label-Level Performance

| Method | SUPPORTS (%) | REFUTES (%) | NEI (%) |
|---|---:|---:|---:|
| Standard | 55.7 | 74.6 | 29.9 |
| CoT | 75.9 | 68.3 | 27.7 |
| CoT-SC | 78.7 | 68.3 | 28.3 |
| Act | 52.9 | 73.2 | 40.8 |
| ReAct | 37.9 | 35.9 | 25.5 |
| ReAct → CoT-SC | 78.7 | 70.4 | 38.6 |
| CoT-SC → ReAct | 75.3 | 66.9 | 26.6 |

NEI is the hardest label for every method. Standard, for example, predicts REFUTES for 97 of 184 NEI claims and achieves only 29.9% NEI accuracy. CoT and CoT-SC improve SUPPORTS accuracy to 75.9% and 78.7%, respectively, but leave NEI below 29%. Act raises NEI accuracy to 40.8%, the highest individual value in the table, while losing ground on SUPPORTS and returning 39 invalid predictions. ReAct → CoT-SC combines the CoT-SC SUPPORTS score with a higher REFUTES score and a 10.3-point improvement on NEI relative to standalone CoT-SC.

These class differences indicate that aggregate accuracy is not driven uniformly. The model often commits to a positive or negative label when the gold decision is insufficient evidence. Self-consistency stabilizes the output format and strengthens SUPPORTS performance, but agreement among closed-book samples does not by itself resolve the epistemic distinction represented by NEI.

## ReAct and Wikipedia Interaction Behavior

Standalone ReAct completed 242 examples, reached the five-step limit on 243, and ended with `parsing_error` on 15. The completed subset contains all 164 correct answers and attains 67.8% accuracy. This conditional score shows that ReAct decisions can be strong when the tool loop reaches `Finish[...]`; the aggregate 32.8% is dominated by completion failures rather than by 336 explicit but wrong labels.

The retrieval traces explain much of that completion problem. ReAct issued 1,705 Search actions, and 1,469 observations reported `Could not find [...]` with suggestions. It used only 11 Lookup actions. The imbalance is characteristic of overly specific searches that repeatedly miss exact Wikipedia titles. Eighty-one trajectories repeat at least one exact Search action. These are search-strategy failures by the model, not evidence that the environment should silently choose a suggested page.

The Peking University claim is representative. ReAct successively searches for “Peking University history,” “Peking University founded,” “Peking University establishment date,” “Peking University founding year,” and a longer variant. Each observation returns suggestions that include `Peking University`, but the model never issues `Search[Peking University]`; the five-step budget expires without a label. The analogous Tenacious D trajectory behaves similarly. In contrast, the Yandex claim completes in two steps: a direct `Search[Yandex]` opens the company article, and the next action returns `Finish[REFUTES]`.

The current ReAct implementation performs one in-step action-format recovery call when a Thought/Action completion cannot be parsed. The recovery is batched only over failed states, stops at a newline, and does not consume an extra logical environment step. In the standalone FEVER trajectory serialization, 29 steps contain an action-only model output consistent with a successful recovery, while 71 steps across 68 examples record an invalid action after the one permitted recovery also failed. Only 15 examples terminate as `parsing_error`; an earlier malformed step can be followed by later valid actions, and a final non-parsing step can instead end at the budget. Thus, malformed generation is real but does not account for all 258 invalid predictions.

## Self-Consistency and Hybrid Execution Paths

Standalone CoT-SC produces a valid label for all 500 examples. Its vote is a strict majority on 482 examples and a lower-confidence plurality on 18; the corresponding accuracies are 57.9% and 38.9%. The sample therefore supports the intended use of vote concentration as a routing signal, although the low-confidence subset is small.

ReAct → CoT-SC keeps 238 completed ReAct decisions. Those cases contain 163 correct labels, or 68.5% accuracy, with 1.857 average tool calls. The other 262 examples take the CoT-SC fallback, which recovers 145 correct labels (55.3%) and removes every empty prediction. Since the fallback follows a full failed ReAct attempt, these examples average 26 recorded steps and 4.889 tool calls. The hybrid gains robustness, but at the highest runtime in the FEVER table.

CoT-SC → ReAct accepts the initial vote for 475 examples and answers 271 correctly (57.1%). Only 25 examples reach ReAct: 10 complete, of which 4 are correct, and 15 fail without a label. Consequently, the fallback reduces the overall score from standalone CoT-SC’s 57.2% to 55.0% in this sample. This does not imply that retrieval is generally harmful; it shows that ReAct was applied to a particularly uncertain subset and completed only 40% of those fallback cases.

## Error and Failure Analysis

The artifacts support five distinct failure categories.

First, protocol-level safeguards are functioning: model-generated Observation text is truncated before parsing, and the repository records only environment observations. Second, malformed actions can survive the single recovery attempt; these appear as invalid trajectory steps, with 15 standalone ReAct examples ending in a parsing error. Third, 243 ReAct examples exhaust the environment budget without `Finish[...]`; their empty predictions are max-step failures, not label-parser bugs. Fourth, many trajectories use over-specified Search queries instead of following an exact-title suggestion, leading to repeated retrieval misses. Fifth, completed trajectories can still interpret evidence incorrectly: 78 of ReAct’s 242 completed examples receive the wrong label.

Act presents a different profile. It reaches 409 ordinary completions and 52 completions at the step limit, while 39 finalizations fail. Its 39 invalid predictions therefore should not all be described as malformed labels: 36 invalid-action steps appear in its trajectories, but termination depends on what happens at the forced final step. Standard and CoT have only 2 and 5 invalid labels, respectively, and CoT-SC has none.

## Efficiency Analysis

Closed-book Standard and CoT finish in 1.31 and 2.93 minutes, respectively. CoT-SC raises accuracy by 1.2 points over CoT but requires 62.93 minutes for 21 samples per example. Act uses 1.802 tools per example and finishes in 27.98 minutes. Standalone ReAct uses almost twice as many tools as Act and takes 79.42 minutes while producing many empty outputs.

The hybrids make different cost commitments. ReAct → CoT-SC averages 3.446 tool calls and runs for 170.95 minutes because 262 failed ReAct cases also incur 21 CoT samples. CoT-SC → ReAct averages only 0.190 tool calls because 475 examples stop after the vote, but it still spends 21 sampled generations on every example. Wall-clock comparisons are descriptive: batch sizes differ and the logs do not provide a hardware-normalized throughput measure.

## Discussion

FEVER differs from HotpotQA in both output space and retrieval objective. Its three labels make answer normalization easier than open-ended QA, but the central challenge is deciding whether retrieved evidence entails, contradicts, or fails to resolve a claim. On this sample, closed-book self-consistency is effective at producing valid labels, while ReAct’s benefit depends heavily on search-title formulation and timely completion. The strongest method uses ReAct first but falls back to CoT-SC whenever the interaction path does not yield a label.

This result should not be reduced to “retrieval versus no retrieval.” Completed ReAct cases are accurate, yet the standalone agent frequently spends its budget on unsuccessful searches. The hybrid succeeds because it preserves those strong completed cases and supplies a total label for the remainder. Conversely, routing only low-confidence CoT-SC cases to the same ReAct policy does not help when that difficult subset rarely completes.

## Limitations

The 500 examples are a seeded sample of `paper_dev`, not the full source corpus. Wikipedia content and API behavior are not pinned by the run metadata, so future re-execution may observe environmental drift. Runtime is affected by batching and hardware. The report does not compute statistical significance. Finally, trajectory fields preserve the recovered completion rather than both the initial malformed generation and recovery response; successful format recovery is therefore inferred from the action-only serialized output in conjunction with the current implementation, rather than from an explicit recovery flag.

## Summary

ReAct → CoT-SC achieved 61.6% accuracy and was the only method to combine strong completed-ReAct decisions with zero invalid final labels. CoT-SC reached 57.2% without tools, while standalone ReAct fell to 32.8% because 258 examples lacked a valid final label. Trajectory evidence attributes most ReAct failures to five-step exhaustion and unproductive search formulation, with a smaller but measurable contribution from malformed actions. The hybrid results show that completion-aware fallback is more effective here than sending only low-confidence CoT-SC cases to ReAct.
