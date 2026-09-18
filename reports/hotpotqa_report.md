# HotpotQA Reproduction Results

## Experimental Setup

This report analyzes the seven completed 500-example HotpotQA runs stored under `outputs/hotpotqa/`. For each method, the selected run is the largest completed compatible run; the newest timestamp is used only to break ties. A run is considered compatible when its task and method metadata agree with its directory, its configured sample count matches both `metrics.json` and the number of prediction records, all example identifiers are unique, and `run.log` reaches the final metrics write without an error. Under this rule, every method has one eligible full run and no smoke run is selected.

All runs use the `hotpotqa/hotpot_qa:distractor:validation` dataset, 500 examples, seed 42, Qwen/Qwen2.5-7B-Instruct, a seven-step agent budget, and deterministic decoding (`temperature=0.0`, `top_p=1.0`) outside the self-consistency samples. CoT-SC and both hybrids draw 21 reasoning samples at temperature 0.7. ReAct best-effort finalization is disabled wherever the setting is recorded.

The runs are historically complete but not a perfectly controlled ablation. Standard, CoT, CoT-SC, and Act record code version 0.10.0; the two hybrids record 0.9.0; standalone ReAct records 0.8.1 and omits its batch size. The remaining batch sizes range from 3 to 32. Model and dataset revisions were not recorded for these older runs. These differences should be kept in mind when comparing small score gaps.

## Evaluated Methods

Standard and CoT are closed-book, single-completion baselines. CoT-SC samples 21 chains and selects the normalized plurality answer. Act interacts with Wikipedia without an explicit thought field, whereas ReAct alternates a thought, a tool action, and a real environment observation. The two hybrids follow the implementation rather than the lexical order of their filenames: ReAct → CoT-SC runs ReAct first and invokes CoT-SC only if ReAct does not complete; CoT-SC → ReAct retains a CoT-SC answer when at least 11 of 21 samples support the winner and otherwise falls back to ReAct.

## Main Results

Answer exact match (EM) is the primary metric implemented for this benchmark. Token-level F1 is reported as a complementary measure. Supporting-fact and joint metrics are zero because these agents do not emit supporting-fact predictions; they should not be interpreted as evidence-retrieval scores.

| Method | Correct | EM (%) | F1 (%) | Avg. steps | Avg. tool calls | Runtime (min) |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 92 / 500 | 18.4 | 27.74 | 1.000 | 0.000 | 0.42 |
| CoT | 108 / 500 | 21.6 | 30.63 | 1.000 | 0.000 | 2.44 |
| CoT-SC | 118 / 500 | 23.6 | 33.37 | 21.000 | 0.000 | 54.16 |
| Act | 135 / 500 | 27.0 | 36.43 | 3.534 | 2.456 | 22.99 |
| ReAct | 129 / 500 | 25.8 | 34.49 | 5.144 | 4.352 | 195.66 |
| ReAct → CoT-SC | 157 / 500 | 31.4 | 43.54 | 15.250 | 4.332 | 131.99 |
| CoT-SC → ReAct | 146 / 500 | 29.2 | 38.38 | 24.138 | 2.678 | 235.86 |

The two hybrids achieved the highest EM and F1, with ReAct → CoT-SC leading this set at 31.4 EM. Among single strategies, Act reached 27.0 EM, slightly above ReAct at 25.8. CoT and CoT-SC improved on Standard, but the 2.0-point EM gain from self-consistency came with a much larger inference burden: 21 recorded reasoning samples per example and a runtime of 54.16 minutes rather than 2.44 minutes.

## Agent Behavior and Retrieval Characteristics

Standalone ReAct completed with a final answer on 279 examples, exhausted the seven-step budget on 197, and stopped on an action loop on 24. All 129 exact-match successes occurred among the completed trajectories, giving that subset 46.2% EM; neither non-completion category produces an answer. Across the run, ReAct issued 1,811 searches and 480 lookups. Its 346 lookup misses and the presence of repeated Search actions in 116 trajectories show that the principal cost was not merely using Wikipedia, but recovering from unproductive retrieval choices within a short interaction budget.

The trajectories also show successful multi-hop behavior. For example, on the question asking which of the genera Greyia and Calibanus contains more species, ReAct opened the article for each genus, extracted counts of three and two species, and finished with “Greyia” in three steps. This is a direct instance of two retrieved facts being compared before answering.

The failure modes are equally concrete. For the question about the founding year of the university where Sergei Aleksandrovich Tokarev taught, ReAct correctly retrieved Moscow State University in its first two steps. It then made two unsuccessful lookups for “founding year” and “founding,” followed by three increasingly specific searches that returned the same short lead. The trajectory reached step seven without emitting `Finish[1755]`. The intermediate entity was found; the failure arose from an ineffective second-hop retrieval strategy and subsequent budget exhaustion.

Act used fewer tools on average and finished all examples under its own finalization policy. Its termination counts were 462 ordinary completions, 8 completions after an action loop, and 30 completions at the step limit. This difference helps explain why Act can outperform ReAct in this run even though ReAct has access to explicit reasoning: a useful thought does not improve EM if the agent never converts the accumulated evidence into a final answer.

## Comparison of ReAct, CoT-SC, and Hybrid Methods

ReAct → CoT-SC retained 260 ReAct completions, of which 122 were exact matches (46.9%). The other 240 examples invoked CoT-SC; 35 of those were exact matches (14.6%). The fallback subset is therefore substantially harder than the completed-ReAct subset, so its lower score should not be read as a direct head-to-head comparison between the component methods. Nevertheless, the fallback recovered enough answers to raise the aggregate from standalone ReAct’s 25.8 EM to 31.4 EM, subject to the historical code-version caveat.

CoT-SC → ReAct accepted the initial CoT-SC consensus for 216 examples, producing 102 exact matches (47.2%). Of the 284 low-confidence cases, ReAct completed 128 and answered 44 correctly (34.4%); the remaining 156 hybrid trajectories failed to produce an answer. The policy therefore identifies a high-quality confident subset, but its total cost is large: every example first incurs 21 CoT samples, and low-confidence examples then add an agent trajectory.

These path-level results clarify why the hybrid ordering matters. ReAct → CoT-SC spends retrieval effort first and uses a closed-book vote to cover missing completions. CoT-SC → ReAct spends 21 generations on every example and reserves retrieval for the uncertain tail. The former obtained higher EM and a lower measured runtime in these runs, while the latter made fewer tool calls on average.

## Error and Failure Analysis

Three recurring patterns are visible in the stored trajectories. First, some questions fail at the entity-linking stage because the search query is too specific or ambiguous. Second, other trajectories retrieve the correct first-hop entity but fail to locate the relation needed for the second hop, as in the Tokarev example. Third, ReAct sometimes continues searching or looking up after the available observation already supports an answer, causing premature budget exhaustion rather than an incorrect final string.

The termination distribution is important when interpreting empty predictions. An empty ReAct prediction caused by `max_steps_exceeded` is an agent-completion failure, not an answer-parser error. Similarly, the 24 `action_loop` terminations reflect the historical loop policy recorded by that run. Only two malformed-action steps appear in the standalone ReAct trajectories, so formatting errors are not the dominant explanation for its 221 non-completions on HotpotQA.

## Efficiency Analysis

Standard is both the least expensive and the least accurate method in this sample. CoT adds 2.9 EM points over Standard at modest cost, whereas CoT-SC adds another 2.0 points but takes roughly 22 times the CoT runtime. Interactive methods incur costs of a different kind: Act averages 2.456 Wikipedia calls and ReAct 4.352. ReAct → CoT-SC offers the strongest accuracy–cost compromise among the two hybrids here, improving EM by 2.2 points over CoT-SC → ReAct while recording 103.87 fewer runtime minutes, although heterogeneous batch sizes prevent treating wall-clock time as a hardware-normalized efficiency measurement.

Recorded “steps” are not homogeneous across all methods. A CoT-SC step represents one sampled reasoning completion, while a ReAct step represents one logical Thought/Action turn and may execute a tool. Average steps are therefore useful for accounting within a method but should not be interpreted as equal computational units across methods.

## Discussion

The results suggest that Wikipedia interaction partially offsets the limitations of a smaller modern instruction-tuned model on open-ended multi-hop questions. Act and ReAct both exceed the closed-book baselines, but explicit ReAct reasoning is coupled to a fragile control problem: the model must choose an effective query, interpret the observation, and finish before the budget expires. The hybrids perform more consistently because they can combine retrieved evidence with parametric reasoning rather than requiring every example to succeed through a single route.

The repository also contains the original ReAct paper, whose reported values were obtained with a different model and serving stack. Absolute comparisons to PaLM-540B are therefore descriptive rather than controlled. The reproduction supports analysis of protocol behavior and relative trade-offs; it does not isolate model scale, prompt version, Wikipedia state, batching, or code version.

## Limitations

The seven HotpotQA methods were not rerun from one code revision, and the standalone ReAct batch size is missing. The runs do not pin a model revision or Wikipedia snapshot. Supporting facts are not predicted, leaving official supporting-fact and joint scores at zero. Finally, the 500 examples are a seeded subset rather than the entire validation split, so small differences should not be presented as statistically significant without an explicit uncertainty analysis.

## Summary

On the selected 500-example HotpotQA runs, ReAct → CoT-SC achieved the highest answer performance at 31.4 EM and 43.54 F1, followed by CoT-SC → ReAct at 29.2 EM. Retrieval improved over the closed-book baselines, but standalone ReAct was limited by 221 trajectories that ended in a loop or without a final answer. The strongest evidence for the hybrids is not simply their aggregate ranking: their execution paths show that high-quality ReAct completions and high-confidence CoT-SC votes cover different portions of the benchmark.
