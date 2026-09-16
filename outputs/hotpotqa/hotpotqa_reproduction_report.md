# Bao cao tong hop HotpotQA reproduction

Ngay tao bao cao: 2026-08-24

## 1. Pham vi va nguon du lieu

Bao cao nay tong hop tat ca run trong `outputs/hotpotqa` cho 7 method:

- `standard`
- `cot`
- `cot-sc`
- `act`
- `react`
- `cot-sc-react`
- `react-cot-sc`

Tat ca run deu dung task `hotpotqa`, dataset `hotpotqa/hotpot_qa:distractor:validation`, model `Qwen/Qwen2.5-7B-Instruct`, `num_samples=500`, `seed=42`, `max_new_tokens=256`, va greedy decoding cho cac leg khong phai CoT-SC (`temperature=0.0`, `top_p=1.0`).

So sanh paper dua tren `main_paper_ReACT.pdf`, Table 1: PaLM-540B prompting results on HotpotQA and Fever. Gia tri HotpotQA cua paper la EM:

| Method | Paper HotpotQA EM |
|---|---:|
| Standard | 28.7 |
| CoT | 29.4 |
| CoT-SC | 33.4 |
| Act | 25.7 |
| ReAct | 27.4 |
| CoT-SC -> ReAct | 34.2 |
| ReAct -> CoT-SC | 35.1 |
| Supervised SOTA | 67.5 |

Paper cung mo ta CoT-SC la sampling 21 CoT trajectories voi decoding temperature 0.7 va lay majority answer. Hybrid `CoT-SC -> ReAct` fallback sang ReAct khi majority answer trong `n` samples xuat hien it hon `n/2` lan. Voi `n=21`, threshold paper la `10.5`, tuc winner can it nhat 11/21 de duoc coi la confident.

## 2. Ket qua tong hop reproduction

| Rank | Method | Correct / 500 | EM (%) | F1 (%) | Precision (%) | Recall (%) | Avg steps | Avg tool calls | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ReAct -> CoT-SC | 157 | 31.4 | 43.54 | 45.37 | 45.01 | 15.25 | 4.33 | 132.0 min |
| 2 | CoT-SC -> ReAct | 146 | 29.2 | 38.38 | 40.55 | 38.87 | 24.14 | 2.68 | 235.9 min |
| 3 | Act | 135 | 27.0 | 36.43 | 38.77 | 36.89 | 3.53 | 2.46 | 23.0 min |
| 4 | ReAct | 129 | 25.8 | 34.49 | 35.77 | 35.47 | 5.14 | 4.35 | 195.7 min |
| 5 | CoT-SC | 118 | 23.6 | 33.37 | 34.99 | 34.24 | 21.00 | 0.00 | 54.2 min |
| 6 | CoT | 108 | 21.6 | 30.63 | 31.49 | 32.31 | 1.00 | 0.00 | 2.4 min |
| 7 | Standard | 92 | 18.4 | 27.74 | 29.15 | 28.21 | 1.00 | 0.00 | 0.4 min |

Nhan xet nhanh:

- Best reproduction la `ReAct -> CoT-SC`: 31.4 EM / 43.54 F1.
- Hai hybrid dung dau, dung cung xu huong paper: ket hop internal knowledge cua CoT-SC voi external knowledge cua ReAct tot hon tung nhom rieng le.
- `Act` dat 27.0 EM, cao hon `ReAct` 25.8 EM trong reproduction nay. Diem nay nguoc voi Table 1 cua paper, noi `ReAct` > `Act`.
- `CoT-SC` chi tang +2.0 pp EM so voi `CoT` (23.6 vs 21.6), thap hon muc tang paper (+4.0 pp, 33.4 vs 29.4).
- Supporting fact va joint metrics deu bang 0 vi cac method hien tai khong sinh `predicted_supporting_facts`; do do so sanh chinh nen dung answer EM/F1.

## 3. So sanh voi paper

| Method | Reproduction EM (%) | Paper EM (%) | Chenh lech (pp) |
|---|---:|---:|---:|
| Standard | 18.4 | 28.7 | -10.3 |
| CoT | 21.6 | 29.4 | -7.8 |
| CoT-SC | 23.6 | 33.4 | -9.8 |
| Act | 27.0 | 25.7 | +1.3 |
| ReAct | 25.8 | 27.4 | -1.6 |
| CoT-SC -> ReAct | 29.2 | 34.2 | -5.0 |
| ReAct -> CoT-SC | 31.4 | 35.1 | -3.7 |

Thu tu cua paper:

`ReAct -> CoT-SC` > `CoT-SC -> ReAct` > `CoT-SC` > `CoT` > `Standard` > `ReAct` > `Act`

Thu tu reproduction:

`ReAct -> CoT-SC` > `CoT-SC -> ReAct` > `Act` > `ReAct` > `CoT-SC` > `CoT` > `Standard`

Diem giong paper:

- `ReAct -> CoT-SC` van la method manh nhat tren HotpotQA.
- Hai hybrid van tot hon tung method don le.
- CoT-SC van tot hon CoT, va CoT van tot hon Standard.
- ReAct/Act dung Wikipedia tools co loi the lon hon closed-book prompting cua Qwen-7B trong run nay.

Diem khac paper:

- Closed-book methods thap hon paper kha nhieu, dac biet `Standard`, `CoT`, `CoT-SC`. Nguyen nhan chinh hop ly la model khac rat lon: reproduction dung Qwen2.5-7B-Instruct, paper dung PaLM-540B.
- `Act` cao hon `ReAct`, nguoc voi paper. Trong run nay ReAct co nhieu `max_steps_exceeded` va tool-query loop, nen reasoning khong luon giup.
- Hybrid gap so voi paper nho hon gap cua closed-book methods, cho thay external Wikipedia interaction bu dap kha nhieu cho model nho.

## 4. Cau hinh va tinh cong bang khi so sanh

Tat ca run co cung 500 example va cung thu tu example ID.

Tuy nhien, co mot caveat quan trong: cac method khong chay tren cung mot `code_version`/`batch_size`.

| Method | Code version | Batch size | Method settings |
|---|---:|---:|---|
| Standard | 0.10.0 | 32 | `{}` |
| CoT | 0.10.0 | 16 | `{}` |
| CoT-SC | 0.10.0 | 16 | `cot_sc_samples=21`, `cot_sc_temperature=0.7` |
| Act | 0.10.0 | 16 | `{}` |
| ReAct | 0.8.1 | not recorded | `react_best_effort_finalization=false` |
| CoT-SC -> ReAct | 0.9.0 | 3 | `cot_sc_samples=21`, `cot_sc_temperature=0.7`, `cot_sc_fallback_threshold=10.5`, `react_best_effort_finalization=false` |
| ReAct -> CoT-SC | 0.9.0 | 8 | `cot_sc_samples=21`, `cot_sc_temperature=0.7`, `cot_sc_fallback_threshold=10.5`, `react_best_effort_finalization=false` |

Anh nen doc ket qua nay la "benchmark reproduction trong trang thai repo hien tai", khong phai mot ablation hoan toan controlled. Neu can so sanh paper-chat chat hon, nen rerun 7 method cung mot code version, cung batch policy hop ly, cung prompt va cung Wikipedia snapshot/API.

## 5. Termination behavior

| Method | Termination reasons |
|---|---|
| Standard | `completed`: 500 |
| CoT | `completed`: 500 |
| CoT-SC | `cot_sc_consensus`: 500 |
| Act | `completed`: 462, `completed_after_action_loop`: 8, `completed_at_step_limit`: 30 |
| ReAct | `completed`: 279, `max_steps_exceeded`: 197, `action_loop`: 24 |
| CoT-SC -> ReAct | `cot_sc_consensus`: 216, `fallback_react_completed`: 128, `hybrid_failed`: 156 |
| ReAct -> CoT-SC | `react_completed`: 260, `fallback_cot_sc`: 240 |

Nhan xet:

- Standalone `ReAct` chi complete tu nhien 279/500. 197/500 bi `max_steps_exceeded`, 24/500 bi `action_loop`. Day la ly do ReAct khong vuot Act trong reproduction.
- `ReAct -> CoT-SC` fallback CoT-SC cho 240 cau ma ReAct khong complete. Tuy nhien nhom fallback nay rat kho: chi 35/240 dung.
- `CoT-SC -> ReAct` giu CoT-SC cho 216 cau du threshold, fallback ReAct cho 284 cau; trong do 128 completed, 156 failed.

## 6. Phan tich hybrid

### 6.1 ReAct -> CoT-SC

| Path | N | Correct | EM (%) | F1 (%) |
|---|---:|---:|---:|---:|
| `react_completed` | 260 | 122 | 46.92 | 61.28 |
| `fallback_cot_sc` | 240 | 35 | 14.58 | 24.31 |

Ket luan:

- Method thang tong the vi nhom ReAct completed rat manh.
- Fallback CoT-SC chu yeu la nhom ReAct that bai, nen EM thap la hop ly.
- Du fallback CoT-SC yeu, no van cuu them mot so cau so voi standalone ReAct: `react-cot-sc` dung 48 cau ma `react` sai, nhung cung mat 20 cau ma standalone `react` dung. Luu y so nay bi anh huong boi code version/batch khac nhau.

### 6.2 CoT-SC -> ReAct

| Path | N | Correct | EM (%) | F1 (%) |
|---|---:|---:|---:|---:|
| `cot_sc_consensus` | 216 | 102 | 47.22 | 57.61 |
| `fallback_react_completed` | 128 | 44 | 34.38 | 52.72 |
| `hybrid_failed` | 156 | 0 | 0.00 | 0.00 |

Ket luan:

- Paper heuristic hoat dong dung huong: nhom CoT-SC co winner >= 11/21 dat EM 47.22, cao hon rat nhieu so voi average CoT-SC 23.6.
- Fallback ReAct cung co ich: 44 cau duoc cuu trong nhom CoT-SC low-confidence.
- Nut that lon la 156 cau `hybrid_failed`; neu giam duoc fail nay, `cot-sc-react` co the tang manh.

## 7. Phan tich CoT-SC

### 7.1 Ket qua va confidence

Standalone `cot-sc`:

- `cot_sc_samples`: 21 cho moi example.
- `cot_sc_temperature`: 0.7.
- `cot_sc_valid_answers`: 21 cho 500/500 example.
- Mean winning count: 10.12/21.
- Median winning count: 9/21.

Accuracy theo winning count:

| Winning count bucket | N | Correct | EM (%) |
|---|---:|---:|---:|
| 1-3 | 109 | 4 | 3.67 |
| 4-5 | 67 | 4 | 5.97 |
| 6-10 | 114 | 13 | 11.40 |
| 11-15 | 70 | 18 | 25.71 |
| 16-21 | 140 | 79 | 56.43 |

Nhan xet:

- Confidence cua CoT-SC co tuong quan rat manh voi correctness.
- Cac case winner 1-5/21 gan nhu khong dang tin, nhung standalone `cot-sc` van gan `termination_reason="cot_sc_consensus"` mien la co prediction. Ve logic vote thi van dung plurality vote, nhung ten `consensus` hoi gay hieu nham.
- Neu viet bao cao paper-style cho standalone CoT-SC, nen report them `winning_count`/`confidence`, hoac tach `plurality_vote` voi `paper_threshold_met`.

### 7.2 Sample 1 den 21 co lap khong?

Tu `trajectories.jsonl` cua standalone CoT-SC:

- 500/500 examples co 21 CoT steps.
- Unique exact thought:
  - min: 3/21
  - mean: 19.11/21
  - median: 21/21
  - max: 21/21
- Khong co example nao 21 thought exact-same hoan toan.
- 19/500 examples co mot thought exact duplicate >= 10 lan.
- 6/500 examples co mot thought exact duplicate >= 15 lan.

Ket luan:

- Viec nhieu sample cho cung answer hoac thought rat gan nhau la binh thuong voi self-consistency, nhat la khi model tu tin.
- Algorithm hien tai giong paper o diem quan trong: 21 CoT samples, temperature 0.7, majority/plurality normalized answer.
- Dieu dang can than khong phai "lap thought" tu than, ma la:
  - hallucination co confidence cao, vi model co the lap cung mot sai lam;
  - low-confidence plurality winner van bi dat ten `cot_sc_consensus`;
  - closed-book CoT-SC tren Qwen-7B yeu hon PaLM-540B kha nhieu.

## 8. Overlap giua cac method

So voi `standard`:

| Method | Both correct with Standard | New correct vs Standard | Lost correct vs Standard |
|---|---:|---:|---:|
| CoT | 69 | 39 | 23 |
| CoT-SC | 74 | 44 | 18 |
| Act | 61 | 74 | 31 |
| ReAct | 51 | 78 | 41 |
| CoT-SC -> ReAct | 74 | 72 | 18 |
| ReAct -> CoT-SC | 74 | 83 | 18 |

Mot so cap dang chu y:

| Comparison | Gain | Loss | Both correct |
|---|---:|---:|---:|
| ReAct -> CoT-SC vs ReAct | 48 | 20 | 109 |
| ReAct -> CoT-SC vs CoT-SC | 61 | 22 | 96 |
| CoT-SC -> ReAct vs CoT-SC | 41 | 13 | 105 |
| CoT-SC -> ReAct vs ReAct | 56 | 39 | 90 |
| CoT-SC vs CoT | 17 | 7 | 101 |
| ReAct vs Act | 44 | 50 | 85 |

Nhan xet:

- Hybrid co loi the that, khong chi la cong them runtime: no dung them nhieu cau ma method nen sai.
- `ReAct` vs `Act` co nhieu trade-off: ReAct dung them 44 cau so voi Act, nhung mat 50 cau Act dung. Day phu hop voi paper analysis rang ReAct grounded hon nhung structural constraint/action loop co the gay reasoning/search error.

## 9. Dien giai so voi paper

### 9.1 Vi sao closed-book thap hon paper?

Paper dung PaLM-540B, reproduction dung Qwen2.5-7B-Instruct. HotpotQA question-only yeu cau internal knowledge lon va multi-hop reasoning. Vi vay gap -7.8 den -10.3 pp o `Standard`, `CoT`, `CoT-SC` la hop ly.

### 9.2 Vi sao Act/ReAct gan paper hon?

Act/ReAct co Wikipedia interaction nen khong phu thuoc hoan toan vao parametric memory cua model. `Act` thuc te cao hon paper +1.3 pp; `ReAct` chi thap hon paper -1.6 pp. Dieu nay goi y tool/environment dang bu dap dang ke cho kich thuoc model.

### 9.3 Vi sao ReAct khong vuot Act trong reproduction?

Standalone `ReAct` co 221/500 non-clean terminations (`max_steps_exceeded` hoac `action_loop`), trong khi Act co 38/500 non-standard completions. Nhieu ReAct trajectories bi ket trong lookup/search reformulation hoac khong `Finish` kip trong 7 steps. Paper cung ghi ReAct co failure mode "reasoning error" va "failing to recover from repetitive steps"; reproduction nay co ve bi mode do kha ro.

### 9.4 Vi sao ReAct -> CoT-SC van tot nhat?

Nhom ReAct completed cua hybrid rat manh: 122/260 dung. Khi ReAct khong xong, CoT-SC fallback cuu them 35 cau. Tong lai dat 157/500, cao nhat. Day trung voi ket luan Table 1 cua paper tren HotpotQA: `ReAct -> CoT-SC` la best prompting method.

## 10. Khuyen nghi tiep theo

1. Rerun tat ca methods tren cung code version moi nhat.
   - Hien tai `react` la 0.8.1, hybrids la 0.9.0, cac method con lai la 0.10.0.
   - Nen rerun `react`, `react-cot-sc`, `cot-sc-react` bang 0.10.0 de so sanh sach hon.

2. Doi ten termination cua standalone CoT-SC.
   - Hien tai moi prediction co answer deu la `cot_sc_consensus`.
   - Nen them metadata/termination nhu `cot_sc_plurality`, `cot_sc_threshold_met`, `cot_sc_low_confidence`.

3. Report CoT-SC confidence trong ket qua chinh.
   - Winner 1-5/21 co EM rat thap.
   - Winner 16-21/21 co EM ~56.4, rat dang tin hon.

4. Debug ReAct loops.
   - Standalone ReAct co 197 `max_steps_exceeded` va 24 `action_loop`.
   - Nen xem top patterns: repeated `Search[founding year ...]`, lookup phrase qua dai, hoac khong chuyen sang `Finish` khi observation da du.

5. Them supporting-fact extraction neu muon so voi official HotpotQA day du.
   - Hien tai answer metrics co y nghia, nhung supporting/joint metrics bang 0 vi khong generate support facts.

6. Neu muc tieu la replicate paper Table 1, can ghi ro model difference.
   - Ket qua nay la reproduction bang Qwen2.5-7B-Instruct, khong phai PaLM-540B.
   - So sanh voi paper nen doc la "trend/relative behavior", khong phai absolute reproduction.

## 11. Ket luan

Reproduction dat pattern quan trong nhat cua paper: hybrid `ReAct -> CoT-SC` la tot nhat tren HotpotQA, va hai hybrid deu vuot tung method don le. Tuy nhien, absolute EM thap hon paper o cac method closed-book do model nho hon nhieu. `Act` vuot `ReAct` trong run nay, chu yeu vi ReAct bi nhieu loop/max-step failure. CoT-SC implementation ve co ban giong paper, va sample 1-21 lap/giong nhau o mot so cau la hanh vi sampling binh thuong; diem nen sua la cach gan nhan `cot_sc_consensus` cho ca cac vote rat thap.
