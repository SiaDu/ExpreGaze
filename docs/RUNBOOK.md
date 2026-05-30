# ExpreGaze Runbook

这是文本流水线的命令速查表。除非特别说明，所有命令都在项目根目录运行。

## 配置文件

每部电影用一个 run config 驱动：

```bash
configs/runs/text_main_tt1591095.yaml
configs/runs/text_main_tt0032138.yaml
configs/runs/text_main_tt1637725.yaml
```

默认 smoke test 可以先用 `tt1591095`：

```bash
RUN_CONFIG=configs/runs/text_main_tt1591095.yaml
```

## Stage00: 生成 Shot-Level 主表

从 MovieNet annotation 和 meta 生成 shot-level CSV/JSONL：

```bash
bash scripts/stages/00_process_one_movie_to_shot_level.sh configs/runs/text_main_tt1591095.yaml
```

主输出在 run yaml 里配置：

```text
data/processed/shot_level/tt1591095__shot_level.csv
data/processed/shot_level/tt1591095__shot_level.jsonl
```

debug 输出：

```text
outputs/text_main/tt1591095/logs/00_process_one_movie_to_shot_level.log
outputs/text_main/tt1591095/logs/00_process_one_movie_to_shot_level_summary.json
outputs/text_main/tt1591095/logs/00_process_one_movie_to_shot_level_preview.csv
```

## Stage01: 生成 Full-Context

Stage01 有三种模式。默认建议先跑 `raw`，它只生成检查文件，不调用 OpenAI。

### Raw Mode

第一步先跑这个。它会生成 raw subtitle segment、screenplay dialogue segment、
alignment steps，以及 LLM repair candidate 表。

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1591095.yaml raw --overwrite
```

debug 输出：

```text
outputs/text_main/tt1591095/logs/01_raw_segments.csv
outputs/text_main/tt1591095/logs/01_script_dialogue_segments.csv
outputs/text_main/tt1591095/logs/01_raw_alignment_steps.csv
outputs/text_main/tt1591095/logs/01_llm_repair_candidates.csv
outputs/text_main/tt1591095/logs/01_align_summary.json
outputs/text_main/tt1591095/logs/01_full_context_preview.csv
```

候选表就在这里看：

```bash
less -S outputs/text_main/tt1591095/logs/01_llm_repair_candidates.csv
```

summary 可以这样看：

```bash
cat outputs/text_main/tt1591095/logs/01_align_summary.json
```

重点检查这些字段：

```text
raw_segment_count
dialogue_segment_count
match_count
confident_match_count
llm_candidate_count
```

### Final Mode

raw 检查没问题后再跑这个。它会写纯 DP 的 `full_context`，不调用 OpenAI。

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1591095.yaml final --overwrite
```

主输出：

```text
data/processed/full_context/tt1591095__full_context.csv
data/processed/full_context/tt1591095__full_context.jsonl
```

### LLM Mode

只在 `raw` 和 `final` 看起来合理后再跑。它会修复 `subtitle_text` 非空、
但 `match_source` 为空的行。

小样本测试：

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1591095.yaml llm --max-llm-rows 3 --overwrite
```

控制批量：

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1591095.yaml llm --max-llm-rows 25 --overwrite
```

全量修复当前候选行：

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1591095.yaml llm --max-llm-rows auto --overwrite
```

`--max-llm-rows` 的含义：

```text
3     最多修 3 行，适合 smoke test
25    最多修 25 行，适合分批检查
auto  修复当前所有候选行
```

LLM 输出：

```text
data/processed/full_context/tt1591095__full_context_llm.csv
data/processed/full_context/tt1591095__full_context_llm.jsonl
outputs/text_main/tt1591095/cache/01_llm_local_fill_cache.jsonl
```

cache 文件保存 prompt、OpenAI 原始返回和解析后的 JSON。`cache_responses: true`
时，Stage01 会复用匹配的 cache，避免重复调用同一行。

## Stage02: 筛选 Candidate Sequences

Stage02 只读取非 LLM 版 `full_context.csv`，从连续 shot window 中筛选适合后续
LLM gaze script 和 video proxy 的片段。screenplay 里的镜头/动作信息会从
`prev_other_text / bridge_other_text / next_other_text / aligned_script_text` 提取，
只用于排序和 debug，不作为硬过滤条件。

```bash
bash scripts/stages/02_build_candidate_sequences.sh configs/runs/text_main_tt1591095.yaml
```

主输出：

```text
data/processed/candidate_sequences/tt1591095__candidate_sequences.csv
data/processed/candidate_sequences/tt1591095__candidate_sequences.jsonl
```

debug 输出：

```text
outputs/text_main/tt1591095/logs/02_candidate_sequences_all.csv
outputs/text_main/tt1591095/logs/02_candidate_sequences_summary.json
```

可以这样检查：

```bash
less -S data/processed/candidate_sequences/tt1591095__candidate_sequences.csv
less -S outputs/text_main/tt1591095/logs/02_candidate_sequences_all.csv
cat outputs/text_main/tt1591095/logs/02_candidate_sequences_summary.json
```

重点看这些字段：

```text
score
start_time_sec
end_time_sec
active_speakers
speaker_changes
visual_action_score
camera_cue_count
gaze_verb_count
movement_cue_count
scene_boundary_count
script_action_preview
```

## Stage03: LLM 生成 Gaze Script

Stage03 读取 Stage02 的 selected candidate sequence JSONL，按
`sequence x active_speaker` 调用 LLM，生成 per-character gaze events 和
timeline。默认每个 sequence 只跑 1 个角色，避免误花 API。

先做 dry-run，只构造任务和 prompt，不调用 OpenAI：

```bash
bash scripts/stages/03_generate_llm_gaze_script.sh configs/runs/text_main_tt1591095.yaml --max-sequences 1 --max-characters-per-sequence 1 --dry-run --overwrite
```

小样本 API 测试：

```bash
bash scripts/stages/03_generate_llm_gaze_script.sh configs/runs/text_main_tt1591095.yaml --max-sequences 1 --max-characters-per-sequence 1 --overwrite
```

批量但仍限制成本：

```bash
bash scripts/stages/03_generate_llm_gaze_script.sh configs/runs/text_main_tt1591095.yaml --max-sequences 5 --max-characters-per-sequence 1 --overwrite
```

对每个 sequence 的所有 active speakers 都生成：

```bash
bash scripts/stages/03_generate_llm_gaze_script.sh configs/runs/text_main_tt1591095.yaml --max-sequences 5 --max-characters-per-sequence auto --overwrite
```

主输出：

```text
outputs/text_main/tt1591095/llm_gaze_scripts/{sequence_id}__{MAIN_CHAR}.json
outputs/text_main/tt1591095/llm_gaze_scripts/{sequence_id}__{MAIN_CHAR}__timeline.json
```

debug/cache 输出：

```text
outputs/text_main/tt1591095/llm_gaze_scripts/prompts/
outputs/text_main/tt1591095/llm_gaze_scripts/raw_responses/
outputs/text_main/tt1591095/logs/03_generate_llm_gaze_script_summary.json
outputs/text_main/tt1591095/cache/03_llm_gaze_script_cache.jsonl
```

summary 可以这样看：

```bash
cat outputs/text_main/tt1591095/logs/03_generate_llm_gaze_script_summary.json
```

重点看：

```text
requested_count
skipped_existing_count
cache_hit_count
api_call_count
written_file_count
zero_event_output_count
validation_warning_count
```

如果 `cache_responses: true`，同一个 `sequence_id + main_char + prompt` 下次会优先复用
cache，避免重复调用 OpenAI。

## Stage04: 切 Selected Sequences / Shots

Stage04 只读取 Stage02 的 selected candidate JSONL，不复用旧的
`data/interim/shot_level/{movie_id}`。切出来的视频和 debug manifest 放在
`outputs/video_proxy/{movie_id}/`，因为它们是当前 video proxy run 的派生文件。

先做 dry-run，只写 manifest，不调用 ffmpeg：

```bash
bash scripts/stages/04_extract_video_sequences.sh configs/runs/video_proxy_tt0032138.yaml --max-sequences 2 --dry-run --overwrite
```

检查 manifest：

```bash
less -S outputs/video_proxy/tt0032138/logs/04_shot_manifest.csv
```

确认无误后实际切前 2 个 sequence：

```bash
bash scripts/stages/04_extract_video_sequences.sh configs/runs/video_proxy_tt0032138.yaml --max-sequences 2 --overwrite
```

只切指定 sequence：

```bash
bash scripts/stages/04_extract_video_sequences.sh configs/runs/video_proxy_tt0032138.yaml --sequence-id-list tt0032138_seq_0282_0290,tt0032138_seq_0590_0593 --overwrite
```

输出位置：

```text
outputs/video_proxy/tt0032138/logs/04_shot_manifest.csv
outputs/video_proxy/tt0032138/sequence_videos/{sequence_id}.mp4
outputs/video_proxy/tt0032138/shot_clips/{sequence_id}/{shot_id}.mp4
```

manifest 每行是一个 shot，重点检查：

```text
sequence_id
shot_id
shot_start
shot_end
duration
subtitle_text
aligned_speakers
cast_pids
num_cast
stage_type
process_status
note
```

`stage_type` 第一版只是粗分：

```text
single_speaking
two_person_dialogue_simple
multi_person
unknown
```

快速确认切片数量：

```bash
find outputs/video_proxy/tt0032138/sequence_videos -maxdepth 1 -name '*.mp4' | wc -l
find outputs/video_proxy/tt0032138/shot_clips -name '*.mp4' | wc -l
```

## Stage05: Face Detection + Shot-Local Tracking

Stage05 读取 Stage04 的 manifest 和 shot clips，只做 shot 内人脸检测与 tracking。
它不会做跨镜头身份链接，也不会跑 OpenFace。

第一版默认使用 YuNet，需要你把模型放到：

```text
models/face_detection/face_detection_yunet_2023mar.onnx
```

也可以运行时指定模型：

```bash
bash scripts/stages/05_run_face_detection_tracks.sh configs/runs/video_proxy_tt0032138.yaml --yunet-model /path/to/face_detection_yunet_2023mar.onnx --max-shots 5 --overwrite
```

smoke test：

```bash
bash scripts/stages/05_run_face_detection_tracks.sh configs/runs/video_proxy_tt0032138.yaml --max-shots 3 --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/face_tracks/05_face_detections.csv
outputs/video_proxy/tt0032138/face_tracks/05_face_tracks.csv
outputs/video_proxy/tt0032138/face_tracks/05_shot_track_summary.csv
outputs/video_proxy/tt0032138/logs/05_face_detection_tracks_summary.json
outputs/video_proxy/tt0032138/face_tracks/debug_overlays/
```

重点检查：

```bash
less -S outputs/video_proxy/tt0032138/face_tracks/05_face_detections.csv
less -S outputs/video_proxy/tt0032138/face_tracks/05_face_tracks.csv
less -S outputs/video_proxy/tt0032138/face_tracks/05_shot_track_summary.csv
cat outputs/video_proxy/tt0032138/logs/05_face_detection_tracks_summary.json
```

`local_track_id` 只在一个 `shot_id` 内有效。跨 shot 的 `trk_000` 可以重复，
后续会单独用 identity linking 生成 `global_person_id`。

## Stage06: Hybrid Evidence Timebins

Stage06 读取 Stage05 的 `05_face_tracks.csv`，给每个
`shot_id + local_track_id` 裁一个单人 crop video，然后同时跑三类 evidence：

```text
OpenFace FeatureExtraction  -> face quality / success / confidence / AU / landmarks / diagnostic gaze-pose
L2CS-Net                    -> main gaze evidence
6DRepNet360                 -> main head-pose evidence
```

Stage08/09 仍读取旧核心列名，但这些列现在来自 hybrid backend：

```text
gaze_angle_x_mean = L2CS pitch mean   # horizontal gaze evidence
gaze_angle_y_mean = L2CS yaw mean     # vertical gaze evidence
pose_Ry_mean      = 6DRepNet yaw mean
pose_Rx_mean      = 6DRepNet pitch mean
pose_Rz_mean      = 6DRepNet roll mean
confidence_mean   = OpenFace confidence mean
valid_ratio       = OpenFace success/quality ratio
```

OpenFace 原始 gaze/pose 仍保留在 diagnostic columns：

```text
openface_gaze_angle_x_mean
openface_gaze_angle_y_mean
openface_pose_Rx_mean
openface_pose_Ry_mean
openface_pose_Rz_mean
openface_success_ratio
openface_confidence_mean
```

需要本地模型：

```text
models/gaze_estimation/L2CS-Net/models/L2CSNet_gaze360.pkl
models/head_pose/6DRepNet360/checkpoints/6DRepNet360_Full-Rotation_300W_LP_Panoptic.pth
```

smoke test：

```bash
PYTHON_BIN=/home/sdu/Desktop/ExpreGaze/.venv/bin/python \
  bash scripts/stages/06_run_openface_per_face.sh configs/runs/video_proxy_tt0032138.yaml --max-tracks 3 --device cuda:0 --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/face_crops/{sequence_id}/{shot_id}__{local_track_id}.mp4
outputs/video_proxy/tt0032138/openface/raw/{sequence_id}/{shot_id}__{local_track_id}/
outputs/video_proxy/tt0032138/openface/06_track_manifest.csv
outputs/video_proxy/tt0032138/openface/06_openface_raw_index.csv
outputs/video_proxy/tt0032138/openface/06_gaze_timebins.csv
outputs/video_proxy/tt0032138/logs/06_run_openface_per_track_summary.json
```

检查：

```bash
less -S outputs/video_proxy/tt0032138/openface/06_track_manifest.csv
less -S outputs/video_proxy/tt0032138/openface/06_openface_raw_index.csv
less -S outputs/video_proxy/tt0032138/openface/06_gaze_timebins.csv
cat outputs/video_proxy/tt0032138/logs/06_run_openface_per_track_summary.json
```

`06_gaze_timebins.csv` 里的时间是 shot-local seconds。`gaze_quality` 当前包括：

```text
gaze_reliable
pose_fallback
unknown
crop_multi_face_contaminated
```

同一个表里也会包含 AU/expression evidence，例如：

```text
AU12_r_mean
AU12_c_ratio
AU04_r_mean
AU04_c_ratio
AU25_c_ratio
AU26_c_ratio
AU45_c_ratio
expression_proxy
```

`expression_proxy` 是粗粒度 debug 标签，暂时不参与 Stage08 target assignment。

## Stage07: Track Identity

Stage07 读取 MovieNet cast annotation/meta、Stage05 face tracks、Stage05 detections
和 Stage04 manifest，把 shot-local `local_track_id` 链接到
`global_person_id/cast_pid`。默认使用 InsightFace visual gallery 做主证据，
MovieNet body bbox 和单说话人单 track 只作为弱回退。它不读取 proxy assignment。

默认配置需要安装 `insightface` 和 `onnxruntime-gpu`，模型 root 放在项目内：

```text
models/face_recognition/insightface
```

InsightFace 会在 root 下查找模型包，所以 `buffalo_l` 的 ONNX 文件应位于：

```text
models/face_recognition/insightface/models/buffalo_l/
```

如果机器没有 CUDA provider，可以在 run config 里把 `insightface_provider` 改成
`CPUExecutionProvider`。如需临时跳过视觉 gallery，可运行：

```bash
bash scripts/stages/07_build_track_identities.sh configs/runs/video_proxy_tt0032138.yaml --identity-backend none --overwrite
```

运行：

```bash
bash scripts/stages/07_build_track_identities.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/track_identities/07_track_identity.csv
outputs/video_proxy/tt0032138/track_identities/07_identity_gallery.csv
outputs/video_proxy/tt0032138/track_identities/07_identity_gallery.pkl
outputs/video_proxy/tt0032138/logs/07_track_identity_summary.json
```

检查：

```bash
less -S outputs/video_proxy/tt0032138/track_identities/07_track_identity.csv
less -S outputs/video_proxy/tt0032138/track_identities/07_identity_gallery.csv
cat outputs/video_proxy/tt0032138/logs/07_track_identity_summary.json
```

`identity_source=insightface_gallery` 表示 track 通过 InsightFace embedding
匹配到自动单人 shot gallery；`movienet_body_bbox` 表示 face track 通过
MovieNet body bbox 匹配到 cast pid；`sface_gallery` 是可选的 OpenCV SFace
后端；`single_speaker_single_track` 是单说话人、单 track 的弱回退。

建议抽检：

```bash
rg 'insightface_gallery|movienet_body_bbox|single_speaker_single_track' outputs/video_proxy/tt0032138/track_identities/07_track_identity.csv
less -S outputs/video_proxy/tt0032138/track_identities/07_identity_gallery.csv
cat outputs/video_proxy/tt0032138/logs/07_track_identity_summary.json
```

重点看 `visual_score`、`visual_margin`、`prototype_id` 和
`weak_fallback_source`。如果 `unknown` 异常少，通常说明 visual 阈值过松。

生成可在浏览器打开的 identity spot-check HTML：

```bash
PYTHONPATH=src .venv/bin/python scripts/render_identity_report.py \
  --run-config configs/runs/video_proxy_tt0032138.yaml \
  --max-samples-per-source 24
```

HTML report 中的 `role` 来自 movie config 的 `identity_display_aliases`
人工显示别名；`MovieNet character` 保留原始 MovieNet credited role，方便区分
`Wizard` vs `Professor Marvel` 这类同演员双角色情况。

输出位置：

```text
outputs/video_proxy/tt0032138/track_identities/debug_identity_report/identity_report.html
outputs/video_proxy/tt0032138/track_identities/debug_identity_report/identity_report_summary.json
outputs/video_proxy/tt0032138/track_identities/debug_identity_report/images/
```

## Stage08: Candidate Targets + Proxy Assignment

Stage08 读取 Stage06 timebins、Stage05 tracks、Stage07 track identity、
Stage04 manifest 和 Stage02 selected sequence JSONL，生成 candidate targets
和高精度 rule-based proxy assignment。输出表已经包含 subject/target identity
字段，不再需要单独 post-link。

运行：

```bash
bash scripts/stages/08_build_proxy_gaze_script.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_candidate_targets.csv
outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_proxy_assignments.csv
outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_proxy_assignments.jsonl
outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_proxy_sequence_packages.jsonl
outputs/video_proxy/tt0032138/logs/08_build_proxy_gaze_script_summary.json
```

检查：

```bash
less -S outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_candidate_targets.csv
less -S outputs/video_proxy/tt0032138/proxy_gaze_scripts/08_proxy_assignments.csv
cat outputs/video_proxy/tt0032138/logs/08_build_proxy_gaze_script_summary.json
```

第一版偏 high precision，所以 `unknown` 或 `ambiguous` 多是正常的。

Stage08 是 bin-level：每个 0.5s bin 独立做一次 candidate scoring。它会做一点
isolated-bin smoothing，但不负责长时序 event segmentation。

## Stage08b: Event-Level Temporal Smoothing

Stage08b 读取 Stage08 的 bin-level candidates/assignments，再生成 shot context，
把连续 bins 合并成 gaze events，并在 event 粒度重新选择 target。它不直接改
Stage08 的 scoring 代码。

运行：

```bash
bash scripts/stages/08b_build_gaze_events.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/gaze_events/08b_shot_context.csv
outputs/video_proxy/tt0032138/gaze_events/08b_gaze_events.csv
outputs/video_proxy/tt0032138/gaze_events/08b_gaze_event_bins.csv
outputs/video_proxy/tt0032138/logs/08b_build_gaze_events_summary.json
```

检查：

```bash
less -S outputs/video_proxy/tt0032138/gaze_events/08b_shot_context.csv
less -S outputs/video_proxy/tt0032138/gaze_events/08b_gaze_events.csv
less -S outputs/video_proxy/tt0032138/gaze_events/08b_gaze_event_bins.csv
cat outputs/video_proxy/tt0032138/logs/08b_build_gaze_events_summary.json
```

核心逻辑：

```text
1. 同一个 movie/sequence/shot/subject_local_track_id 内分 event，不跨 shot 合并。
2. 连续 bins 的 gaze/pose direction 相近时合并。
3. 强 left/right 切换会切成不同 event。
4. 中间单个 unknown/ambiguous，若前后 target 一致，会 bridge 到同一 event。
5. 少于 1.0s 的孤立 target switch 视为 flicker，合并回前后稳定 event。
6. event-level target 由 event 内 candidate scores 聚合，再加 shot context prior。
```

`08b_shot_context.csv` 一 shot 一行，重点字段：

```text
visible_global_persons
current_speaker
active_participants
is_single_closeup
is_two_person_shot
is_reaction_shot
prev_visible_focus
next_visible_focus
likely_interlocutor
```

## Stage09: Final Proxy Table

Stage09 把 Stage06 timebins、Stage05 tracks、Stage07 identity、Stage08 candidates /
assignments 合成最终 bin-level table。如果存在 Stage08b 输出，Stage09 会用
event-level target 覆盖 Stage08 的 bin-level target；如果不存在 Stage08b 输出，
则自动回退旧 Stage08 assignment。

运行：

```bash
bash scripts/stages/09_build_final_proxy_table.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

主输出：

```text
outputs/video_proxy/tt0032138/final_proxy/09_final_proxy_table.csv
outputs/video_proxy/tt0032138/final_proxy/09_final_proxy_table.jsonl
outputs/video_proxy/tt0032138/logs/09_build_final_proxy_table_summary.json
```

检查：

```bash
less -S outputs/video_proxy/tt0032138/final_proxy/09_final_proxy_table.csv
cat outputs/video_proxy/tt0032138/logs/09_build_final_proxy_table_summary.json
```

Stage08b 存在时，final table 里会新增并填充：

```text
event_id
event_start
event_end
event_status
event_confidence
event_target_type
event_target_id
event_target_global_person_id
```

最终用于 debug / audit / SFT 的核心字段：

```text
subject_local_track_id
subject_global_person_id
gaze_quality
gaze_direction_bucket
pose_direction_bucket
candidate_list
target_type
target_id
target_global_person_id
proxy_confidence
proxy_status
failure_reason
event_id
```

## 一键 Video Proxy Pipeline

完整 video proxy 主线：

```bash
PYTHON_BIN=/home/sdu/Desktop/ExpreGaze/.venv/bin/python \
  bash scripts/pipelines/run_video_proxy.sh configs/runs/video_proxy_tt0032138.yaml --overwrite --device cuda:0
```

只 smoke test 少量 tracks：

```bash
PYTHON_BIN=/home/sdu/Desktop/ExpreGaze/.venv/bin/python \
  bash scripts/stages/06_run_openface_per_face.sh configs/runs/video_proxy_tt0032138.yaml --max-tracks 3 --device cuda:0 --overwrite

bash scripts/stages/07_build_track_identities.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/08_build_proxy_gaze_script.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/08b_build_gaze_events.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/09_build_final_proxy_table.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

如果 Stage04-06 已经是最新，只想重跑 identity/proxy/final：

```bash
bash scripts/stages/07_build_track_identities.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/08_build_proxy_gaze_script.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/08b_build_gaze_events.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
bash scripts/stages/09_build_final_proxy_table.sh configs/runs/video_proxy_tt0032138.yaml --overwrite
```

三部电影重跑 Stage07-09：

```bash
for cfg in \
  configs/runs/video_proxy_tt0032138.yaml \
  configs/runs/video_proxy_tt1591095.yaml \
  configs/runs/video_proxy_tt1637725.yaml
do
  bash scripts/stages/07_build_track_identities.sh "$cfg" --overwrite
  bash scripts/stages/08_build_proxy_gaze_script.sh "$cfg" --overwrite
  bash scripts/stages/08b_build_gaze_events.sh "$cfg" --overwrite
  bash scripts/stages/09_build_final_proxy_table.sh "$cfg" --overwrite
done
```

## 一键 Text Pipeline

这个 wrapper 会先跑 Stage00，再跑 Stage01。默认停在 Stage01 `raw`，所以不会误调用 OpenAI：

```bash
bash scripts/pipelines/run_text_main.sh configs/runs/text_main_tt1591095.yaml
```

如果想继续生成纯 DP `full_context`：

```bash
bash scripts/pipelines/run_text_main.sh configs/runs/text_main_tt1591095.yaml final --overwrite
```

如果想跑小样本 LLM 修复：

```bash
bash scripts/pipelines/run_text_main.sh configs/runs/text_main_tt1591095.yaml llm --max-llm-rows 3 --overwrite
```

## 其他电影

换 run config 即可：

```bash
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt0032138.yaml raw --overwrite
bash scripts/stages/01_align_full_context_with_gpt.sh configs/runs/text_main_tt1637725.yaml raw --overwrite
```

输出目录会跟着 movie id 变，例如：

```text
outputs/text_main/tt0032138/logs/
outputs/text_main/tt1637725/logs/
```
