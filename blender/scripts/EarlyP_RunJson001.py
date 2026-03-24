import bpy, json

FPS = bpy.context.scene.render.fps
EVENTS_PATH = "/home/sdu/Desktop/ExpreGaze/blender_Char_Assets/EarlyP_GazeEvents_001.json"
GAZE_AIM_NAME = "Gaze_Aim"
TRANS_FRAMES = 2  # 1或2，代表几帧内完成视线切换

aim = bpy.data.objects.get(GAZE_AIM_NAME)
if aim is None:
    raise RuntimeError("找不到 Gaze_Aim：请先 Shift+A 创建 Empty 并命名为 Gaze_Aim")

# ✅ 清掉旧动画（不碰 fcurves，兼容性最好）
if aim.animation_data:
    aim.animation_data_clear()
aim.animation_data_create()

def sec_to_frame(t):
    return int(round(float(t) * FPS))

def set_frame(f):
    bpy.context.scene.frame_set(int(f))

with open(EVENTS_PATH, "r", encoding="utf-8") as f:
    events = json.load(f)

# 转成帧并排序
ev = []
for e in events:
    ev.append({
        "f0": sec_to_frame(e["t0"]),
        "f1": sec_to_frame(e["t1"]),
        "target": e["target"],
    })
ev.sort(key=lambda x: x["f0"])

# ✅ 解决同帧覆盖：让每段至少有1帧，并且下一段 start >= 上一段 end + TRANS_FRAMES
cur_min_start = ev[0]["f0"] if ev else 1
for i in range(len(ev)):
    ev[i]["f0"] = max(ev[i]["f0"], cur_min_start)
    ev[i]["f1"] = max(ev[i]["f1"], ev[i]["f0"] + 1)  # 至少1帧长度
    cur_min_start = ev[i]["f1"] + TRANS_FRAMES

# ✅ 设“新建关键帧默认插值”为 LINEAR（避免慢吞吞 ease）
prefs = bpy.context.preferences.edit
old_interp = prefs.keyframe_new_interpolation_type
prefs.keyframe_new_interpolation_type = 'LINEAR'

for i, e in enumerate(ev):
    tgt = bpy.data.objects.get(e["target"])
    if tgt is None:
        raise RuntimeError(f"找不到 target Empty: {e['target']}（请在场景里创建同名 Empty）")

    f0, f1 = e["f0"], e["f1"]

    # 在 f0 到达目标
    set_frame(f0)
    aim.location = tgt.matrix_world.translation
    aim.keyframe_insert(data_path="location")

    # 在 f1 仍保持目标
    set_frame(f1)
    aim.location = tgt.matrix_world.translation
    aim.keyframe_insert(data_path="location")

    # 如果有下一段：在 f1 + TRANS_FRAMES 抵达下一目标（1~2帧完成切换）
    if i < len(ev) - 1:
        nxt = bpy.data.objects.get(ev[i + 1]["target"])
        if nxt is None:
            raise RuntimeError(f"找不到 next target Empty: {ev[i+1]['target']}")
        set_frame(f1 + TRANS_FRAMES)
        aim.location = nxt.matrix_world.translation
        aim.keyframe_insert(data_path="location")

# 还原偏好
prefs.keyframe_new_interpolation_type = old_interp

print("OK：已生成关键帧，并强制在 1–2 帧内完成 gaze 切换。")