import bpy, json, random
from mathutils import Vector

# ====== 你只需要改这里 ======
CONFIG_PATH = "/home/sdu/Desktop/ExpreGaze/blender_Char_Assets/EarlyP_GazeEvents_002.json"
# ===========================

FPS = bpy.context.scene.render.fps

def sec_to_frame(t):
    return int(round(float(t) * FPS))

def set_frame(f):
    bpy.context.scene.frame_set(int(f))

def ensure_object(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"找不到对象: {name}")
    return obj

def ensure_armature(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "ARMATURE":
        raise RuntimeError(f"找不到骨架(Armature): {name}")
    return obj

def ensure_pose_bone(arm, bone_name: str):
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        raise RuntimeError(f"找不到骨骼: {bone_name} (in {arm.name})")
    return pb

def add_copy_location_to_bone(arm, bone_name: str, target_obj_name: str):
    """给某根pose bone加 Copy Location -> target_obj（只加在骨骼上，不会拖动整个rig）"""
    tgt = ensure_object(target_obj_name)
    pb = ensure_pose_bone(arm, bone_name)

    # 避免重复叠加同名约束：删掉旧的 COPY_LOCATION（可按需更严格筛选）
    for c in list(pb.constraints):
        if c.type == "COPY_LOCATION":
            pb.constraints.remove(c)

    con = pb.constraints.new("COPY_LOCATION")
    con.target = tgt
    con.owner_space = "WORLD"
    con.target_space = "WORLD"
    con.influence = 1.0

def clear_object_location_keys(obj):
    # 最稳：清掉整套动画数据（避免 Blender 5 API 变化坑）
    if obj.animation_data:
        obj.animation_data_clear()
    obj.animation_data_create()

def set_key_interp_for_object(obj, data_path="location", interp="LINEAR"):
    """尽量把关键帧插值设成线性/常量（如果你不关心插值可忽略）"""
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    act = ad.action
    # Blender 新旧 API 可能变化，这里用 try 做兼容
    try:
        fcurves = act.fcurves
    except Exception:
        return
    for fc in fcurves:
        if fc.data_path == data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = interp
                # VECTOR 让转向更“干脆”，避免ease
                if interp == "LINEAR":
                    kp.handle_left_type = "VECTOR"
                    kp.handle_right_type = "VECTOR"

def insert_blink_at_frame(arm, eyelid_bones, open_y, close_y, f0, close_in, hold, open_out):
    """在 f0 开始眨眼：open -> close -> hold -> open (控制 location Y)"""
    f_open  = f0
    f_close = f0 + int(close_in)
    f_hold  = f_close + int(hold)
    f_back  = f_hold + int(open_out)

    for b in eyelid_bones:
        pb = arm.pose.bones.get(b)
        if pb is None:
            continue

        def key_y(frame, y):
            set_frame(frame)
            pb.location[1] = float(y)
            pb.keyframe_insert(data_path="location", index=1)

        key_y(f_open,  open_y[b])
        key_y(f_close, close_y[b])
        key_y(f_hold,  close_y[b])
        key_y(f_back,  open_y[b])

def insert_inv_triangle_saccade(aim, tgt, f_start, f_end, dx=0.015, dz=0.01, hold_frames=6):
    """
    在 [f_start, f_end] 内，让 aim 在 tgt 附近做倒三角扫视：
    顶点 -> 左下 -> 右下 循环；落点之间 1 帧跳转。
    """
    # 三个落点（tgt 的局部坐标系里），Y=0 表示不往前后偏
    offsets = [
        Vector((0.0, 0.0, +dz)),   # 上
        Vector((-dx, 0.0, -dz)),   # 左下
        Vector((+dx, 0.0, -dz)),   # 右下
    ]

    # 把局部偏移转成世界坐标（这样 tgt 自己跟着头动时，扫视范围也跟着头）
    R = tgt.matrix_world.to_3x3()
    T = tgt.matrix_world.translation

    def world_pos(off_local):
        return T + R @ off_local

    cur = int(f_start)
    hold_frames = max(int(hold_frames), 2)  # 至少2帧，才能做到“最后一帧保持 + 下一帧跳转”

    i = 0
    while cur < f_end:
        p = world_pos(offsets[i % 3])

        # 关键帧：cur 到 cur+hold-1 保持同一落点
        bpy.context.scene.frame_set(cur)
        aim.location = p
        aim.keyframe_insert(data_path="location")

        last_hold = min(cur + hold_frames - 1, f_end)
        bpy.context.scene.frame_set(last_hold)
        aim.location = p
        aim.keyframe_insert(data_path="location")

        cur = last_hold + 1
        i += 1

    # 结束时回到中心（更自然，也方便接下一个 target）
    bpy.context.scene.frame_set(int(f_end))
    aim.location = T
    aim.keyframe_insert(data_path="location")

# ========= 读取配置 =========
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

rig_cfg   = cfg.get("rig", {})
gaze_cfg  = cfg.get("gaze", {})
blink_cfg = cfg.get("blink", {})
events    = cfg.get("events", [])
# 读取 saccade 配置（在读 cfg 后加这几行）
sacc_cfg = cfg.get("saccade", {})
SACC_ON = bool(sacc_cfg.get("enabled", False))
SACC_TARGETS = set(sacc_cfg.get("targets", []))
SACC_PATTERN = sacc_cfg.get("pattern", "inv_triangle")
SACC_DX = float(sacc_cfg.get("dx", 0.015))
SACC_DZ = float(sacc_cfg.get("dz", 0.010))
SACC_HOLD = int(sacc_cfg.get("hold_frames", 6))

ARM_NAME = rig_cfg.get("armature", "RIG-rain")
EYE_BONE = rig_cfg.get("eye_target_bone", "TGT-Eyes")
AUTO_EYE = bool(rig_cfg.get("auto_add_eye_constraint", False))

AIM_NAME = gaze_cfg.get("aim_object", "Gaze_Aim")
TRANS_FRAMES = int(gaze_cfg.get("trans_frames", 2))
INTERP = gaze_cfg.get("interpolation", "LINEAR").upper()

BLINK_ON = bool(blink_cfg.get("enabled", True))
CLOSE_IN = int(blink_cfg.get("close_in", 2))
HOLD = int(blink_cfg.get("hold", 1))
OPEN_OUT = int(blink_cfg.get("open_out", 2))
P_SHIFT = float(blink_cfg.get("p_shift", 0.35))
NAT_MIN = float(blink_cfg.get("nat_min", 3.0))
NAT_MAX = float(blink_cfg.get("nat_max", 6.0))
SEED = int(blink_cfg.get("seed", 123))

eyelid_bones = rig_cfg.get("eyelid_bones", [])
open_y = blink_cfg.get("open_y", {})
close_y = blink_cfg.get("close_y", {})

# ========= 基础检查 =========
aim = ensure_object(AIM_NAME)
arm = ensure_armature(ARM_NAME)

# 若你希望脚本自动把“眼睛target bone”绑定到 Gaze_Aim，就开 AUTO_EYE
if AUTO_EYE:
    add_copy_location_to_bone(arm, EYE_BONE, AIM_NAME)

# ========= 生成 gaze：给 Gaze_Aim 插关键帧 =========
clear_object_location_keys(aim)

# events 转帧并排序
ev = []
for e in events:
    ev.append({"f0": sec_to_frame(e["t0"]), "f1": sec_to_frame(e["t1"]), "target": e["target"]})
ev.sort(key=lambda x: x["f0"])

# 解决同帧覆盖：保证下一段 start >= 上一段 end + TRANS_FRAMES
cur_min_start = ev[0]["f0"] if ev else 1
for i in range(len(ev)):
    ev[i]["f0"] = max(ev[i]["f0"], cur_min_start)
    ev[i]["f1"] = max(ev[i]["f1"], ev[i]["f0"] + 1)
    cur_min_start = ev[i]["f1"] + TRANS_FRAMES

# 设置“新建关键帧插值”
prefs = bpy.context.preferences.edit
old_interp = prefs.keyframe_new_interpolation_type
prefs.keyframe_new_interpolation_type = "LINEAR" if INTERP != "CONSTANT" else "CONSTANT"

for i, e in enumerate(ev):
    tgt = ensure_object(e["target"])
    f0, f1 = e["f0"], e["f1"]

    if SACC_ON and (e["target"] in SACC_TARGETS) and (SACC_PATTERN == "inv_triangle"):
        insert_inv_triangle_saccade(
            aim, tgt,
            f_start=f0,
            f_end=f1,
            dx=SACC_DX, dz=SACC_DZ,
            hold_frames=SACC_HOLD
        )
    else:
        # 到达目标
        set_frame(f0)
        aim.location = tgt.matrix_world.translation
        aim.keyframe_insert(data_path="location")

        # 保持
        set_frame(f1)
        aim.location = tgt.matrix_world.translation
        aim.keyframe_insert(data_path="location")

        # 过渡：在 f1 + TRANS_FRAMES 抵达下一目标（实现1~2帧切换）
        if i < len(ev) - 1:
            nxt = ensure_object(ev[i + 1]["target"])
            set_frame(f1 + TRANS_FRAMES)
            aim.location = nxt.matrix_world.translation
            aim.keyframe_insert(data_path="location")

prefs.keyframe_new_interpolation_type = old_interp

# 尽量设插值（如果 Blender API 允许的话）
if INTERP in ("LINEAR", "CONSTANT"):
    set_key_interp_for_object(aim, "location", INTERP)

# ========= 生成 blink =========
if BLINK_ON:
    random.seed(SEED)

    # 检查 eyelid 数据完整性
    for b in eyelid_bones:
        if b not in open_y or b not in close_y:
            raise RuntimeError(f"blink 的 open_y/close_y 缺少骨骼: {b}")

    # shift 边界：用每段的 f1 当作“切换点”
    shift_frames = [e["f1"] for e in ev[:-1]]

    # 自然眨眼时间
    if ev:
        start_f = ev[0]["f0"]
        end_f = ev[-1]["f1"]
    else:
        start_f, end_f = 1, 250

    natural_frames = []
    cur = start_f + int(NAT_MIN * FPS)
    while cur < end_f:
        step = random.uniform(NAT_MIN, NAT_MAX)
        cur += int(step * FPS)
        if cur < end_f:
            natural_frames.append(cur)

    # 插入眨眼：shift-triggered
    for f0 in shift_frames:
        if random.random() < P_SHIFT:
            insert_blink_at_frame(arm, eyelid_bones, open_y, close_y, f0, CLOSE_IN, HOLD, OPEN_OUT)

    # 插入眨眼：natural
    for f0 in natural_frames:
        insert_blink_at_frame(arm, eyelid_bones, open_y, close_y, f0, CLOSE_IN, HOLD, OPEN_OUT)

print("DONE: gaze + blink generated.")