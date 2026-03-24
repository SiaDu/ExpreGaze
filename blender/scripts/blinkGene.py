import bpy, json, random

# ====== 配置 ======
ARM = "RIG-rain"
EVENTS_PATH = "/home/sdu/Desktop/ExpreGaze/blender_Char_Assets/EarlyP_GazeEvents_001.json"

FPS = bpy.context.scene.render.fps

# 眨眼形状（帧）
CLOSE_IN = 2     # 几帧闭上
HOLD = 1         # 闭眼停留几帧
OPEN_OUT = 2     # 几帧睁开

# 概率/频率
P_SHIFT = 0.35   # gaze shift 触发眨眼概率
NAT_MIN = 3.0    # 自然眨眼最短间隔（秒）
NAT_MAX = 6.0    # 自然眨眼最长间隔（秒）

# ====== 你需要填的校准值（把下面数字替换成你打印出来的）======
OPEN_Y = {
    "ACT-Eyelid_Upper.L": 0.0,
    "ACT-Eyelid_Lower.L": 0.0,
    "ACT-Eyelid_Upper.R": 0.0,
    "ACT-Eyelid_Lower.R": 0.0,
}
CLOSE_Y = {
    "ACT-Eyelid_Upper.L": -0.023356152698397636,
    "ACT-Eyelid_Lower.L": 0.004560943692922592,
    "ACT-Eyelid_Upper.R": -0.023356152698397636,
    "ACT-Eyelid_Lower.R": 0.004560943692922592,
}

BONES = list(OPEN_Y.keys())

# ====== 工具函数 ======
def sec_to_frame(t):
    return int(round(float(t) * FPS))

def set_frame(f):
    bpy.context.scene.frame_set(int(f))

def key_y(pb, y):
    pb.location[1] = y
    pb.keyframe_insert(data_path="location", index=1)

def insert_blink_at_frame(arm, f0):
    """在 f0 开始眨眼：open -> close -> hold -> open"""
    f_open = f0
    f_close = f0 + CLOSE_IN
    f_hold = f_close + HOLD
    f_back = f_hold + OPEN_OUT

    for b in BONES:
        pb = arm.pose.bones.get(b)
        if pb is None:
            continue

        # open
        set_frame(f_open)
        key_y(pb, OPEN_Y[b])

        # close
        set_frame(f_close)
        key_y(pb, CLOSE_Y[b])

        # hold
        set_frame(f_hold)
        key_y(pb, CLOSE_Y[b])

        # open back
        set_frame(f_back)
        key_y(pb, OPEN_Y[b])

# ====== 读取 gaze events，生成“切换点” ======
with open(EVENTS_PATH, "r", encoding="utf-8") as f:
    events = json.load(f)

# 事件按时间排序
events = sorted(events, key=lambda e: float(e["t0"]))

shift_frames = []
for i in range(len(events) - 1):
    # 在每段结束附近插（用 t1 当边界）
    f_boundary = sec_to_frame(events[i]["t1"])
    shift_frames.append(f_boundary)

# ====== 生成自然眨眼时间 ======
if events:
    start_f = sec_to_frame(events[0]["t0"])
    end_f = sec_to_frame(events[-1]["t1"])
else:
    start_f, end_f = 1, 250

natural_frames = []
cur = start_f + int(NAT_MIN * FPS)
while cur < end_f:
    step = random.uniform(NAT_MIN, NAT_MAX)
    cur += int(step * FPS)
    if cur < end_f:
        natural_frames.append(cur)

# ====== 执行插入 ======
arm = bpy.data.objects[ARM]

# 1) shift-triggered blinks
for f0 in shift_frames:
    if random.random() < P_SHIFT:
        insert_blink_at_frame(arm, f0)

# 2) natural blinks
for f0 in natural_frames:
    insert_blink_at_frame(arm, f0)

print("OK: inserted blinks. shift:", len(shift_frames), "natural:", len(natural_frames))