import bpy

ARM = "RIG-rain"
BONES = [
    "ACT-Eyelid_Upper.L", "ACT-Eyelid_Lower.L",
    "ACT-Eyelid_Upper.R", "ACT-Eyelid_Lower.R",
]

arm = bpy.data.objects[ARM]
for b in BONES:
    pb = arm.pose.bones.get(b)
    if pb is None:
        print("Missing bone:", b)
        continue
    print(b, "open_Y =", pb.location[1])