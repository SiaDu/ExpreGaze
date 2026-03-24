import bpy

arm = bpy.data.objects["RIG-rain"]
pb = arm.pose.bones["TGT-Eyes"]

# 清掉已有的同类型约束（可选）
for c in list(pb.constraints):
    if c.type == 'COPY_LOCATION':
        pb.constraints.remove(c)

con = pb.constraints.new('COPY_LOCATION')
con.target = bpy.data.objects["Gaze_Aim"]
con.owner_space = 'WORLD'
con.target_space = 'WORLD'
con.influence = 1.0

print("OK: Copy Location added to bone TGT-Eyes")