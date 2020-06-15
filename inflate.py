import bpy
import bmesh


obj = bpy.data.objects['jaw']
arm = bpy.data.objects['bone']

bpy.ops.object.mode_set(mode='EDIT')
    
bm = bmesh.from_edit_mesh(obj.data)
num_verts = len(bm.verts)

faces_select = [f for f in bm.faces if f.select] 
bmesh.ops.delete(bm, geom=faces_select, context=3)
bmesh.update_edit_mesh(cone.data, True)

#Extrude faces
bpy.ops.mesh.select_mode( type  = 'FACE'   )
bpy.ops.mesh.select_all( action = 'SELECT' )
bpy.ops.mesh.extrude_region_move(
    TRANSFORM_OT_translate={"value":(0, 0, 0.01)} ) 
bpy.ops.mesh.extrude_region_shrink_fatten(
    TRANSFORM_OT_shrink_fatten={"value":35})
bpy.ops.object.mode_set(mode='OBJECT')

for idx in range(num_verts):
    bm.verts.ensure_lookup_table()
    vert = bm.verts[idx]
    vert.co.z = vert.co.z -depth/2.0
    theta = math.atan2(vert.co.x, vert.co.y)
    
    if abs(vert.co.z)<shape_parameters["stem_length"]:
        rad = shape_parameters["fixture_radius"]
    else:
        rad = morph_shape(abs(vert.co.z)-shape_parameters["stem_length"], 
                          depth-shape_parameters["stem_length"], 
                          radius-shape_parameters["fixture_radius"], 
                          morph_type=shape_parameters["division_pattern"])
        rad += shape_parameters["fixture_radius"]
    vert.co.x = rad*math.sin(theta)
    vert.co.y = rad*math.cos(theta)        

for face in bm.faces:
    if UpOrDown(face.normal):
        face.select = True
    else:
        face.select = False 
faces_select = [f for f in bm.faces if f.select] 

bmesh.ops.delete(bm, geom=faces_select, context=3)
bmesh.update_edit_mesh(cone.data, True)
  