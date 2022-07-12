import bpy
import bmesh

# load the objects from the environment
obj = bpy.data.objects['jaw']
arm = bpy.data.objects['bone']

# select the object and turn the EDIT mode on
bpy.ops.object.mode_set(mode='EDIT')
   
# returns a BMesh from selected mesh
bm = bmesh.from_edit_mesh(obj.data)
num_verts = len(bm.verts) # get vertices number

# get the faces, delete the duplicated and update the structure
faces_select = [f for f in bm.faces if f.select] 
bmesh.ops.delete(bm, geom=faces_select, context=3)
bmesh.update_edit_mesh(cone.data, True)

# extrude faces, armature operators are used.
bpy.ops.mesh.select_mode( type  = 'FACE'   )
bpy.ops.mesh.select_all( action = 'SELECT' )
bpy.ops.mesh.extrude_region_move(
    TRANSFORM_OT_translate={"value":(0, 0, 0.01)} )  # translate, move selected items
bpy.ops.mesh.extrude_region_shrink_fatten(
    TRANSFORM_OT_shrink_fatten={"value":35}) # extrude each individual face separately along local normals with a selected step
bpy.ops.object.mode_set(mode='OBJECT')


# iterate over vertices
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
    
    
# iterate over faces, move using local normals
for face in bm.faces:
    if UpOrDown(face.normal):
        face.select = True
    else:
        face.select = False 
faces_select = [f for f in bm.faces if f.select] 


# remove and update the mesh
bmesh.ops.delete(bm, geom=faces_select, context=3)
bmesh.update_edit_mesh(cone.data, True)
  
