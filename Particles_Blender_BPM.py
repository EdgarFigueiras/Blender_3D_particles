import bpy
import cArray
import time
import math
import random
import struct
import binascii
import os.path
import numpy as np
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup,
                       )

# ------------------------------------------------------------------------
#    Panel which allows the user to interact with the simulator
# ------------------------------------------------------------------------

#Clean the scene
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

image_format = (
    ('BMP', 'BMP', ''),
    ('IRIS', 'IRIS', ''),
    ('PNG', 'PNG', ''),
    ('JPEG', 'JPEG', ''),
    ('JPEG2000', 'JPEG2000', ''),
    ('TARGA', 'TARGA', ''),
    ('TARGA_RAW', 'TARGA_RAW', ''), 
    ('CINEON', 'CINEON', ''),
    ('DPX', 'DPX', ''),
    ('OPEN_EXR_MULTILAYER', 'OPEN_EXR_MULTILAYER', ''),
    ('OPEN_EXR', 'OPEN_EXR', ''), 
    ('HDR', 'HDR', ''),
    ('TIFF', 'TIFF', '')
)

video_format = (
    ('AVI_JPEG', 'AVI_JPEG', ''),
    ('AVI_RAW', 'AVI_RAW', ''), 
    ('FRAMESERVER', 'FRAMESERVER', ''), 
    ('H264', 'H264', ''),
    ('FFMPEG', 'FFMPEG', ''),
    ('THEORA', 'THEORA', '')
)

calculation_format = (
    ('2D', '2D', ''),
    ('3D', '3D', '')
)

planes_number = (
    ('1P', '1P', ''),
    ('2P', '2P', '')
)

enum_items = (
    ('FOO', 'Foo', ''),
    ('BAR', 'Bar', '')
)

class MySettings(PropertyGroup):

    folder_path = StringProperty(
        name="Data Folder",
        description="Select the folder with the simulation data.",
        default="",
        maxlen=1024,
        subtype='FILE_PATH')

    path = StringProperty(
        name="Data File",
        description="Select the file with the simulation data.",
        default="",
        maxlen=1024,
        subtype='FILE_PATH')

    image_path = StringProperty(
        name="Store Path",
        description="Path where renders will be stored, by default uses the path of the simulation data",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')

    total_states_info = IntProperty(
        name="Min :0  Max ", 
        description="Total number of states of the simulation",
        min = 0, max = 1000000,
        default = 0)

    int_box_particulas_Simulacion = IntProperty(
        name="Simulation particles", 
        description="Total number of particles for generating the matrix",
        min = 1000, max = 10000000,
        default = 100000)

    int_box_n_particulas = IntProperty(
        name="Particles to show ", 
        description="Total number of particles of the simulation",
        min = 1000, max = 10000000,
        default = 100000)

    int_box_granularity = IntProperty(
        name="Granularity ", 
        description="Modifies the granularity. Min = 1 , Max = 10",
        min = 1, max = 10,
        default = 5)

    int_box_saturation = IntProperty(
        name="Saturation ", 
        description="Modify the saturation. Min = 1, Max = 10",
        min = 1, max = 10,
        default = 5)

    int_box_state = IntProperty(
        name="State ", 
        description="Modify the State",
        min = 0, max = 9999,
        default = 0)

    bool_cut_box = BoolProperty(
        name="Cut box side",
        description="Enables the oposite cut plane view",
        default = True)      

#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Panel class                                                          #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class OBJECT_OT_AddColors(bpy.types.Operator):
    bl_idname = "add.colors"
    bl_label = "Add colors"
    country = bpy.props.StringProperty()

    def execute(self, context):

        #TODO Set the number of particles for each step reading the data from the original file, 
        #remember to set 10 steps using the 10% of the biggest probability value
        #also you will have to use the dupli_weights list of values to iterate over it when the key changes because te 
        #number of particles
        #Define an error message if occurs a problem during the run, is showed using a popup 
        def error_message(self, context):
            self.layout.label("Error opening the original array data")

        try:
            path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
       
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas

        #Use an auxiliar array to work with a variable number of points, 
        #allowing the user to make diferent points simulation with good results
        array_aux = np.zeros((particles_number, 4))
        #Fill the auxiliar array with the data of the original one 
        for point_number in range (0, particles_number):
            array_aux[point_number] = array_3d[0][point_number]

        #Sort the array to place first the particles with more probability
        array_aux = array_aux[np.argsort(array_aux[:,3])]


        #With the array sorted the last position is the particle with the biggest probability to appear
        actual_max_prob_value = array_aux[particles_number-1][3]
        #With the array sorted the first position is the particle with less probability to appear
        actual_min_prob_value = array_aux[0][3]

        general_maximum = actual_max_prob_value
        general_minimum = actual_min_prob_value

        for state_array in array_3d:
            maxi = np.max(state_array[:,3])
            mini = np.max(state_array[:,3])

            if (maxi > general_maximum):
                general_maximum = maxi
            if (mini < general_minimum):
                general_minimum = mini


        #Obtain an scalated step to distribute the points between the 10 scales of probability
        step_prob = (general_maximum-general_minimum)/10
        prob_using = step_prob

        steps = np.zeros(11)
        actual_step = 0

        for cont_particle in range(particles_number):
            if(array_aux[cont_particle][3] <= prob_using):
                steps[actual_step] += 1
            else:
                actual_step += 1
                prob_using += step_prob 

        #solves the problem with the extra particles not asigned to the last position
        steps[9] += steps[10]

        bpy.data.objects['Sphere'].select = True
        bpy.context.scene.objects.active = bpy.data.objects['Sphere']
        for cont_mat in range(10):
            material_value = bpy.data.objects['Sphere'].particle_systems['Drops'].settings.dupli_weights.get("Ico_"+str(9-cont_mat)+": 1")
            material_value.count = steps[cont_mat]

        return{'FINISHED'} 

class OBJECT_OT_ResetButton(bpy.types.Operator):
    bl_idname = "reset.image"
    bl_label = "Reiniciar entorno"
    country = bpy.props.StringProperty()

    def execute(self, context):

        def confirm_message(self, context):
            self.layout.label("The system environment was cleaned")

        nombreObjeto = "Sphere"  

        bpy.data.objects[nombreObjeto].hide = False

        bpy.context.space_data.viewport_shade = 'MATERIAL'
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        bpy.context.scene.frame_current = 0

        bpy.data.scenes["Scene"].my_tool.int_box_state = -1
        bpy.context.window_manager.popup_menu(confirm_message, title="Reset", icon='VIEW3D_VEC')

        return{'FINISHED'} 



class OBJECT_OT_RenderButton(bpy.types.Operator):
    bl_idname = "render.image"
    bl_label = "RenderizarImagen"
    country = bpy.props.StringProperty()


    #This code 
    def execute(self, context):

        dir_image_path = bpy.data.scenes['Scene'].my_tool.image_path

        #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Unable to save the Render. Try again with other path")


        try:    
            #Set the image format, PNG by default
            bpy.context.scene.render.image_settings.file_format = bpy.context.scene['ImageFormat']

        except:        
            bpy.context.scene.render.image_settings.file_format = 'PNG'

        try:

            #Sets the path where the file will be stored, by default the same as the datafile
            if dir_image_path == "":
                bpy.data.scenes['Scene'].render.filepath = bpy.data.scenes['Scene'].my_tool.path + time.strftime("%c%s") + '.jpg'
                
                #Define a confirmation message to the default path            
                def confirm_message(self, context):
                    self.layout.label("Render image saved at: " + bpy.data.scenes['Scene'].my_tool.path )

            else:                
                bpy.data.scenes['Scene'].render.filepath = dir_image_path + time.strftime("%c%s") + '.jpg'
               
                #Define a confirmation message to the selected path 
                def confirm_message(self, context):
                    self.layout.label("Rendered image saved at: " + dir_image_path )   

            bpy.ops.render.render( write_still=True ) 


            bpy.context.window_manager.popup_menu(confirm_message, title="Saved successful", icon='SCENE')

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        return{'FINISHED'} 


#Renders all objects one by one jumping between states
class OBJECT_OT_RenderAllButton(bpy.types.Operator):
    bl_idname = "render_all.image"
    bl_label = "RenderizarAllImagen"
    country = bpy.props.StringProperty()


    #This code 
    def execute(self, context):

        dir_image_path = bpy.data.scenes['Scene'].my_tool.image_path

        #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Unable to save the Renders. Try again with other path")

        #Calculate the total of states
        #Calculate the total of states
        try:
            path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

            total_states = len(array_3d)

            file_with_binary_data.close()

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')


        for x in range(int(total_states)):

            try:    
                #Set the image format, PNG by default
                bpy.context.scene.render.image_settings.file_format = bpy.context.scene['ImageFormat']

            except:        
                bpy.context.scene.render.image_settings.file_format = 'PNG'

            try:

                #Sets the path where the file will be stored, by default the same as the datafile
                if dir_image_path == "":
                    bpy.data.scenes['Scene'].render.filepath = bpy.data.scenes['Scene'].my_tool.path + str(x) + '.jpg'
                    
                    #Define a confirmation message to the default path            
                    def confirm_message(self, context):
                        self.layout.label("Render image saved at: " + bpy.data.scenes['Scene'].my_tool.path )

                else:                
                    bpy.data.scenes['Scene'].render.filepath = dir_image_path + str(x) + '.jpg'
                   
                    #Define a confirmation message to the selected path 
                    def confirm_message(self, context):
                        self.layout.label("Rendered image saved at: " + dir_image_path )   

                bpy.ops.render.render( write_still=True ) 

                bpy.ops.particle.forward()
                

            except:
                bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')


        bpy.context.window_manager.popup_menu(confirm_message, title="Saved successful", icon='SCENE')

        return{'FINISHED'} 


class OBJECT_OT_RenderVideoButton(bpy.types.Operator):
    bl_idname = "render.video"
    bl_label = "RenderizarVideo"
    country = bpy.props.StringProperty()


    #This code 
    def execute(self, context):

        dir_image_path = bpy.data.scenes['Scene'].my_tool.image_path

        #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Unable to save the Render. Try again with other path")


        try:    
            #Set the video format, AVI_JPEG by default            
            bpy.context.scene.render.image_settings.file_format = bpy.context.scene['VideoFormat'] 

        except:        
            bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG' 
        
        try:

            #Sets the path where the file will be stored, by default the same as the datafile
            if dir_image_path == "":
                bpy.data.scenes['Scene'].render.filepath = bpy.data.scenes['Scene'].my_tool.path + time.strftime("%c%s") + '.avi'
                
                #Define a confirmation message to the default path            
                def confirm_message(self, context):
                    self.layout.label("Rendered video saved at: " + bpy.data.scenes['Scene'].my_tool.path )

            else:                
                bpy.data.scenes['Scene'].render.filepath = dir_image_path + time.strftime("%c%s") + '.avi'
               
                #Define a confirmation message to the selected path 
                def confirm_message(self, context):
                    self.layout.label("Rendered video saved at: " + dir_image_path )   

            bpy.ops.render.render(animation=True, write_still=True)


            bpy.context.window_manager.popup_menu(confirm_message, title="Saved successful", icon='SCENE')

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        return{'FINISHED'}

class OBJECT_OT_CameraPlacement(bpy.types.Operator):
    bl_idname = "place.camera"
    bl_label = "Camera management"
    country = bpy.props.StringProperty()

    def execute(self, context):

        object_name = "Camera"  

        bpy.data.objects[object_name].location=(0,0,300)
        bpy.data.objects[object_name].rotation_euler=(0,0,0)
        bpy.data.objects[object_name].data.clip_end=1000


        return{'FINISHED'} 

class OBJECT_OT_CameraPlacement2(bpy.types.Operator):
    bl_idname = "place.camera2"
    bl_label = "Camera management2"
    country = bpy.props.StringProperty()

    def execute(self, context):

        object_name = "Camera"  

        bpy.data.objects[object_name].location=(0,-500,440)
        bpy.data.objects[object_name].rotation_euler=(0.872665,0,0)
        bpy.data.objects[object_name].data.clip_end=1000

        return{'FINISHED'} 

class OBJECT_OT_PlanePlacement(bpy.types.Operator):
    bl_idname = "place.plane"
    bl_label = "Plane management"
    country = bpy.props.StringProperty()

    def execute(self, context):

        bpy.ops.mesh.primitive_plane_add(radius=5, view_align=False, enter_editmode=False, location=(0, 0, 6), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
        
        cut_plane_name = "Cut_plane"
        bpy.context.object.name = "Cut_plane"
        bpy.data.objects[cut_plane_name].rotation_euler=(0,1.5708,0)

        if (bpy.context.scene.PlanesNumber =="2P"):
            bpy.ops.mesh.primitive_plane_add(radius=5, view_align=False, enter_editmode=False, location=(5, 0, 1), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
            cut_plane_name = "Cut_plane2"
            bpy.context.object.name = "Cut_plane2"
            bpy.data.objects[cut_plane_name].rotation_euler=(0,0,0)

        return{'FINISHED'} 

class OBJECT_OT_PlaneDelete(bpy.types.Operator):
    bl_idname = "delete.plane"
    bl_label = "Plane delete"
    country = bpy.props.StringProperty()

    def execute(self, context):

        bpy.context.object.select = 0

        if (bpy.context.scene.PlanesNumber =="1P"):
            bpy.data.objects["Cut_plane"].select = True
            bpy.ops.object.delete() 

        if (bpy.context.scene.PlanesNumber =="2P"):
            bpy.data.objects["Cut_plane"].select = True
            bpy.ops.object.delete() 
            bpy.data.objects["Cut_plane2"].select = True
            bpy.ops.object.delete() 

        return{'FINISHED'} 



class OBJECT_PT_my_panel(Panel):
    bl_idname = "OBJECT_PT_my_panel"
    bl_label = "Simulation Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_category = "Tools"
    bl_context = "objectmode"

class PanelSimulation(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "Simulation Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        col = layout.column(align=True)
        box0 = layout.box()

        box0.label(text="CALCULATION")

        box0.label(text="Select the folder with the data", icon='FILE_FOLDER')

        box0.prop(scn.my_tool, "folder_path", text="")

        box0.label(text="Select the type of psi matrix (2D by default)", icon='SETTINGS')

        box0.prop_search(context.scene, "CalculationFormat", context.scene, "calculationformats", text="" , icon='OBJECT_DATA')

        box0.label(text="Total particles number", icon='PARTICLE_DATA')

        box0.prop(scn.my_tool, "int_box_particulas_Simulacion")

        box0.operator("particles.calculation", text="Calculate data")
       


class PanelDataSelection(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "Data Selection Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        col = layout.column(align=True)
        box1 = layout.box()
        box2 = layout.box()


        box1.label(text="PARAMETERS")

        box1.label(text="Select the data file", icon='LIBRARY_DATA_DIRECT')

        box1.prop(scn.my_tool, "path", text="")

        box1.label(text="Select the number of particles to be shown by step", icon='PARTICLE_DATA')

        box1.prop(scn.my_tool, "int_box_n_particulas")



        box2.label(text="SIMULATION")

        box2.operator("particle.calculator", text="Place Particles")

        box2.operator("add.colors", text="Add Colors")

        box2.operator("reset.image", text="Reset Environment")


 

class PanelStates(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "States Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        col = layout.column(align=True)
        box22 = layout.box()


        #Box to move back and forward between states

        box22.label(text="STATES")

        col = box22.column(align = True)

        row = col.row(align = True)
        
        row.prop(scn.my_tool, "total_states_info")

        row.enabled = False 

        row_box = box22.row()

        row_box.prop(scn.my_tool, "int_box_state")

        row_box.enabled = True    

        row = box22.row()

        row.operator("particle.backward", text="Previous State", icon='BACK')

        row.operator("particle.forward", text="Next State", icon='FORWARD')

class PanelCut(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "Cut Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        box22 = layout.box()


        #Box to move back and forward between states

        box22.label(text="STATES")

        box22.label(text="Number of planes (1 by default)", icon='MOD_SOLIDIFY')

        box22.prop_search(context.scene, "PlanesNumber", context.scene, "planesnumber", text="" , icon='OBJECT_DATA')

        box22.label(text="Places the cut plane in the 3D view", icon='MOD_UVPROJECT')

        box22.operator("place.plane", text="Place plane")

        #box22.prop(scn.my_tool, "bool_cut_box", text="Inverse cut area")

        box22.operator("bool_cut_box", text="Place plane")

        box22.label(text="Makes a cut using the plane", icon='MOD_DECIM')

        box22.operator("particle.cut", text="Cut")

        box22.label(text="Delete planes", icon='FACESEL_HLT')

        box22.operator("delete.plane", text="Delete Planes")

        



class PanelRenderData(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "Rendering Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        col = layout.column(align=True)
        box23 = layout.box()
        box3 = layout.box()

        box23.label(text="CAMERA OPTIONS")

        box23.label(text="Camera predefs", icon='SCENE')

        box23.operator("place.camera", text="Sets camera to Top")

        box23.operator("place.camera2", text="Sets camera 50º angle")


        box3.label(text="RENDER")

        box3.label(text="Select the folder to store renders", icon='FILE_FOLDER')

        box3.prop(scn.my_tool, "image_path", text="")

        box3.label(text="Select the image format (PNG by default)", icon='RENDER_STILL')

        box3.prop_search(context.scene, "ImageFormat", context.scene, "imageformats", text="" , icon='OBJECT_DATA')

        box3.operator("render.image", text="Save image")

        box3.operator("render_all.image", text="Save all images")

        box3.label(text="Select the video format (AVI by default)", icon='RENDER_ANIMATION')

        box3.prop_search(context.scene, "VideoFormat", context.scene, "videoformats", text="" , icon='OBJECT_DATA')

        box3.operator("render.video", text="Save video")


class PanelInfoShortcuts(bpy.types.Panel):
    """Panel para añadir al entorno 3D"""
    bl_label = "Information Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    COMPAT_ENGINES = {'BLENDER_RENDER'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        col = layout.column(align=True)
        box4 = layout.box()

        box4.label(text="SHORTCUTS")

        box4.label(text="To switch view press SHIFT + Z", icon='INFO')

        box4.label(text="To start the animation press ALT + A", icon='INFO')

        box4.label(text="To modify grid values F6", icon='INFO')


        


# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def rellenar_selectores(scene):
    bpy.app.handlers.scene_update_pre.remove(rellenar_selectores)
    scene.imageformats.clear()
    scene.videoformats.clear()
    scene.calculationformats.clear()
    scene.planesnumber.clear()

    for identifier, name, description in image_format:
        scene.imageformats.add().name = name

    for identifier, name, description in video_format:
        scene.videoformats.add().name = name

    for identifier, name, description in calculation_format:
        scene.calculationformats.add().name = name

    for identifier, name, description in planes_number:
        scene.planesnumber.add().name = name


def register():
    bpy.utils.register_module(__name__)

    bpy.types.Scene.imageformats = bpy.props.CollectionProperty(
            type=bpy.types.PropertyGroup
        )

    bpy.types.Scene.videoformats = bpy.props.CollectionProperty(
            type=bpy.types.PropertyGroup
        )

    bpy.types.Scene.calculationformats = bpy.props.CollectionProperty(
            type=bpy.types.PropertyGroup
        )

    bpy.types.Scene.planesnumber = bpy.props.CollectionProperty(
            type=bpy.types.PropertyGroup
        )

    bpy.types.Scene.ImageFormat = bpy.props.StringProperty()

    bpy.types.Scene.VideoFormat = bpy.props.StringProperty()

    bpy.types.Scene.CalculationFormat = bpy.props.StringProperty()

    bpy.types.Scene.PlanesNumber = bpy.props.StringProperty()

    bpy.app.handlers.scene_update_pre.append(rellenar_selectores)

    bpy.types.Scene.my_tool = PointerProperty(type=MySettings)

    

def unregister():
    bpy.utils.unregister_module(__name__)
    del bpy.types.Scene.my_tool
    del bpy.types.Scene.coll
    del bpy.types.Scene.coll_string

if __name__ == "__main__":
    register()

bl_info = {    
    "name": "Particles calculator",    
    "category": "Object",
}


#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particle calculator                                                  #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticleCalculator(bpy.types.Operator):
    """My Object Moving Script"""                 # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particle.calculator"             # unique identifier for buttons and menu items to reference.
    bl_label = "Particle calculator"              # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}             # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.
       
        def sphere_object(x):
            if x == 0 : 
                emitter = bpy.data.objects['Sphere']
            if (x > 0 and x < 10) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            if (x >= 10 and x < 100) :
                emitter = bpy.data.objects['Sphere.0' + str(x)]
            if (x >= 100) :
                emitter = bpy.data.objects['Sphere.' + str(x)]
            return emitter
            

        #Creating the Icospheres that will give the particles the color gradiations as an origin of the material
        #Takes as input the materials vector
        def ico_creation(materials_vector): 
            number_of_icos=10 #10 plus the extra
            #First Ico and Group creation
            bpy.ops.mesh.primitive_ico_sphere_add(view_align=False, enter_editmode=False, location=(2, 2, -99), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
            bpy.context.object.name = "Ico_0"
            bpy.ops.object.group_add()
            bpy.data.groups["Group"].name = "Ico"
            bpy.ops.object.group_link(group='Ico')
            bpy.context.object.scale = (0.0001, 0.0001, 0.0001)
            bpy.context.active_object.active_material=materials_vector[0]

            #Icos iterative creation
            for cont in range(1, number_of_icos):
                bpy.ops.mesh.primitive_ico_sphere_add(view_align=False, enter_editmode=False, location=(2, 2, -100 + (-1*cont)), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
                bpy.context.object.name = "Ico_" + str(cont)
                bpy.ops.object.group_link(group='Ico')
                bpy.context.object.scale = (0.0001, 0.0001, 0.0001)
                bpy.context.active_object.active_material=materials_vector[cont]

           #Extra Ico for ordenation purpouses 
            bpy.ops.mesh.primitive_ico_sphere_add(view_align=False, enter_editmode=False, location=(2, 2, -111), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
            bpy.context.object.name = "Ico_9_extra"
            bpy.ops.object.group_link(group='Ico')
            bpy.context.object.scale = (0.0001, 0.0001, 0.0001)

        #Define an error message if occurs a problem during the run, is showed using a popup 
        def error_message(self, context):
            self.layout.label("No datafile selected. Remember to select a compatible datafile")

        #Delete all the old materials    
        for material in bpy.data.materials:
            material.user_clear();
            bpy.data.materials.remove(material);

        bpy.context.space_data.viewport_shade = 'MATERIAL'
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        bpy.context.scene.frame_current = 0    
        bpy.data.scenes["Scene"].my_tool.int_box_state = -1
        
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)   #Refresh the actual visualization with the new generator object placed  

        #Reading the data to generate the function who originated it
        #Read the data from te panel 
        path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
        
        try:
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        N = len(array_3d)   #Size of the matrix

        #Sets the maximum state avaliable
        bpy.context.scene.my_tool.total_states_info = N-1


        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas #Read from the panel 
        

        #Set the world color 
        bpy.context.scene.world.horizon_color = (0, 0, 0)
        #bpy.context.scene.world.horizon_color = (0.0041, 0.0087, 0.145)
        bpy.context.scene.world.light_settings.use_environment_light = True
        #Set the light propertyes
        bpy.data.objects['Lamp'].data.type = 'SUN'
        bpy.data.objects['Lamp'].location = (0,0,10)
        bpy.data.objects['Lamp'].rotation_euler = (0,0,10)

        #Emitter object, father of the particles
        ob = bpy.ops.mesh.primitive_uv_sphere_add(size=0.2, location=(0,0,-10000))
        bpy.context.object.scale = (0.0001, 0.0001, 0.0001)
        emitter = bpy.context.object

        bpy.ops.object.particle_system_add()    
        psys1 = emitter.particle_systems[-1]
                    
        psys1.name = 'Drops' 

        #Create materialsx
        #Creating the 10 materials to make the degradation
        mat_bur = bpy.data.materials.new('Burdeaux')
        mat_bur.diffuse_color = (0.174563, 0.004221, 0.0208302)
        mat_bur.type='SURFACE'

        mat_red = bpy.data.materials.new('Red')
        mat_red.diffuse_color = (0.8, 0, 0.0414922)
        mat_red.type='SURFACE'

        mat_ora = bpy.data.materials.new('Orange')
        mat_ora.diffuse_color = (0.801, 0.0885, 0.0372)
        mat_ora.type='SURFACE'

        mat_pap = bpy.data.materials.new('Papaya')
        mat_pap.diffuse_color = (0.8, 0.40282, 0)
        mat_pap.type='SURFACE'

        mat_lim = bpy.data.materials.new('Lima')
        mat_lim.diffuse_color = (0.517173, 0.8, 0)
        mat_lim.type='SURFACE'

        mat_gre = bpy.data.materials.new('Green')
        mat_gre.diffuse_color = (0.188, 0.8, 0)
        mat_gre.type='SURFACE'

        mat_tur = bpy.data.materials.new('Turquoise')
        mat_tur.diffuse_color = (0.006, 0.8, 0.366286)
        mat_tur.type='SURFACE'

        mat_sky = bpy.data.materials.new('Skyline')
        mat_sky.diffuse_color = (0, 0.266361, 0.8)
        mat_sky.type='SURFACE'

        mat_blu = bpy.data.materials.new('Blue')
        mat_blu.diffuse_color = (0.001, 0.0521, 0.8)
        mat_blu.type='SURFACE'

        mat_dar = bpy.data.materials.new('DarkBlue')
        mat_dar.diffuse_color = (0.017, 0, 0.8)
        mat_dar.type='SURFACE'

        #materials_vector = ([mat_dar, mat_blu, mat_sky, mat_tur, mat_gre, mat_lim, mat_pap, mat_ora, mat_red, mat_bur])
        materials_vector = ([mat_bur, mat_red, mat_ora, mat_pap, mat_lim, mat_gre, mat_tur, mat_sky, mat_blu, mat_dar])
        
        #Creating the Icospheres that will give the particles the color gradiations as an origin of the material
        ico_creation(materials_vector)

        psys1 = bpy.data.objects['Sphere'].particle_systems[-1]

        #Sets the configuration for the particle system of each emitter
        #configure_particles(psys1)
        psys1.settings.frame_start=bpy.context.scene.frame_current
        psys1.settings.frame_end=bpy.context.scene.frame_current+1
        psys1.settings.lifetime=1000
        psys1.settings.count = particles_number 

        psys1.settings.render_type='GROUP'
        psys1.settings.dupli_group=bpy.data.groups["Ico"]
        psys1.settings.use_group_count = True

        psys1.settings.normal_factor = 0.0
        psys1.settings.factor_random = 0.0
     
        # Physics
        psys1.settings.physics_type = 'NEWTON'
        psys1.settings.mass = 0
        psys1.settings.particle_size = 1000 #Remember the object scale 0.0001 of the icospheres to not be showed in the visualization
        psys1.settings.use_multiply_size_mass = False
     
        # Effector weights
        ew = psys1.settings.effector_weights
        ew.gravity = 0
        ew.wind = 0


        nombreMesh = "Figura" + str(0)
        me = bpy.data.meshes.new(nombreMesh)

        psys1 = emitter.particle_systems[-1] 

        x_pos = 0
        y_pos = 0
        z_pos = 0
        prob = 0
        cont = 0

        #Use an auxiliar array to work with a variable number of points, 
        #allowing the user to make diferent points simulation with good results
        array_aux = np.zeros((particles_number, 4))
        #Fill the auxiliar array with the data of the original one 
        for point_number in range (0, particles_number):
            array_aux[point_number] = array_3d[0][point_number]

        #Sort the array to place first the particles with more probability
        array_aux = array_aux[np.argsort(array_aux[:,3])]

        for pa in psys1.particles:
            #God´s particle solution
            #if pa.die_time < 500 :
            pa.die_time = 500
            pa.lifetime = 500
            pa.velocity = (0,0,0)
            #3D placement
            x_pos = array_aux[cont][0] 
            y_pos = array_aux[cont][1] 
            z_pos = array_aux[cont][2]
            pa.location = (x_pos,y_pos,z_pos)
            prob = array_aux[cont][3] 
            cont += 1 
        
        bpy.context.scene.frame_current = bpy.context.scene.frame_current + 1   #Goes one frame forward to show particles clear at rendering MANDATORY


        file_with_binary_data.close()

        bpy.ops.particle.stabilizer()

        #bpy.ops.particle.generation() #Next step, go to particle generation

        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticleCalculator)


def unregister():
    bpy.utils.unregister_class(ParticleCalculator)
    
# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()   


bl_info = {    
    "name": "Particles Stabilizer",    
    "category": "Object",
}

#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particles Stabilizer                                                 #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticlesStabilizer(bpy.types.Operator):
    """My Object Moving Script"""               # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particle.stabilizer"           # unique identifier for buttons and menu items to reference.
    bl_label = "Particles Stabilization"        # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}           # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.

        #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Imposible to stabilize particles. Try to Run simulation again")

        #return the name of the emitter wich is asigned to this number by order
        def emitter_system(x):
            if x == 0 : 
                emitter = bpy.data.objects['Sphere']
            if (x > 0 and x < 10) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            if (x >= 10 and x < 100) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            return emitter.particle_systems[-1] 

        path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
        try:
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        N = len(array_3d[0])   #Size of the matrix

        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas #Read from the panel 
        
        x_pos = 0
        y_pos = 0
        z_pos = 0
        prob = 0
        cont = 0

        actual_state = bpy.data.scenes["Scene"].my_tool.int_box_state 

        if (actual_state == -1):
            actual_state=0

        object_name = "Sphere"  
        emitter = bpy.data.objects[object_name]  
        psys1 = emitter.particle_systems[-1] 

        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas

        #Use an auxiliar array to work with a variable number of points, 
        #allowing the user to make diferent points simulation with good results
        array_aux = np.zeros((particles_number, 4))
        #Fill the auxiliar array with the data of the original one
        for point_number in range (0, particles_number):
            array_aux[point_number] = array_3d[actual_state][point_number]

        array_aux = array_aux[np.argsort(array_aux[:,3])]
        for pa in psys1.particles:
            #God´s particle solution
            #if pa.die_time < 500 :
            pa.die_time = 500
            pa.lifetime = 500
            pa.velocity = (0,0,0)
            #3D placement
            x_pos = array_aux[cont][0] 
            y_pos = array_aux[cont][1] 
            z_pos = array_aux[cont][2]
            pa.location = (x_pos,y_pos,z_pos)
            prob = array_aux[cont][3] 
            cont += 1 


        file_with_binary_data.close()


        #bpy.context.scene.frame_current = bpy.context.scene.frame_current + 1
        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticlesStabilizer)


def unregister():
    bpy.utils.unregister_class(ParticlesStabilizer)
    
# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()   



bl_info = {    
    "name": "Particles Forward",    
    "category": "Object",
}

import bpy
import time
import math
import random
import struct
import binascii
import numpy as np

#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particles forward                                                    #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticlesForward(bpy.types.Operator):
    """My Object Moving Script"""               # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particle.forward"           # unique identifier for buttons and menu items to reference.
    bl_label = "Particles Forward"        # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}           # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.
    #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Imposible to read from original file. Try to Run simulation again")

        def draw(self, context):
            self.layout.label("Returned to the origin state")

        def sphere_object(x):
            if x == 0 : 
                emitter = bpy.data.objects['Sphere']
            if (x > 0 and x < 10) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            if (x >= 10 and x < 100) :
                emitter = bpy.data.objects['Sphere.0' + str(x)]
            if (x >= 100) :
                emitter = bpy.data.objects['Sphere.' + str(x)]
            return emitter


        #Calculates the position of spheres in a state given
        def sphere_placement(state, array_3d):
            actual_state = state
            particles_number = bpy.context.scene.my_tool.int_box_n_particulas

            x_pos = 0
            y_pos = 0
            z_pos = 0

            prob = 0
            cont = 0

            object_name = "Sphere"  
            emitter = bpy.data.objects[object_name]  
            psys1 = emitter.particle_systems[-1]



            #Use an auxiliar array to work with a variable number of points, 
            #allowing the user to make diferent points simulation with good results
            array_aux = np.zeros((particles_number, 4))
            #Fill the auxiliar array with the data of the original one 
            for point_number in range (0, particles_number):
                array_aux[point_number] = array_3d[actual_state][point_number]

            #Sort the array to place first the particles with more probability
            array_aux = array_aux[np.argsort(array_aux[:,3])]

            #With the array sorted the last position is the particle with the biggest probability to appear
            actual_max_prob_value = array_aux[particles_number-1][3]
            #With the array sorted the first position is the particle with less probability to appear
            actual_min_prob_value = array_aux[0][3]

            general_maximum = actual_max_prob_value
            general_minimum = actual_min_prob_value

            for state_array in array_3d:
                maxi = np.max(state_array[:,3])
                mini = np.max(state_array[:,3])

                if (maxi > general_maximum):
                    general_maximum = maxi
                if (mini < general_minimum):
                    general_minimum = mini

            #Obtain an scalated step to distribute the points between the 10 scales of probability
            step_prob = (general_maximum-general_minimum)/10
            prob_using = step_prob

            steps = np.zeros(11)
            actual_step = 9

            for cont_particle in range(particles_number):
                if(array_aux[cont_particle][3] < prob_using):
                    steps[actual_step] += 1
                else:
                    actual_step -= 1
                    prob_using += step_prob 

            #solves the problem with the extra particles not asigned to the last position
            #steps[9] += steps[10]

            bpy.data.objects['Sphere'].select = True
            bpy.context.scene.objects.active = bpy.data.objects['Sphere']
            for cont_mat in range(10):
                #Avoid the Ico9_extra problem using "9-"
                bpy.data.objects['Sphere'].particle_systems['Drops'].settings.active_dupliweight_index = 9-cont_mat
                dupli_weights_name = bpy.data.objects['Sphere'].particle_systems['Drops'].settings.active_dupliweight.name
                material_value = bpy.data.objects['Sphere'].particle_systems['Drops'].settings.dupli_weights.get(dupli_weights_name)
                material_value.count = steps[cont_mat]

            for pa in psys1.particles:
                #God´s particle solution
                #if pa.die_time < 500 :
                pa.die_time = 500
                pa.lifetime = 500
                pa.velocity = (0,0,0)
                #3D placement
                x_pos = array_aux[cont][0] 
                y_pos = array_aux[cont][1] 
                z_pos = array_aux[cont][2]
                pa.location = (x_pos,y_pos,z_pos)
                prob = array_aux[cont][3] 
                cont += 1 

            bpy.ops.particle.stabilizer()

        #Take the actual state
        actual_state = bpy.data.scenes["Scene"].my_tool.int_box_state
        
        #Calculate the total of states
        try:
            path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

            total_states = len(array_3d)

            file_with_binary_data.close()

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')


        #First time do this
        if(actual_state == -1):
            #Calculate the coordinates of each particle
            bpy.data.scenes["Scene"].my_tool.int_box_state = 1
            sphere_placement(1, array_3d)
            
        
        else:
            #If is not the last state
            if((actual_state+1) < int(total_states)):
                #Take the name of the Sphere to make the complete name and disable it
                bpy.data.scenes["Scene"].my_tool.int_box_state = actual_state + 1
                sphere_placement(actual_state+1, array_3d)
                
            #If its the last state 
            if((actual_state+1) >= int(total_states)):
                bpy.data.scenes["Scene"].my_tool.int_box_state = 0
                sphere_placement(0, array_3d)
                

        #Goes one frame forward
        bpy.context.scene.frame_current = bpy.context.scene.frame_current + 1




        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticlesForward)


def unregister():
    bpy.utils.unregister_class(ParticlesForward)
    
# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()   

bl_info = {    
    "name": "Particles Backward",    
    "category": "Object",
}


#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particles backward                                                   #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticlesBackward(bpy.types.Operator):
    """My Object Moving Script"""               # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particle.backward"           # unique identifier for buttons and menu items to reference.
    bl_label = "Particles Backward"        # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}           # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.
    #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Imposible to read from original file. Try to Run simulation again")

        def draw(self, context):
            self.layout.label("Returned to the origin state")

        def sphere_object(x):
            if x == 0 : 
                emitter = bpy.data.objects['Sphere']
            if (x > 0 and x < 10) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            if (x >= 10 and x < 100) :
                emitter = bpy.data.objects['Sphere.0' + str(x)]
            if (x >= 100) :
                emitter = bpy.data.objects['Sphere.' + str(x)]
            return emitter
 

        #Calculates the position of spheres in a state given
        def sphere_placement(state, array_3d):
            actual_state = state
            particles_number = bpy.context.scene.my_tool.int_box_n_particulas

            x_pos = 0
            y_pos = 0
            z_pos = 0

            prob = 0
            cont = 0

            object_name = "Sphere"  
            emitter = bpy.data.objects[object_name]  
            psys1 = emitter.particle_systems[-1]


            #Use an auxiliar array to work with a variable number of points, 
            #allowing the user to make diferent points simulation with good results
            array_aux = np.zeros((particles_number, 4))
            #Fill the auxiliar array with the data of the original one 
            for point_number in range (0, particles_number):
                array_aux[point_number] = array_3d[actual_state][point_number]

            #Sort the array to place first the particles with more probability
            array_aux = array_aux[np.argsort(array_aux[:,3])]

            #With the array sorted the last position is the particle with the biggest probability to appear
            actual_max_prob_value = array_aux[particles_number-1][3]
            #With the array sorted the first position is the particle with less probability to appear
            actual_min_prob_value = array_aux[0][3]

            general_maximum = actual_max_prob_value
            general_minimum = actual_min_prob_value

            for state_array in array_3d:
                maxi = np.max(state_array[:,3])
                mini = np.max(state_array[:,3])

                if (maxi > general_maximum):
                    general_maximum = maxi
                if (mini < general_minimum):
                    general_minimum = mini

            #Obtain an scalated step to distribute the points between the 10 scales of probability
            step_prob = (general_maximum-general_minimum)/10
            prob_using = step_prob

            steps = np.zeros(11)
            actual_step = 9

            for cont_particle in range(particles_number):
                if(array_aux[cont_particle][3] < prob_using):
                    steps[actual_step] += 1
                else:
                    actual_step -= 1
                    prob_using += step_prob 

            #solves the problem with the extra particles not asigned to the last position
            steps[9] += steps[10]

            bpy.data.objects['Sphere'].select = True
            bpy.context.scene.objects.active = bpy.data.objects['Sphere']
            for cont_mat in range(10):
                #Avoid the Ico9_extra problem using "9-"
                bpy.data.objects['Sphere'].particle_systems['Drops'].settings.active_dupliweight_index = 9-cont_mat
                dupli_weights_name = bpy.data.objects['Sphere'].particle_systems['Drops'].settings.active_dupliweight.name
                material_value = bpy.data.objects['Sphere'].particle_systems['Drops'].settings.dupli_weights.get(dupli_weights_name)
                material_value.count = steps[cont_mat]

            for pa in psys1.particles:
                #God´s particle solution
                #if pa.die_time < 500 :
                pa.die_time = 500
                pa.lifetime = 500
                pa.velocity = (0,0,0)
                #3D placement
                x_pos = array_aux[cont][0] 
                y_pos = array_aux[cont][1] 
                z_pos = array_aux[cont][2]
                pa.location = (x_pos,y_pos,z_pos)
                prob = array_aux[cont][3] 
                cont += 1

            bpy.ops.particle.stabilizer()


        #Take the actual state
        actual_state = bpy.data.scenes["Scene"].my_tool.int_box_state
        
        #Calculate the total of states
        try:
            path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

            total_states = len(array_3d)

            file_with_binary_data.close()

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')



        #First time do this
        if(actual_state == -1):
            bpy.data.scenes["Scene"].my_tool.int_box_state = total_states - 1
            sphere_placement(total_states-1,array_3d)
            
        else:
            #If is not the last state
            if((actual_state-1) >= 0):
                bpy.data.scenes["Scene"].my_tool.int_box_state = actual_state - 1
                sphere_placement(actual_state-1,array_3d)

            #If its the last state
            if((actual_state-1) < 0):
                bpy.data.scenes["Scene"].my_tool.int_box_state = total_states - 1
                sphere_placement(total_states-1,array_3d)


        #Goes one frame forward
        bpy.context.scene.frame_current = bpy.context.scene.frame_current + 1


        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticlesBackward)


def unregister():
    bpy.utils.unregister_class(ParticlesBackward)
    
# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register() 

bl_info = {    
    "name": "Particles Calculation",    
    "category": "Object",
}

#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particles calculation                                                #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticlesCalculation(bpy.types.Operator):
    """My Object Moving Script"""               # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particles.calculation"           # unique identifier for buttons and menu items to reference.
    bl_label = "Particles Calculation"        # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}           # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.
        
        def error_message(self, context):
            self.layout.label("Imposible to read psi data from the selected folder.")

        #Set the calculation format, 2D by default
        calculation_format_final = bpy.data.scenes["Scene"].CalculationFormat
        if (calculation_format_final == ''):
            calculation_format_final = '2D'
        #Takes the data from the folder with all psi files
        try:
            path = bpy.data.scenes['Scene'].my_tool.folder_path #Origin from where the data will be readen, selected by the first option in the Panel
            
            psi_files_number=0

            if (calculation_format_final == '2D'):
                while (os.path.isfile(path+ str(psi_files_number) +"psi")):
                    psi_files_number += 1

            if (calculation_format_final == '3D'):
                while (os.path.isfile(path+ "psi_" + str(psi_files_number) + ".pkl")):
                    psi_files_number += 1
            


        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')
    
        #number of 3D points for each step
        number_of_points=bpy.context.scene.my_tool.int_box_particulas_Simulacion
        #3D matrix creation
        matrix_3d = np.zeros((psi_files_number,number_of_points,4))

        #Data storage matrix
        array_aux = np.zeros((number_of_points, 4))
 

        path=bpy.data.scenes['Scene'].my_tool.folder_path


        #2D calculation    
        if (calculation_format_final == '2D'):
            for cont_file in range(0, psi_files_number):

                file_with_binary_data = open(path+ str(cont_file) +"psi", 'rb+') #File with binary data

                array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)

                #Matrix with the data of the 2D grid
                Z = array_with_all_data['arr_0'] 
                
                cArray.matrix2Dprob(Z, array_aux)

                matrix_3d[cont_file]=array_aux

        #3D calculation 
        if (calculation_format_final == '3D'):
            for cont_file in range(0, psi_files_number):

                file_with_binary_data = open(path+ "psi_" + str(cont_file) + ".pkl", 'rb+') #File with binary data

                array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)

                #Matrix with the data of the 3D grid
                Z = array_with_all_data['arr_0'] 

                cArray.matrix3Dprob(Z, array_aux)

                matrix_3d[cont_file]=array_aux

        
        f = open(path + '3dData.3d', 'wb+')
        np.savez(f, matrix_3d)
        f.close()

        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticlesCalculation)

def unregister():
    bpy.utils.unregister_class(ParticlesCalculation)

bl_info = {    
    "name": "Particles Stabilizer",    
    "category": "Object",
}

#*************************************************************************# 
# ----------------------------------------------------------------------- #
#    Particles Stabilizer                                                 #
# ----------------------------------------------------------------------- #
#*************************************************************************# 

class ParticlesCut(bpy.types.Operator):
    """My Object Moving Script"""               # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "particle.cut"           # unique identifier for buttons and menu items to reference.
    bl_label = "Particles Cut"        # display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}           # enable undo for the operator.
   
    def execute(self,context):        # execute() is called by blender when running the operator.

        #Define an error message if occurs a problem during the run, is showed using a popup
        def error_message(self, context):
            self.layout.label("Imposible to stabilize particles. Try to Run simulation again")

        #return the name of the emitter wich is asigned to this number by order
        def emitter_system(x):
            if x == 0 : 
                emitter = bpy.data.objects['Sphere']
            if (x > 0 and x < 10) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            if (x >= 10 and x < 100) :
                emitter = bpy.data.objects['Sphere.00' + str(x)]
            return emitter.particle_systems[-1] 

        path = bpy.data.scenes['Scene'].my_tool.path #Origin from where the data will be readen, selected by the first option in the Panel
        try:
            file_with_binary_data = open(path, 'rb+') #File with binary data

            array_with_all_data = np.load(file_with_binary_data) #Gets the binary data as an array with 6 vectors (x_data, x_probability, y_data, y_probability, z_data, z_probability)
       
            #Matrix with the data of the 2D grid
            array_3d = array_with_all_data['arr_0'] 

        except:
            bpy.context.window_manager.popup_menu(error_message, title="An error ocurred", icon='CANCEL')

        N = len(array_3d[0])   #Size of the matrix

        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas #Read from the panel 
        
        x_pos = 0
        y_pos = 0
        z_pos = 0
        prob = 0
        cont = 0

        actual_state = bpy.data.scenes["Scene"].my_tool.int_box_state 

        if (actual_state == -1):
            actual_state=0

        object_name = "Sphere"  
        emitter = bpy.data.objects[object_name]  
        psys1 = emitter.particle_systems[-1] 

        particles_number = bpy.data.scenes['Scene'].my_tool.int_box_n_particulas

        #Use an auxiliar array to work with a variable number of points, 
        #allowing the user to make diferent points simulation with good results
        array_aux = np.zeros((particles_number, 4))
        #Fill the auxiliar array with the data of the original one
        for point_number in range (0, particles_number):
            array_aux[point_number] = array_3d[actual_state][point_number]

        #Plane info
        if(bpy.context.scene.PlanesNumber == "1P"):
            cut_plane_1= "Cut_plane"
            plane_pos_1 = bpy.data.objects[cut_plane_1].location
            plane_size_1 = bpy.data.objects[cut_plane_1].dimensions

            array_aux = array_aux[np.argsort(array_aux[:,3])]
            for pa in psys1.particles:
                #God´s particle solution
                #if pa.die_time < 500 :
                pa.die_time = 500
                pa.lifetime = 500
                pa.velocity = (0,0,0)
                #3D placement
                x_pos = array_aux[cont][0] 
                y_pos = array_aux[cont][1] 
                z_pos = array_aux[cont][2]
                bpy.context.scene.my_tool.bool_cut_box
                if(x_pos > (plane_pos_1[0])):
                    pa.location = (-10000,-10000,-10000)
                prob = array_aux[cont][3] 
                cont += 1

        if(bpy.context.scene.PlanesNumber == "2P"):
            cut_plane_1= "Cut_plane"
            plane_pos_1 = bpy.data.objects[cut_plane_1].location
            plane_size_1 = bpy.data.objects[cut_plane_1].dimensions

            cut_plane_2= "Cut_plane2"
            plane_pos_2 = bpy.data.objects[cut_plane_2].location
            plane_size_2 = bpy.data.objects[cut_plane_2].dimensions

            array_aux = array_aux[np.argsort(array_aux[:,3])]
            for pa in psys1.particles:
                #God´s particle solution
                #if pa.die_time < 500 :
                pa.die_time = 500
                pa.lifetime = 500
                pa.velocity = (0,0,0)
                #3D placement
                x_pos = array_aux[cont][0] 
                y_pos = array_aux[cont][1] 
                z_pos = array_aux[cont][2]
                if(x_pos > plane_pos_1[0] and z_pos > plane_pos_2[2]):
                    pa.location = (-10000,-10000,-10000)
                prob = array_aux[cont][3] 
                cont += 1 





        file_with_binary_data.close()


        #bpy.context.scene.frame_current = bpy.context.scene.frame_current + 1
        return {'FINISHED'}            # this lets blender know the operator finished successfully.

# ------------------------------------------------------------------------
#    Register and unregister functions
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ParticlesCut)


def unregister():
    bpy.utils.unregister_class(ParticlesCut)
    
# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()  