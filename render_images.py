# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math
import sys
import random
import json
import os
import tempfile
from datetime import datetime as dt
from collections import Counter
from dataclasses import dataclass
import numpy as np

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

@dataclass
class args:
    base_scene_blendfile: str = 'data/base_scene.blend'
    properties_json: str = 'data/properties.json'
    properties_arrow_json: str = 'data/properties_arrow.json'
    shape_dir: str = 'data/shapes'
    material_dir: str = 'data/materials'
    min_dist: float = 0.25
    margin: float = 0.4
    min_pixels_per_object: int = 200
    max_retries: int = 50
    use_gpu: int = 1
    width: int = 320
    height: int = 240
    key_light_jitter: float = 1.0
    fill_light_jitter: float = 1.0
    back_light_jitter: float = 1.0
    camera_jitter: float = 0.5
    render_num_samples: int = 512
    render_min_bounces: int = 8
    render_max_bounces: int = 8
    restricted_zone: float = 0.1


def render_scene(objects_candidate,
                 output_index=0,
                 output_split='none',
                 output_image='render.png',
                 output_blendfile=None,
                 args=args(),
                 arrow_idx=None,
                 ):

    # replace png to json
    output_scene = output_image.replace('.png', '.json')

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    # render_args.tile_x = args.render_tile_size
    # render_args.tile_y = args.render_tile_size

    if args.use_gpu == 1:
        # set gpu rendering
        '''credit to https://github.com/nytimes/rd-blender-docker/issues/3'''
        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'

        prefs = bpy.context.preferences
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        prefs.addons['cycles'].preferences.compute_device_type = 'CUDA'
        prefs.addons['cycles'].preferences.devices[0].use = True

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=5, calc_uvs=False)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(
                args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(
                args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(
                args.fill_light_jitter)

    # Now make some random objects
    if arrow_idx is not None:
        objects = add_objects_w_arrow(
            scene_struct, objects_candidate, arrow_idx, args, camera)        
        scene_struct['points_to'] = arrow_idx
    else:
        objects = add_objects(
            scene_struct, objects_candidate, args, camera)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    return scene_struct

def add_objects(scene_struct, objects_candidate, args, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba

        size_mapping = list(properties['sizes'].items())

    positions = []
    objects = []
    blender_objects = []
    for obj in objects_candidate:
        obj_name, (shape_name, color_name, mat_name, size_name) = obj

        r = properties['sizes'][size_name]
        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_objects(scene_struct, objects_candidate, args, camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(margin, args.margin, direction_name)
                        print('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose random color and shape

        rgba = color_name_to_rgba[color_name]

        # For cube, adjust the size a bit
        if shape_name == 'Cube':
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        utils.add_object(
            args.shape_dir, properties['shapes'][shape_name], r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach a random material
        utils.add_material(properties['materials'][mat_name], Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)

        objects.append({
            'name': obj_name,
            'shape': shape_name,
            'size': size_name,
            'material': mat_name,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })

    # # Check that all objects are at least partially visible in the rendered image
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')

        for obj in blender_objects:
            print(obj.name)
            utils.delete_object(obj)
        objects = add_objects(scene_struct, objects_candidate, args, camera)
        print(objects)
        return objects

    return objects

def add_objects_w_arrow(scene_struct, objects_candidate, arrow_idx, args, camera):
    """
    Add random objects to the current blender scene, with arrows
    """

    # Load the property file
    with open(args.properties_arrow_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba

        size_mapping = list(properties['sizes'].items())
        obj_colors = list(properties['colors'].keys())
    
    canvas_grid_size = 600
    canvas_actual_size = 5.5
    canvas_bias = [-0.5, 0.5] # y, x
    phase_restrict_corner = True
    restricted_zone = int(args.restricted_zone * canvas_grid_size)

    positions = []
    objects = []
    blender_objects = []

    mask_object = np.ones(shape=(canvas_grid_size, canvas_grid_size))

    for i, obj in enumerate(objects_candidate):
        obj_name, (shape_name, color_name, mat_name, size_name) = obj
        mask = np.ones(shape=(canvas_grid_size, canvas_grid_size))

        if i == 0:  # upper left, in blender left
            mask[:, canvas_grid_size // 2 - restricted_zone:] = 0
            mask[canvas_grid_size // 2 - restricted_zone:] = 0
        elif i == 1:  # lower left, in blender back
            mask[:, canvas_grid_size // 2 - restricted_zone:] = 0
            mask[:canvas_grid_size // 2 + restricted_zone] = 0
        elif i == 2:  # lower right, in blender right
            mask[:, :canvas_grid_size // 2 + restricted_zone] = 0
            mask[:canvas_grid_size // 2 + restricted_zone] = 0
        elif i == 3:  # upper right, in blender front
            mask[:, :canvas_grid_size // 2 + restricted_zone] = 0
            mask[canvas_grid_size // 2 - restricted_zone:] = 0

        if phase_restrict_corner:
            mask[:restricted_zone, :restricted_zone] = 0
            mask[-restricted_zone:, -restricted_zone:] = 0

        r = properties['sizes'][size_name]

        num_tries = 0

        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_objects_w_arrow(scene_struct, objects_candidate, arrow_idx, args, camera)

            coor, new_mask = utils.draw_location(args, mask * mask_object, r, properties['sizes']['small'],
                                                 canvas_grid_size, canvas_actual_size, phase_output_object_mask=True)

            if coor is None:
                continue

            x = coor[1] + canvas_bias[1]
            y = coor[0] + canvas_bias[0]
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                # not doing margin check on arrow
                if shape_name != 'arrow':
                    for direction_name in ['left', 'right']:
                        direction_vec = scene_struct['directions'][direction_name]
                        assert direction_vec[2] == 0
                        margin = dx * direction_vec[0] + dy * direction_vec[1]
                        if 0 < margin < args.margin:
                            print(margin, args.margin, direction_name)
                            print('BROKEN MARGIN!')
                            margins_good = False
                            break
                    # for direction_name in ['left', 'right', 'front', 'behind']:
                    #     direction_vec = scene_struct['directions'][direction_name]
                    #     assert direction_vec[2] == 0
                    #     margin = dx * direction_vec[0] + dy * direction_vec[1]
                    #     if 0 < margin < args.margin:
                    #         print(margin, args.margin, direction_name)
                    #         print('BROKEN MARGIN!')
                    #         margins_good = False
                    #         break
                if not margins_good:
                    break

            if dists_good and margins_good:
                mask_object = new_mask * mask_object
                break

        # ignore the folloing part for cube since we want the cube to be bigger
        # # For cube, adjust the size a bit
        # if obj_name_out == 'cube':
        #     r /= math.sqrt(2)
        rgba = color_name_to_rgba[color_name]

        if shape_name == 'arrow':
            cube_x, cube_y, _ = positions[arrow_idx]
            theta = math.pi - math.atan2((cube_x - x), (cube_y - y))
            r = 0.6
            rgba = [240 / 255, 191 / 255, 29 / 255, 1]
        else:
            theta = 2 * math.pi * random.random()

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, properties['shapes'][shape_name], r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        utils.add_material(properties['materials'][mat_name], Color=rgba)
        # utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)

        objects.append({
            'name': obj_name,
            'shape': shape_name,
            'size': size_name,
            'material': mat_name,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })

    # # Check that all objects are at least partially visible in the rendered image
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')

        for obj in blender_objects:
            print(obj.name)
            utils.delete_object(obj)
        objects = add_objects_w_arrow(scene_struct, objects_candidate, arrow_idx, args, camera)
        return objects

    return objects 

def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = {}
        for i, obj1 in enumerate(scene_struct['objects']):
            if obj1['name'] is None:
                continue
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj2['name'] is None:
                    continue
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(obj2['name'])
            all_relationships[name][obj1['name']] = list(related)
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                          for i in range(0, len(p), 4))
    objects_rendered = 0
    os.close(f)
    os.remove(path)
    for _, count in color_count.most_common():
        if count >= min_pixels_per_object:
            objects_rendered += 1
        else:
            break
    
    if objects_rendered == len(blender_objects) + 1:
        return True
    else:
        return False


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.simplify_gpencil_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'CYCLES'
    render_args.simplify_gpencil_antialiasing = False

    # hid the lights and ground so they don't render
    bpy.data.objects['Lamp_Key'].hide_render = True
    bpy.data.objects['Lamp_Fill'].hide_render = True
    bpy.data.objects['Lamp_Back'].hide_render = True
    bpy.data.objects['Ground'].hide_render = True
    
    # Add random shadeless materials to all objects
    object_colors = 0
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        obj.data.materials[0] = bpy.data.materials['Shadeless']
        object_colors += 1

    # Render the scene
    bpy.ops.render.render(write_still=True)
    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    bpy.data.objects['Lamp_Key'].hide_render = False
    bpy.data.objects['Lamp_Fill'].hide_render = False
    bpy.data.objects['Lamp_Back'].hide_render = False
    bpy.data.objects['Ground'].hide_render = False

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.simplify_gpencil_antialiasing = old_use_antialiasing

    return object_colors
