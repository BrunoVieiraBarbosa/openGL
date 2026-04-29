import os
import random

import arcade
import numpy
from OpenGL.GL import *

from core.core import *
from core.light import DirectionalLight, PointLight
from core.mesh import Mesh, MeshRGB, TerrainGridSampler
from core.utils import *


class GameWindow(App):
    def __init__(self):
        size = (1280, 720)
        super().__init__(size, ambient_color=(0.62, 0.67, 0.73, 1.0))
        self.fog_color = numpy.array([0.78, 0.79, 0.74], dtype=numpy.float32)
        self.fog_near = 14.0
        self.fog_far = 42.0
        self.set_mouse_visible(False)
        self.set_exclusive_mouse(True)
        self.cleaned_up = False
        self._setup_scene()

    def _setup_scene(self):
        self.terrain_origin = numpy.array([10.0, 4.0, -0.02], dtype=numpy.float32)
        self.terrain_glb_path = os.path.join("obj", "terrain_main.glb")
        self.terrain_sampler = TerrainGridSampler.from_glb(self.terrain_glb_path)
        self.terrain_height = lambda world_x, world_y: self.terrain_sampler.sample_height(
            world_x - self.terrain_origin[0],
            world_y - self.terrain_origin[1],
        ) + self.terrain_origin[2]
        self.terrain_contains = lambda world_x, world_y, radius=0.0: self.terrain_sampler.contains_circle(
            world_x - self.terrain_origin[0],
            world_y - self.terrain_origin[1],
            radius,
        )

        self.add_shader("first", Shader.create_shader("shaders/vertex.c", "shaders/fragment.c"))
        self.add_shader("simple", Shader.create_shader("shaders/vertex_rgb.c", "shaders/fragment_rgb.c"))

        self.start_()

        self.light = [
            DirectionalLight(
                [self.shaders[0], self.shaders[1]],
                [-0.4, -0.8, -1.0],
                [1.0, 0.95, 0.86],
                30,
                0,
                [True, False],
            ),
            PointLight([self.shaders[0], self.shaders[1]], [6, -2, 8], [1.0, 0.96, 0.9], 32, 1, [True, False]),
            PointLight(
                [self.shaders[0], self.shaders[1]],
                [20, 6, 10],
                [0.85, 0.9, 1.0],
                32,
                2,
                [True, False],
            ),
        ]

        self.lampada = [
            MeshRGB(self.shaders[1], self.light[1], color=[1, 0.95, 0.8]),
            MeshRGB(self.shaders[1], self.light[2], color=[0.7, 0.85, 1.0]),
        ]
        self.sky = MeshRGB(
            self.shaders[1],
            self.camera if hasattr(self, "camera") else [0, 0, 0],
            vertices=MeshRGB.create_gradient_box(
                size=1.0,
                top_color=(0.40, 0.64, 0.93),
                bottom_color=(0.96, 0.82, 0.58),
            ),
            scale=90.0,
        )

        self.crate_texture = Material(
            os.path.join("textures", "box.jpg"),
            os.path.join("textures", "box_specular.jpg"),
            Material._solid_image((128, 128, 255, 255)),
        )
        self.ground_texture = Material(
            os.path.join("textures", "block.png"),
            os.path.join("textures", "block_specular.png"),
            os.path.join("textures", "normal.jpg"),
        )

        self.terrain_meshes, self.terrain_materials = self._build_glb_model(
            self.terrain_glb_path,
            self.terrain_origin,
            target_size=52.0,
            normalize=False,
        )
        self.terrain = self._get_primary_mesh(self.terrain_meshes)

        glb_scene = [
            ("IronMan.glb", [27, 1, 0], 1.8, 0.38, 0.1),
            ("break_time.glb", [12, 12, 0], 1.8, 0.44, 0.1),
        ]
        self.scene_meshes = []
        self.scene_materials = []
        for file_name, position, target_size, collider_radius_scale, collider_radius_padding in glb_scene:
            grounded_position = [position[0], position[1], self.terrain_height(position[0], position[1])]
            model_meshes, model_materials = self._build_glb_model(
                os.path.join("obj", file_name),
                grounded_position,
                target_size=target_size,
                normalize=True,
            )
            self.scene_materials.extend(model_materials)
            primary_mesh = max(model_meshes, key=lambda mesh: mesh.vertex_count)
            primary_mesh.extra_meshes = [mesh for mesh in model_meshes if mesh is not primary_mesh]
            primary_mesh.set_collider(
                mode="circle",
                radius_scale=collider_radius_scale,
                radius_padding=collider_radius_padding,
                height_padding=0.1,
            )
            self.scene_meshes.append(primary_mesh)

        self.cubes = [
            Mesh(
                self.shaders[0],
                self.crate_texture,
                [
                    random.randint(-6, 30),
                    random.randint(-8, 16),
                    0.0,
                ],
                scale=0.18,
            )
            for _ in range(10)
        ]
        for cube in self.cubes:
            cube.position[2] = self.terrain_height(cube.position[0], cube.position[1]) + 0.18
            cube.set_collider(mode="circle", radius_scale=1.0, radius_padding=0.08, height_padding=0.08)

        player_start = numpy.array([10.0, -14.0, self.terrain_height(10, -14)], dtype=numpy.float32)
        self.player_meshes, self.player_materials = self._build_glb_model(
            os.path.join("obj", "Player.glb"),
            player_start,
            target_size=1.9,
            normalize=True,
        )
        self.player_mesh = self._get_primary_mesh(self.player_meshes)
        self.player_mesh.set_collider(mode="circle", radius_scale=0.55, radius_padding=0.06, height_padding=0.12)

        self.camera = CameraThirdPerson(player_start, distance=5.6, height=1.55)
        self.camera.theta = 72
        self.camera.phi = -20
        self.sky.position = self.camera
        static_colliders = self.scene_meshes + self.cubes
        self.player = PlayerThirdPerson(
            self.camera,
            [self.shaders[0], self.shaders[1]],
            self.player_mesh,
            player_start,
            visual_meshes=self.player_meshes,
            mesh_rotation_offset=(0.0, 0.0, 0.0),
            mesh_position_offset=(0.0, 0.0, 0.0),
            mesh_heading_offset=-90.0,
            colliders=static_colliders,
            terrain_bounds=self.terrain.get_world_bounds(),
            ground_height_fn=self.terrain_height,
            terrain_contains_fn=self.terrain_contains,
        )
        self.player.update(0.0)

    def _build_glb_model(self, file_path, position, target_size, normalize=True, rotation_degrees=(0.0, 0.0, 0.0)):
        submeshes = Mesh.load_glb_submeshes_prepared(
            file_path,
            invert_texcoord=False,
            st_pos=4,
            vertex_size=8,
            target_size=target_size,
            normalize=normalize,
            rotation_degrees=rotation_degrees,
        )
        material_cache = {}
        materials = []
        meshes = []
        for submesh in submeshes:
            material_index = submesh["material_index"]
            material = material_cache.get(material_index)
            if material is None:
                if material_index >= 0:
                    material_images = Mesh.load_glb_material_images(file_path, material_index=material_index)
                    material = Material.from_compatible_glb_images(material_images)
                else:
                    material = self.ground_texture
                material_cache[material_index] = material
                if material is not self.ground_texture:
                    materials.append(material)
            meshes.append(Mesh(self.shaders[0], material, numpy.array(position, dtype=numpy.float32).copy(), submesh["vertices"]))
        return meshes, materials

    def _get_primary_mesh(self, meshes):
        return max(meshes, key=lambda mesh: mesh.vertex_count)

    def _cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True
        glDeleteProgram(self.shaders[0])
        glDeleteProgram(self.shaders[1])
        self.crate_texture.destroy()
        self.ground_texture.destroy()
        [material.destroy() for material in self.terrain_materials]
        [material.destroy() for material in self.player_materials]
        [material.destroy() for material in self.scene_materials]
        [mesh.destroy() for mesh in self.terrain_meshes]
        [mesh.destroy() for mesh in self.player_meshes]
        [mesh.destroy() for primary in self.scene_meshes for mesh in [primary, *getattr(primary, "extra_meshes", [])]]
        [x.destroy() for x in self.cubes]
        [x.destroy() for x in self.lampada]
        self.sky.destroy()

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthMask(GL_FALSE)
        self.sky.draw()
        glDepthMask(GL_TRUE)
        [mesh.draw() for mesh in self.terrain_meshes]
        [mesh.draw() for mesh in self.player_meshes]
        [x.draw() for x in self.cubes]
        [
            mesh.draw()
            for primary in self.scene_meshes
            for mesh in [primary, *getattr(primary, "extra_meshes", [])]
        ]
        [x.draw() for x in self.lampada]

    def on_update(self, delta_time):
        [x.update() for x in self.light]
        self.player.update(delta_time)

        fps = 0 if delta_time <= 0 else round(1 / delta_time, 0)
        self.set_caption(str(fps))

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.close()
            return

        self.player.on_key_press(symbol)

    def on_key_release(self, symbol: int, modifiers: int):
        self.player.on_key_release(symbol)

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        self.player.on_mouse_motion(dx, dy)

    def on_close(self):
        self._cleanup()
        super().on_close()


def main():
    GameWindow()
    arcade.run()


if __name__ == "__main__":
    main()
