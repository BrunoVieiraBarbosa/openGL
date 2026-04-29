import os
import random

import arcade
import numpy
from OpenGL.GL import *

from core.core import *
from core.light import DirectionalLight, PointLight
from core.mesh import Mesh, MeshRGB
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

        self.texture = Material(
            os.path.join("textures", "teste.png"),
            os.path.join("textures", "teste_specular.png"),
            os.path.join("textures", "teste_specular.png"),
        )
        self.texture2 = Material(
            os.path.join("textures", "box.jpg"),
            os.path.join("textures", "box_specular.jpg"),
            os.path.join("textures", "box_specular.jpg"),
        )
        self.ground_texture = Material(
            os.path.join("textures", "block.png"),
            os.path.join("textures", "block_specular.png"),
            os.path.join("textures", "normal.jpg"),
        )

        ground_vertices = Mesh.create_plane(width=52.0, depth=38.0, uv_scale=3.5)
        self.terrain = Mesh(self.shaders[0], self.ground_texture, [10, 4, -0.02], ground_vertices)

        obj_scene = [
            ("nem.obj", [-2, -1, 0], self.texture),
            ("monkey.obj", [7, -3, 0], self.texture),
            ("IronMan.obj", [27, 1, 0], self.texture),
            ("untitled.obj", [1, 10, 0], self.texture),
            ("break_time.obj", [12, 12, 0], self.texture),
            ("cube.obj", [24, 10, 0], self.texture2),
        ]
        self.scene_meshes = []
        for file_name, position, material in obj_scene:
            vertices = Mesh.load_obj_prepared(
                os.path.join("obj", file_name),
                invert_texcoord=True,
                st_pos=4,
                vertex_size=8,
                target_size=1.8,
            )
            self.scene_meshes.append(Mesh(self.shaders[0], material, position, vertices))

        self.cubes = [
            Mesh(
                self.shaders[0],
                self.texture2,
                [random.randint(-6, 30), random.randint(-8, 16), 0.18],
                scale=0.18,
            )
            for _ in range(10)
        ]
        self.camera = CameraFirstPerson([10, -14, 1.7])
        self.camera.theta = 72
        self.camera.phi = -8
        self.sky.position = self.camera
        static_colliders = self.scene_meshes + self.cubes
        self.player = PlayerFirstPerson(
            self.camera,
            [self.shaders[0], self.shaders[1]],
            colliders=static_colliders,
            terrain_bounds=self.terrain.get_world_bounds(),
        )
        self.cubes_rotate = [random.randint(-5, 5) / 10 for _ in self.cubes]

    def _cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True
        glDeleteProgram(self.shaders[0])
        glDeleteProgram(self.shaders[1])
        self.texture.destroy()
        self.texture2.destroy()
        self.ground_texture.destroy()
        self.terrain.destroy()
        [x.destroy() for x in self.scene_meshes]
        [x.destroy() for x in self.cubes]
        [x.destroy() for x in self.lampada]
        self.sky.destroy()

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthMask(GL_FALSE)
        self.sky.draw()
        glDepthMask(GL_TRUE)
        self.terrain.draw()
        [x.draw() for x in self.cubes]
        [x.draw() for x in self.scene_meshes]
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
