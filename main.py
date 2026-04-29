import os
import random

import arcade
from OpenGL.GL import *

from core.core import *
from core.light import FlashLight, PointLight
from core.mesh import Mesh, MeshRGB
from core.utils import *


class GameWindow(App):
    def __init__(self):
        size = (1280, 720)
        super().__init__(size)
        self.set_mouse_visible(False)
        self.set_exclusive_mouse(True)
        self.cleaned_up = False
        self._setup_scene()

    def _setup_scene(self):
        self.add_shader("first", Shader.create_shader("shaders/vertex.c", "shaders/fragment.c"))
        self.add_shader("simple", Shader.create_shader("shaders/vertex_rgb.c", "shaders/fragment_rgb.c"))

        self.start_()

        self.light = [
            PointLight([self.shaders[0], self.shaders[1]], [15, 14, 15], [1, 1, 1], 8, 0, [True, False]),
            FlashLight(
                [self.shaders[0], self.shaders[1]],
                [15, 0, 15],
                [-0.2, -1.0, -0.3],
                [0.8, 0.8, 0.8],
                8,
                1,
                15,
                20,
                [True, False],
            ),
            FlashLight(
                [self.shaders[0], self.shaders[1]],
                [15, 7, 15],
                [1, 0.2, 0.1],
                [0.8, 0.8, 0.8],
                10,
                1,
                15,
                20,
                [True, False],
            ),
        ]

        self.lampada = [
            MeshRGB(self.shaders[1], self.light[0], color=[1, 1, 1]),
            MeshRGB(self.shaders[1], self.light[1], color=[0, 1, 0]),
            MeshRGB(self.shaders[1], self.light[2], color=[0, 1, 1]),
        ]

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

        vertices = Mesh.load_obj("obj/nem.obj")
        vertices = Mesh.invert_s_or_t(vertices, 4, 8)
        self.monkey = [Mesh(self.shaders[0], self.texture, [2, 7, 1], vertices)]

        self.cubes = [
            Mesh(
                self.shaders[0],
                self.texture2,
                [random.randint(x, x * 2), random.randint(y, y * 2), 0],
            )
            for y in range(10)
            for x in range(10)
        ]
        self.camera = CameraFirstPerson([-10, 7, 2])
        self.player = PlayerFirstPerson(self.camera, [self.shaders[0], self.shaders[1]])
        self.cubes_rotate = [random.randint(-5, 5) / 10 for _ in self.cubes]

    def _cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True
        glDeleteProgram(self.shaders[0])
        glDeleteProgram(self.shaders[1])
        self.texture.destroy()
        self.texture2.destroy()
        [x.destroy() for x in self.monkey]
        [x.destroy() for x in self.cubes]
        [x.destroy() for x in self.lampada]

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        [x.draw() for x in self.cubes]
        [x.draw() for x in self.monkey]
        [x.draw() for x in self.lampada]

    def on_update(self, delta_time):
        self.light[0].position[0] -= 0.02
        self.light[0].position[1] -= 0.02
        self.light[0].position[2] -= 0.02

        self.light[1].position[0] -= 0.02
        self.light[1].position[1] += 0.02
        self.light[1].position[2] -= 0.02

        self.light[2].position[0] -= 0.02
        self.light[2].position[2] -= 0.02

        [x.update() for x in self.light]
        [x.rotate_xyz(0.5) for x in self.monkey]
        [x.rotate_xyz(self.cubes_rotate[i]) for i, x in enumerate(self.cubes)]
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
