from typing import Union

import arcade
import numpy
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from core.light import Light


class Shader:
    @staticmethod
    def create_shader(vertex_file_path, fragment_file_path):
        with open(vertex_file_path, "r") as file:
            vertex = file.readlines()

        with open(fragment_file_path, "r") as file:
            fragment = file.readlines()

        shader = compileProgram(
            compileShader(vertex, GL_VERTEX_SHADER),
            compileShader(fragment, GL_FRAGMENT_SHADER),
        )
        return shader


class Material:
    def __init__(
        self,
        file_path_diffuse: Union[str, arcade.Texture],
        file_path_specular: Union[str, arcade.Texture],
        file_path_normal: Union[str, arcade.Texture],
    ) -> None:
        def load_image(source):
            if isinstance(source, str):
                return arcade.load_texture(source).image.convert("RGBA")

            return source.image.convert("RGBA")

        def texture(texture_id, image):
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glGenerateMipmap(GL_TEXTURE_2D)
            image_w, image_h = image.size
            img_data = image.tobytes()
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                image_w,
                image_h,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                img_data,
            )

        image_diffuse = load_image(file_path_diffuse)
        image_specular = load_image(file_path_specular)
        image_normal = load_image(file_path_normal)

        self.diffuse_texture = glGenTextures(1)
        texture(self.diffuse_texture, image_diffuse)

        self.specular_texture = glGenTextures(1)
        texture(self.specular_texture, image_specular)

        self.normal_texture = glGenTextures(1)
        texture(self.normal_texture, image_normal)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.diffuse_texture)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.specular_texture)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.normal_texture)

    def destroy(self):
        glDeleteTextures(2, (self.diffuse_texture, self.specular_texture, self.normal_texture))


class CameraFirstPerson:
    def __init__(self, position) -> None:
        self.position = numpy.array(position, dtype=numpy.float32)
        self.forward = numpy.array([0, 0, 0], dtype=numpy.float32)
        self.theta = 0
        self.phi = 0
        self.move_speed = 1
        self.global_up = numpy.array([0, 0, 1], dtype=numpy.float32)

    def move(self, direction, amount):
        walk_direction = numpy.radians((direction + self.theta) % 360)
        self.position[0] += amount * self.move_speed * numpy.cos(walk_direction, dtype=numpy.float32)
        self.position[1] += amount * self.move_speed * numpy.sin(walk_direction, dtype=numpy.float32)

    def increment_direction(self, horizontal, vertical):
        self.theta = (self.theta + horizontal) % 360
        self.phi = min(max((self.phi + vertical), -89), 89)

    def update(self, shaders):
        theta, phi = numpy.radians(self.theta), numpy.radians(self.phi)
        camera_cos = numpy.cos(theta, dtype=numpy.float32)
        camera_sin = numpy.sin(theta, dtype=numpy.float32)
        camera_cos2 = numpy.cos(phi, dtype=numpy.float32)
        camera_sin2 = numpy.sin(phi, dtype=numpy.float32)

        self.forward[0] = camera_cos * camera_cos2
        self.forward[1] = camera_sin * camera_cos2
        self.forward[2] = camera_sin2

        right = pyrr.vector3.cross(self.global_up, self.forward)
        up = pyrr.vector3.cross(self.forward, right)
        look_at_matrix = pyrr.matrix44.create_look_at(
            self.position,
            self.position + self.forward,
            up,
            dtype=numpy.float32,
        )

        for shader in shaders:
            glUseProgram(shader)
            glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, look_at_matrix)
            glUniform3fv(glGetUniformLocation(shader, "cameraPos"), 1, self.position)


class App(arcade.Window):
    def __init__(self, size, ambient_color=(0.1, 0.1, 0.1, 1), title="openGL") -> None:
        super().__init__(
            width=size[0],
            height=size[1],
            title=title,
            update_rate=1 / 60,
            draw_rate=1 / 60,
            resizable=False,
        )
        self.window_size = size
        self.ambient_color = ambient_color
        self.shaders = []
        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.ambient_color)

    def add_shader(self, name, shader):
        self.shaders.append(shader)

    def _apply_projection(self):
        projection_transform = pyrr.matrix44.create_perspective_projection(
            45,
            self.window_size[0] / self.window_size[1],
            0.1,
            100,
            numpy.float32,
        )

        glUseProgram(self.shaders[0])
        glUniformMatrix4fv(
            glGetUniformLocation(self.shaders[0], "projection"),
            1,
            GL_FALSE,
            projection_transform,
        )

        glUseProgram(self.shaders[1])
        glUniformMatrix4fv(
            glGetUniformLocation(self.shaders[1], "projection"),
            1,
            GL_FALSE,
            projection_transform,
        )

    def start_(self):
        glUseProgram(self.shaders[0])
        glUniform3fv(
            glGetUniformLocation(self.shaders[0], "ambient"),
            1,
            numpy.array(self.ambient_color[:3], dtype=numpy.float32),
        )
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.diffuse"), 0)
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.specular"), 1)
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.normal"), 2)

        self._apply_projection()
        Light.reset_lights([self.shaders[0]])

    def on_resize(self, width: int, height: int):
        super().on_resize(width, height)
        self.window_size = (width, max(height, 1))
        glViewport(0, 0, width, height)
        if len(self.shaders) >= 2:
            self._apply_projection()
