from typing import Union

import arcade
import numpy
import pyrr
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from core.light import Light


def get_uniform_location(shader, name):
    cache = getattr(get_uniform_location, "_cache", None)
    if cache is None:
        cache = {}
        get_uniform_location._cache = cache

    shader_cache = cache.setdefault(shader, {})
    location = shader_cache.get(name)
    if location is None:
        location = glGetUniformLocation(shader, name)
        shader_cache[name] = location
    return location


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
    @staticmethod
    def _solid_image(color):
        return Image.new("RGBA", (1, 1), color)

    @classmethod
    def from_compatible_glb_images(cls, glb_images):
        diffuse = glb_images.get("diffuse") or cls._solid_image((255, 255, 255, 255))
        specular = glb_images.get("specular") or cls._solid_image((64, 64, 64, 255))
        normal = glb_images.get("normal") or cls._solid_image((128, 128, 255, 255))
        return cls(diffuse, specular, normal)

    def __init__(
        self,
        file_path_diffuse: Union[str, arcade.Texture, Image.Image],
        file_path_specular: Union[str, arcade.Texture, Image.Image],
        file_path_normal: Union[str, arcade.Texture, Image.Image],
    ) -> None:
        def load_image(source):
            if isinstance(source, str):
                return arcade.load_texture(source).image.convert("RGBA")
            if isinstance(source, Image.Image):
                return source.convert("RGBA")

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
        glDeleteTextures(3, (self.diffuse_texture, self.specular_texture, self.normal_texture))


class Camera:
    def __init__(self, position) -> None:
        self.position = numpy.array(position, dtype=numpy.float32)
        self.forward = numpy.array([0, 0, 0], dtype=numpy.float32)
        self.move_speed = 1
        self.global_up = numpy.array([0, 0, 1], dtype=numpy.float32)
        self._last_view_signature = None

    def apply_view(self, shaders, target):
        target = numpy.array(target, dtype=numpy.float32)
        view_signature = (
            float(self.position[0]),
            float(self.position[1]),
            float(self.position[2]),
            float(target[0]),
            float(target[1]),
            float(target[2]),
        )
        if self._last_view_signature == view_signature:
            return
        self._last_view_signature = view_signature
        self.forward = target - self.position
        forward_norm = max(float(numpy.linalg.norm(self.forward)), 1e-6)
        self.forward /= forward_norm
        right = pyrr.vector3.cross(self.global_up, self.forward)
        right_norm = max(float(numpy.linalg.norm(right)), 1e-6)
        right /= right_norm
        up = pyrr.vector3.cross(self.forward, right)
        look_at_matrix = pyrr.matrix44.create_look_at(
            self.position,
            target,
            up,
            dtype=numpy.float32,
        )

        for shader in shaders:
            glUseProgram(shader)
            view_location = get_uniform_location(shader, "view")
            if view_location >= 0:
                glUniformMatrix4fv(view_location, 1, GL_FALSE, look_at_matrix)

            camera_location = get_uniform_location(shader, "cameraPos")
            if camera_location >= 0:
                glUniform3fv(camera_location, 1, self.position)


class CameraFirstPerson(Camera):
    def __init__(self, position) -> None:
        super().__init__(position)
        self.theta = 0
        self.phi = 0

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
        look_target = self.position + numpy.array(
            [
                camera_cos * camera_cos2,
                camera_sin * camera_cos2,
                camera_sin2,
            ],
            dtype=numpy.float32,
        )
        self.apply_view(shaders, look_target)


class CameraThirdPerson(Camera):
    def __init__(self, focus_position, distance=5.8, height=1.6) -> None:
        super().__init__(focus_position)
        self.focus_position = numpy.array(focus_position, dtype=numpy.float32)
        self.theta = 70
        self.phi = -22
        self.distance = distance
        self.height = height
        self.min_distance = 2.4
        self.min_phi = -55
        self.max_phi = 20

    def increment_direction(self, horizontal, vertical):
        self.theta = (self.theta + horizontal) % 360
        self.phi = min(max(self.phi + vertical, self.min_phi), self.max_phi)

    def update_focus(self, focus_position):
        self.focus_position = numpy.array(focus_position, dtype=numpy.float32)

    def update(self, shaders):
        theta = numpy.radians(self.theta)
        phi = numpy.radians(self.phi)
        orbit_forward = numpy.array(
            [
                numpy.cos(theta, dtype=numpy.float32) * numpy.cos(phi, dtype=numpy.float32),
                numpy.sin(theta, dtype=numpy.float32) * numpy.cos(phi, dtype=numpy.float32),
                numpy.sin(phi, dtype=numpy.float32),
            ],
            dtype=numpy.float32,
        )
        target = self.focus_position + numpy.array([0.0, 0.0, self.height], dtype=numpy.float32)
        desired_position = target - orbit_forward * self.distance
        self.position = desired_position.astype(numpy.float32)
        self.apply_view(shaders, target)


class App(arcade.Window):
    def __init__(self, size, ambient_color=(0.1, 0.1, 0.1, 1), title="openGL") -> None:
        super().__init__(
            width=size[0],
            height=size[1],
            title=title,
            update_rate=1 / 5000,
            draw_rate=1 / 5000,
            fixed_rate=1 / 5000,
            fixed_frame_cap=None,
            vsync=False,
            resizable=False,
        )
        self.window_size = size
        self.ambient_color = ambient_color
        self.fog_color = numpy.array(self.ambient_color[:3], dtype=numpy.float32)
        self.fog_near = 22.0
        self.fog_far = 58.0
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

        for shader in self.shaders:
            glUseProgram(shader)
            location = get_uniform_location(shader, "projection")
            if location >= 0:
                glUniformMatrix4fv(location, 1, GL_FALSE, projection_transform)

    def start_(self):
        for shader in self.shaders:
            glUseProgram(shader)

            ambient_location = get_uniform_location(shader, "ambient")
            if ambient_location >= 0:
                glUniform3fv(
                    ambient_location,
                    1,
                    numpy.array(self.ambient_color[:3], dtype=numpy.float32),
                )

            fog_color_location = get_uniform_location(shader, "fogColor")
            if fog_color_location >= 0:
                glUniform3fv(fog_color_location, 1, self.fog_color)

            fog_near_location = get_uniform_location(shader, "fogNear")
            if fog_near_location >= 0:
                glUniform1f(fog_near_location, self.fog_near)

            fog_far_location = get_uniform_location(shader, "fogFar")
            if fog_far_location >= 0:
                glUniform1f(fog_far_location, self.fog_far)

            diffuse_location = get_uniform_location(shader, "material.diffuse")
            if diffuse_location >= 0:
                glUniform1i(diffuse_location, 0)

            specular_location = get_uniform_location(shader, "material.specular")
            if specular_location >= 0:
                glUniform1i(specular_location, 1)

            normal_location = get_uniform_location(shader, "material.normal")
            if normal_location >= 0:
                glUniform1i(normal_location, 2)

        self._apply_projection()
        Light.reset_lights(self.shaders)

    def on_resize(self, width: int, height: int):
        super().on_resize(width, height)
        self.window_size = (width, max(height, 1))
        glViewport(0, 0, width, height)
        if len(self.shaders) >= 2:
            self._apply_projection()
