from dataclasses import dataclass

import numpy
import pyrr
from OpenGL.GL import *

from core.core import ShadowMap


@dataclass
class ShadowSettings:
    enabled: bool = True
    map_size: int = 2048
    texture_unit: int = 3
    strength: float = 0.7
    debug_mode: int = 0
    disable_fog_debug: bool = False
    ortho_padding: float = 4.0
    focus_radius: float = 16.0
    focus_forward: float = 8.0
    focus_height: float = 16.0
    ground_offset: float = 3.0
    light_distance: float = 36.0
    depth_padding: float = 10.0


class SceneShadowController:
    def __init__(
        self,
        receiver_shaders,
        shadow_shader,
        shadow_skinned_shader,
        light_direction_provider,
        focus_position_provider,
        camera_heading_provider,
        shadow_mesh_iterator,
        window_size_provider,
        skinned_mesh_type,
        settings=None,
    ) -> None:
        self.receiver_shaders = list(receiver_shaders)
        self.shadow_shader = shadow_shader
        self.shadow_skinned_shader = shadow_skinned_shader
        self.light_direction_provider = light_direction_provider
        self.focus_position_provider = focus_position_provider
        self.camera_heading_provider = camera_heading_provider
        self.shadow_mesh_iterator = shadow_mesh_iterator
        self.window_size_provider = window_size_provider
        self.skinned_mesh_type = skinned_mesh_type
        self.settings = settings or ShadowSettings()
        self.shadow_map = ShadowMap(size=self.settings.map_size)
        self.light_space_matrix = pyrr.matrix44.create_identity(dtype=numpy.float32)
        self._configure_receivers()

    def _configure_receivers(self):
        for shader in self.receiver_shaders:
            glUseProgram(shader)
            shadow_map_location = glGetUniformLocation(shader, "shadowMap")
            if shadow_map_location >= 0:
                glUniform1i(shadow_map_location, int(self.settings.texture_unit))
            shadows_enabled_location = glGetUniformLocation(shader, "shadowsEnabled")
            if shadows_enabled_location >= 0:
                glUniform1i(shadows_enabled_location, 1 if self.settings.enabled else 0)
            shadow_debug_mode_location = glGetUniformLocation(shader, "shadowDebugMode")
            if shadow_debug_mode_location >= 0:
                glUniform1i(shadow_debug_mode_location, int(self.settings.debug_mode))
            shadow_strength_location = glGetUniformLocation(shader, "shadowStrength")
            if shadow_strength_location >= 0:
                glUniform1f(shadow_strength_location, float(self.settings.strength))
            shadow_disable_fog_location = glGetUniformLocation(shader, "shadowDisableFog")
            if shadow_disable_fog_location >= 0:
                glUniform1i(shadow_disable_fog_location, 1 if self.settings.disable_fog_debug else 0)

    def _compute_focus_center(self):
        focus_center = numpy.array(self.focus_position_provider(), dtype=numpy.float32).copy()
        theta = numpy.radians(float(self.camera_heading_provider()))
        focus_center[0] += numpy.cos(theta, dtype=numpy.float32) * self.settings.focus_forward
        focus_center[1] += numpy.sin(theta, dtype=numpy.float32) * self.settings.focus_forward
        focus_center[2] += self.settings.focus_height * 0.35
        return focus_center

    def _build_focus_corners(self, focus_center):
        min_z = float(focus_center[2] - self.settings.focus_height * 0.5 - self.settings.ground_offset)
        max_z = float(focus_center[2] + self.settings.focus_height * 0.5)
        x0 = float(focus_center[0] - self.settings.focus_radius)
        x1 = float(focus_center[0] + self.settings.focus_radius)
        y0 = float(focus_center[1] - self.settings.focus_radius)
        y1 = float(focus_center[1] + self.settings.focus_radius)
        return numpy.array(
            [
                [x0, y0, min_z, 1.0],
                [x0, y0, max_z, 1.0],
                [x0, y1, min_z, 1.0],
                [x0, y1, max_z, 1.0],
                [x1, y0, min_z, 1.0],
                [x1, y0, max_z, 1.0],
                [x1, y1, min_z, 1.0],
                [x1, y1, max_z, 1.0],
            ],
            dtype=numpy.float32,
        )

    def _compute_light_space_matrix(self):
        focus_center = self._compute_focus_center()
        light_direction = numpy.array(self.light_direction_provider(), dtype=numpy.float32)
        light_direction /= max(float(numpy.linalg.norm(light_direction)), 1e-6)
        light_position = focus_center - (light_direction * self.settings.light_distance)
        light_view = pyrr.matrix44.create_look_at(
            light_position,
            focus_center,
            numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32),
            dtype=numpy.float32,
        )
        focus_corners = self._build_focus_corners(focus_center)
        light_space_corners = numpy.array(
            [pyrr.matrix44.apply_to_vector(light_view, corner) for corner in focus_corners],
            dtype=numpy.float32,
        )
        light_bounds_min = light_space_corners.min(axis=0)
        light_bounds_max = light_space_corners.max(axis=0)

        light_projection = pyrr.matrix44.create_orthogonal_projection(
            float(light_bounds_min[0] - self.settings.ortho_padding),
            float(light_bounds_max[0] + self.settings.ortho_padding),
            float(light_bounds_min[1] - self.settings.ortho_padding),
            float(light_bounds_max[1] + self.settings.ortho_padding),
            float(max(-light_bounds_max[2] - self.settings.depth_padding, 0.1)),
            float(max(-light_bounds_min[2] + self.settings.depth_padding, 1.0)),
            dtype=numpy.float32,
        )
        return pyrr.matrix44.multiply(light_view, light_projection)

    def _apply_light_space_matrix(self):
        for shader in self.receiver_shaders:
            glUseProgram(shader)
            location = glGetUniformLocation(shader, "lightSpaceMatrix")
            if location >= 0:
                glUniformMatrix4fv(
                    location,
                    1,
                    GL_FALSE,
                    numpy.ascontiguousarray(self.light_space_matrix, dtype=numpy.float32),
                )

    def render_pass(self):
        if not self.settings.enabled:
            return

        self.light_space_matrix = self._compute_light_space_matrix()
        self._apply_light_space_matrix()
        self.shadow_map.begin()
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.5, 4.0)
        for mesh in self.shadow_mesh_iterator():
            if isinstance(mesh, self.skinned_mesh_type):
                mesh.draw_shadow(self.shadow_skinned_shader, self.light_space_matrix)
            else:
                mesh.draw_shadow(self.shadow_shader, self.light_space_matrix)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glCullFace(GL_BACK)
        glDisable(GL_CULL_FACE)
        viewport_width, viewport_height = self.window_size_provider()
        self.shadow_map.end(viewport_width, viewport_height)

    def bind_texture(self):
        if self.settings.enabled:
            self.shadow_map.bind(self.settings.texture_unit)

    def cycle_debug_mode(self):
        self.settings.debug_mode = (self.settings.debug_mode + 1) % 4
        self._configure_receivers()
        debug_labels = {
            0: "sombra final",
            1: "fator de sombra",
            2: "depth map",
            3: "cobertura da luz",
        }
        print(f"Shadow debug mode: {self.settings.debug_mode} ({debug_labels.get(self.settings.debug_mode, 'desconhecido')})")

    def destroy(self):
        self.shadow_map.destroy()
        glDeleteProgram(self.shadow_shader)
        glDeleteProgram(self.shadow_skinned_shader)
