import os
import random

import arcade
import numpy
from OpenGL.GL import *

from core.core import *
from core.light import DirectionalLight, PointLight
from core.mesh import Mesh, MeshRGB, SkinnedAnimator, SkinnedMesh, SkinnedModel, TerrainGridSampler
from core.shadows import SceneShadowController, ShadowSettings
from core.utils import *

# Diagnostic options for isolating scale vs skinning issues in Player.glb:
# - "static": same pipeline as the scene assets
# - "skinned_bind_pose": skinned mesh frozen in bind/rest pose
# - "skinned_walk": skinned mesh with animation updates enabled
PLAYER_RENDER_MODE = "skinned_walk"
PLAYER_TARGET_SIZE = 1.9
PLAYER_NORMALIZE = True
PLAYER_ALLOW_STATIC_FALLBACK = False
PLAYER_ALWAYS_PLAY_WALK = False
SHADOW_SETTINGS = ShadowSettings(
    enabled=True,
    map_size=2048,
    strength=0.7,
    debug_mode=0,
    disable_fog_debug=False,
    ortho_padding=4.0,
    focus_radius=16.0,
    focus_forward=8.0,
    focus_height=16.0,
    ground_offset=3.0,
    light_distance=36.0,
    depth_padding=10.0,
)


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
        self.player_fallback_submeshes = []
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
        self.add_shader("skinned", Shader.create_shader("shaders/vertex_skinned.c", "shaders/fragment.c"))
        self.start_()

        self.light = [
            DirectionalLight(
                [self.shaders[0], self.shaders[1], self.shaders[2]],
                [-0.4, -0.8, -1.0],
                [1.0, 0.95, 0.86],
                30,
                0,
                [True, False, True],
            ),
            PointLight([self.shaders[0], self.shaders[1], self.shaders[2]], [6, -2, 8], [1.0, 0.96, 0.9], 32, 1, [True, False, True]),
            PointLight(
                [self.shaders[0], self.shaders[1], self.shaders[2]],
                [20, 6, 10],
                [0.85, 0.9, 1.0],
                32,
                2,
                [True, False, True],
            ),
        ]
        self.shadow_controller = SceneShadowController(
            receiver_shaders=[self.shaders[0], self.shaders[2]],
            shadow_shader=Shader.create_shader("shaders/shadow_vertex.c", "shaders/shadow_fragment.c"),
            shadow_skinned_shader=Shader.create_shader("shaders/shadow_vertex_skinned.c", "shaders/shadow_fragment.c"),
            light_direction_provider=lambda: self.light[0].direction,
            focus_position_provider=lambda: self.player.position if hasattr(self, "player") else numpy.array([10.0, 0.0, 0.0], dtype=numpy.float32),
            camera_heading_provider=lambda: self.camera.theta if hasattr(self, "camera") else 0.0,
            shadow_mesh_iterator=self._iter_shadow_meshes,
            window_size_provider=lambda: self.window_size,
            skinned_mesh_type=SkinnedMesh,
            settings=SHADOW_SETTINGS,
        )

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
        self.player_meshes, self.player_materials, player_visual = self._build_player_visual(
            os.path.join("obj", "Player.glb"),
            player_start,
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
            self.shaders,
            self.player_mesh,
            player_start,
            visual_meshes=self.player_meshes,
            animated_visual=player_visual,
            mesh_rotation_offset=(0.0, 0.0, 0.0),
            mesh_position_offset=(0.0, 0.0, 0.0),
            mesh_heading_offset=-90.0,
            colliders=static_colliders,
            terrain_bounds=self.terrain.get_world_bounds(),
            ground_height_fn=self.terrain_height,
            terrain_contains_fn=self.terrain_contains,
            always_play_walk=PLAYER_ALWAYS_PLAY_WALK,
        )
        self.player.update(0.1 if PLAYER_ALWAYS_PLAY_WALK and PLAYER_RENDER_MODE == "skinned_walk" else 0.0)

    def _iter_shadow_meshes(self):
        for mesh in self.player_meshes:
            yield mesh
        for cube in self.cubes:
            yield cube
        for primary in self.scene_meshes:
            yield primary
            for extra_mesh in getattr(primary, "extra_meshes", []):
                yield extra_mesh

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

    def _build_player_visual(self, file_path, position):
        if PLAYER_RENDER_MODE == "static":
            meshes, materials = self._build_glb_model(
                file_path,
                position,
                target_size=PLAYER_TARGET_SIZE,
                normalize=PLAYER_NORMALIZE,
            )
            return meshes, materials, None

        if PLAYER_RENDER_MODE == "skinned_bind_pose":
            visual = self._build_skinned_glb_model(
                file_path,
                position,
                target_size=PLAYER_TARGET_SIZE,
                normalize=PLAYER_NORMALIZE,
            )
            return visual.meshes, visual.materials, None

        if PLAYER_RENDER_MODE == "skinned_walk":
            visual = self._build_skinned_glb_model(
                file_path,
                position,
                target_size=PLAYER_TARGET_SIZE,
                normalize=PLAYER_NORMALIZE,
            )
            return visual.meshes, visual.materials, visual

        raise ValueError(f"Unsupported PLAYER_RENDER_MODE: {PLAYER_RENDER_MODE}")

    def _build_skinned_glb_model(self, file_path, position, target_size, normalize):
        self.player_fallback_submeshes = []
        model_data = Mesh.load_glb_skinned_data(
            file_path,
            invert_texcoord=False,
            target_size=target_size,
            normalize=normalize,
        )
        static_submeshes = {
            submesh["name"]: submesh
            for submesh in Mesh.load_glb_submeshes_prepared(
                file_path,
                invert_texcoord=False,
                st_pos=4,
                vertex_size=8,
                target_size=target_size,
                normalize=normalize,
            )
        }
        animator = SkinnedAnimator(
            model_data["node_transforms"],
            model_data["node_parents"],
            model_data["animations"],
            model_data["skins"],
        )

        material_cache = {}
        materials = []
        meshes = []
        model_transform = model_data["transform"]
        for submesh in model_data["submeshes"]:
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

            static_submesh = static_submeshes.get(submesh["name"])
            correction_matrix, correction_diff = self._compute_skinned_correction_matrix(
                submesh,
                animator,
                static_submesh["vertices"] if static_submesh else None,
                "standard",
            )

            mesh = SkinnedMesh(
                self.shaders[2],
                material,
                numpy.array(position, dtype=numpy.float32).copy(),
                submesh["vertices"],
                animator,
                submesh["skin_index"],
                submesh["mesh_bind_matrix"],
                skinning_mode="standard",
                post_skinning_transform=correction_matrix,
                origin_offset=numpy.zeros(3, dtype=numpy.float32),
                scale=1.0,
            )
            needs_fallback = (
                static_submesh is not None
                and self._skinned_mesh_needs_static_fallback(mesh, static_submesh["vertices"])
            )
            if needs_fallback:
                self.player_fallback_submeshes.append(
                    f"{submesh['name']} (bind diff corrigido {correction_diff:.4f})"
                )
                if PLAYER_ALLOW_STATIC_FALLBACK:
                    mesh.destroy()
                    mesh = Mesh(
                        self.shaders[0],
                        material,
                        numpy.array(position, dtype=numpy.float32).copy(),
                        static_submesh["vertices"],
                    )
            meshes.append(mesh)

        if self.player_fallback_submeshes:
            if PLAYER_ALLOW_STATIC_FALLBACK:
                print("Player fallback estatico por submalha:")
            else:
                print("Player submalhas com bind divergente, mantendo skinning para animacao:")
            for submesh_name in self.player_fallback_submeshes:
                print(f" - {submesh_name}")
        else:
            print("Player skinned sem fallback de submalha.")

        return SkinnedModel(meshes, materials, animator)

    def _skinned_mesh_needs_static_fallback(self, skinned_mesh, static_vertices, max_allowed_diff=0.45):
        static_positions = numpy.array(static_vertices, dtype=numpy.float32).reshape(-1, 11)[:, :3]
        if hasattr(skinned_mesh, "_apply_skinning_to_positions"):
            skinned_positions = skinned_mesh._apply_skinning_to_positions(skinned_mesh.bone_matrices)[:, :3]
        else:
            skinned_positions = skinned_mesh.vertices[:, :3]
        if len(static_positions) != len(skinned_positions):
            return True
        return float(numpy.max(numpy.abs(skinned_positions - static_positions))) > max_allowed_diff

    def _compute_skinned_correction_matrix(self, submesh, animator, static_vertices, skinning_mode):
        if static_vertices is None:
            return numpy.identity(4, dtype=numpy.float32), float("inf")

        static_positions = numpy.array(static_vertices, dtype=numpy.float32).reshape(-1, 11)[:, :3]
        bind_positions = self._evaluate_skinned_bind_positions(submesh, animator, skinning_mode)
        if len(bind_positions) != len(static_positions):
            return numpy.identity(4, dtype=numpy.float32), float("inf")

        source_augmented = numpy.concatenate(
            [bind_positions, numpy.ones((len(bind_positions), 1), dtype=numpy.float32)],
            axis=1,
        )
        affine_matrix, *_ = numpy.linalg.lstsq(source_augmented, static_positions, rcond=None)
        fitted_positions = source_augmented @ affine_matrix
        max_diff = float(numpy.max(numpy.abs(fitted_positions - static_positions)))

        correction_matrix = numpy.identity(4, dtype=numpy.float32)
        correction_matrix[:3, :3] = affine_matrix[:3, :].T
        correction_matrix[:3, 3] = affine_matrix[3, :]
        print(f"{submesh['name']} correction diff: {max_diff:.4f}")
        return correction_matrix, max_diff

    def _evaluate_skinned_bind_positions(self, submesh, animator, skinning_mode):
        base_vertices = numpy.array(submesh["vertices"], dtype=numpy.float32).reshape(-1, 19)
        position4 = numpy.concatenate(
            [base_vertices[:, :3], numpy.ones((len(base_vertices), 1), dtype=numpy.float32)],
            axis=1,
        )
        joint_ids = base_vertices[:, 11:15].astype(numpy.int32)
        joint_weights = base_vertices[:, 15:19].astype(numpy.float32)
        bone_matrices = animator.get_skin_matrices(
            submesh["skin_index"],
            submesh["mesh_bind_matrix"],
            mode=skinning_mode,
        )
        selected_bones = bone_matrices[joint_ids]
        skin_matrices = (selected_bones * joint_weights[:, :, numpy.newaxis, numpy.newaxis]).sum(axis=1)
        skinned_positions = numpy.einsum("nij,nj->ni", skin_matrices, position4)
        bind_positions = numpy.einsum(
            "ij,nj->ni",
            numpy.array(submesh["mesh_bind_matrix"], dtype=numpy.float32),
            skinned_positions,
        )
        return bind_positions[:, :3]

    def _cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True
        [glDeleteProgram(shader) for shader in self.shaders]
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
        self.shadow_controller.destroy()

    def on_draw(self):
        self.clear()
        self.shadow_controller.render_pass()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shadow_controller.bind_texture()
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
        if symbol == arcade.key.F3:
            self.shadow_controller.cycle_debug_mode()
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
