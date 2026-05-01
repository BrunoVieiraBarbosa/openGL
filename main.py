import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import arcade
import numpy
from OpenGL.GL import *

from core.core import *
from core.light import DirectionalLight, PointLight
from core.mesh import Mesh, MeshRGB, SkinnedAnimator, SkinnedMesh, SkinnedModel, TerrainGridSampler
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
ENABLE_FRAME_PROFILER = False
FRAME_PROFILER_PRINT_INTERVAL = 2.0
ENABLE_NPC_VISION_DEBUG = False
NPC_PATROLS = (
    ((10.0, -10.0), (20.0, -10.5)),
    ((7.0, -9), (7.0, -1)),
    ((15.5, -4.0), (22.0, -3.5)),
    ((3.5, -12.0), (5.5, -6.5)),
    ((11.5, 1.0), (17.0, 0.5)),
    ((1.5, -2.0), (4.5, 2.0)),
    ((18.0, -11.5), (24.0, -9.5)),
    ((-0.5, -7.0), (3.0, -10.5)),
    ((13.0, 4.5), (18.5, 6.0)),
    ((6.0, 3.0), (9.5, 7.0)),
    ((20.5, -0.5), (24.5, 3.5)),
    ((2.0, 5.5), (5.5, 9.0)),
    ((9.0, -13.5), (14.0, -15.0)),
    ((16.0, 8.0), (22.0, 8.5)),
    ((-1.0, 1.0), (2.5, 4.0)),
    ((23.0, -6.0), (27.0, -2.0)),
)
SENTRY_NPC_POSTS = (
    ((8.0, 11.0), 180.0),
    ((21.5, 10.0), 215.0),
    ((-0.5, 8.5), 330.0),
)
SCENE_GLB_FILES = (
    ("IronMan.glb", [27, 1, 0], 1.8, 0.38, 0.1),
    ("break_time.glb", [12, 12, 0], 1.8, 0.44, 0.1),
)


class FrameProfiler:
    def __init__(self, enabled=True, print_interval=2.0):
        self.enabled = bool(enabled)
        self.print_interval = float(print_interval)
        self._started_at = time.perf_counter()
        self._last_print = self._started_at
        self._section_totals = {}
        self._section_counts = {}

    def section(self, name):
        if not self.enabled:
            return _NullProfileSection()
        return _ProfileSection(self, name)

    def add_sample(self, name, duration):
        self._section_totals[name] = self._section_totals.get(name, 0.0) + float(duration)
        self._section_counts[name] = self._section_counts.get(name, 0) + 1

    def maybe_print(self):
        if not self.enabled:
            return

        now = time.perf_counter()
        elapsed = now - self._last_print
        if elapsed < self.print_interval or not self._section_totals:
            return

        ordered_sections = sorted(
            self._section_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        summary_parts = []
        for name, total in ordered_sections:
            count = max(self._section_counts.get(name, 0), 1)
            average_ms = (total / count) * 1000.0
            summary_parts.append(f"{name}={average_ms:.2f}ms")

        print(f"[frame-profiler] {', '.join(summary_parts)}")
        self._section_totals.clear()
        self._section_counts.clear()
        self._last_print = now


class _NullProfileSection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ProfileSection:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler.add_sample(self.name, time.perf_counter() - self.started_at)
        return False


def _preload_static_glb_asset(file_path, target_size, normalize, rotation_degrees=(0.0, 0.0, 0.0), include_positions=False):
    if include_positions:
        Mesh.extract_glb_unique_positions(file_path)

    submeshes = Mesh.load_glb_submeshes_prepared(
        file_path,
        invert_texcoord=False,
        st_pos=4,
        vertex_size=8,
        target_size=target_size,
        normalize=normalize,
        rotation_degrees=rotation_degrees,
    )
    material_indices = sorted(
        {
            int(submesh["material_index"])
            for submesh in submeshes
            if int(submesh["material_index"]) >= 0
        }
    )
    for material_index in material_indices:
        Mesh.load_glb_material_images(file_path, material_index=material_index)
    return file_path


def _preload_player_asset(file_path, target_size, normalize, render_mode):
    material_indices = set()

    static_submeshes = Mesh.load_glb_submeshes_prepared(
        file_path,
        invert_texcoord=False,
        st_pos=4,
        vertex_size=8,
        target_size=target_size,
        normalize=normalize,
        rotation_degrees=(0.0, 0.0, 0.0),
    )
    material_indices.update(
        int(submesh["material_index"])
        for submesh in static_submeshes
        if int(submesh["material_index"]) >= 0
    )

    if render_mode != "static":
        skinned_data = Mesh.load_glb_skinned_data(
            file_path,
            invert_texcoord=False,
            target_size=target_size,
            normalize=normalize,
        )
        material_indices.update(
            int(submesh["material_index"])
            for submesh in skinned_data["submeshes"]
            if int(submesh["material_index"]) >= 0
        )

    for material_index in sorted(material_indices):
        Mesh.load_glb_material_images(file_path, material_index=material_index)
    return file_path


class GameWindow(App):
    def __init__(self):
        size = (1280, 720)
        super().__init__(size, ambient_color=(0.62, 0.67, 0.73, 1.0))
        self.fog_color = numpy.array([0.78, 0.79, 0.74], dtype=numpy.float32)
        self.fog_near = 42.0
        self.fog_far = 140.0
        self.set_mouse_visible(False)
        self.set_exclusive_mouse(True)
        self.cleaned_up = False
        self.player_fallback_submeshes = []
        self._glb_material_cache = {}
        self._owned_materials = []
        self._scene_draw_meshes = []
        self.profiler = FrameProfiler(
            enabled=ENABLE_FRAME_PROFILER,
            print_interval=FRAME_PROFILER_PRINT_INTERVAL,
        )
        self._setup_scene()

    def _setup_scene(self):
        self.terrain_origin = numpy.array([10.0, 4.0, -0.02], dtype=numpy.float32)
        self.terrain_glb_path = os.path.join("obj", "terrain_main.glb")
        self._preload_scene_assets_parallel()
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
        self.dynamic_lights = [light for light in self.light if light.dynamic]
        for light in self.light:
            light.update()

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

        self.scene_meshes = []
        self.scene_materials = []
        for file_name, position, target_size, collider_radius_scale, collider_radius_padding in SCENE_GLB_FILES:
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
            self._scene_draw_meshes.extend([primary_mesh, *primary_mesh.extra_meshes])

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
        self.npcs = []
        self.npc_draw_meshes = []
        self.npc_vision_gizmos = []
        npc_index = 0
        for patrol_points in NPC_PATROLS:
            grounded_waypoints = [
                numpy.array([point[0], point[1], self.terrain_height(point[0], point[1])], dtype=numpy.float32)
                for point in patrol_points
            ]
            self._add_npc(
                PatrolNPC,
                grounded_waypoints[0],
                waypoints=grounded_waypoints,
                move_speed=1.35 + ((npc_index % 5) * 0.12),
                turn_speed=165.0 + (npc_index * 25.0),
                wait_time=0.8 + (npc_index * 0.35),
                look_target_radius=4.2 + (npc_index * 0.35),
                investigate_speed=1.75 + ((npc_index % 5) * 0.14),
                investigate_radius=5.6 + (npc_index * 0.4),
                investigate_duration=2.1 + (npc_index * 0.35),
                investigate_stop_radius=1.05 + (npc_index * 0.08),
                vision_angle_deg=max(42.0, 78.0 - (npc_index * 2.5)),
                debug_color=(1.0, 0.35 + ((npc_index % 4) * 0.08), 0.18),
            )
            npc_index += 1

        for post_position, facing_yaw in SENTRY_NPC_POSTS:
            grounded_position = numpy.array(
                [post_position[0], post_position[1], self.terrain_height(post_position[0], post_position[1])],
                dtype=numpy.float32,
            )
            self._add_npc(
                SentryNPC,
                grounded_position,
                home_position=grounded_position.copy(),
                facing_yaw=facing_yaw,
                move_speed=1.2,
                turn_speed=190.0,
                wait_time=0.0,
                look_target_radius=5.2,
                investigate_speed=1.95,
                investigate_radius=6.8,
                investigate_duration=2.8,
                investigate_stop_radius=1.15,
                vision_angle_deg=58.0,
                scan_half_angle=40.0,
                scan_speed=52.0,
                return_speed=1.45,
                debug_color=(0.35, 0.95, 0.5),
            )

        static_colliders = self.scene_meshes + self.cubes + [npc.primary_mesh for npc in self.npcs]
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
            profiler=self.profiler,
        )
        for npc in self.npcs:
            npc.look_target = self.player
        self.player.update(0.1 if PLAYER_ALWAYS_PLAY_WALK and PLAYER_RENDER_MODE == "skinned_walk" else 0.0)

    def _add_npc(self, npc_cls, spawn_position, debug_color=(1.0, 0.4, 0.2), **npc_kwargs):
        npc_meshes, _npc_materials, npc_visual = self._build_player_visual(
            os.path.join("obj", "Player.glb"),
            spawn_position,
        )
        npc_primary_mesh = self._get_primary_mesh(npc_meshes)
        npc_primary_mesh.set_collider(mode="circle", radius_scale=0.55, radius_padding=0.06, height_padding=0.12)
        npc = npc_cls(
            npc_primary_mesh,
            spawn_position,
            visual_meshes=npc_meshes,
            animated_visual=npc_visual,
            ground_height_fn=self.terrain_height,
            mesh_rotation_offset=(0.0, 0.0, 0.0),
            mesh_position_offset=(0.0, 0.0, 0.0),
            mesh_heading_offset=-90.0,
            look_target=None,
            perception_rotation_offset_deg=0.0,
            profiler=self.profiler,
            **npc_kwargs,
        )
        self.npcs.append(npc)
        self.npc_draw_meshes.extend(npc_meshes)
        if ENABLE_NPC_VISION_DEBUG:
            self.npc_vision_gizmos.append(
                MeshRGB(
                    self.shaders[1],
                    spawn_position.copy(),
                    vertices=MeshRGB.create_sector(
                        radius=float(npc.investigate_radius),
                        angle_degrees=float(npc.vision_angle_deg),
                        segments=18,
                        color=debug_color,
                        z_offset=0.03,
                    ),
                    scale=1.0,
                )
            )

    def _preload_scene_assets_parallel(self):
        cpu_count = os.cpu_count() or 1
        if cpu_count < 2:
            return

        preload_jobs = [
            (_preload_static_glb_asset, (self.terrain_glb_path, 52.0, False, (0.0, 0.0, 0.0), True)),
        ]
        preload_jobs.extend(
            (
                _preload_static_glb_asset,
                (os.path.join("obj", file_name), target_size, True, (0.0, 0.0, 0.0), False),
            )
            for file_name, _position, target_size, _radius_scale, _radius_padding in SCENE_GLB_FILES
        )
        preload_jobs.append(
            (
                _preload_player_asset,
                (os.path.join("obj", "Player.glb"), PLAYER_TARGET_SIZE, PLAYER_NORMALIZE, PLAYER_RENDER_MODE),
            )
        )

        max_workers = min(len(preload_jobs), cpu_count)
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(job, *args) for job, args in preload_jobs]
                for future in as_completed(futures):
                    future.result()
        except Exception as exc:
            print(f"Preload paralelo desativado, seguindo em modo normal: {exc}")

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
        position_vector = numpy.array(position, dtype=numpy.float32)
        for submesh in submeshes:
            material_index = submesh["material_index"]
            material = material_cache.get(material_index)
            if material is None:
                material = self._get_glb_material(file_path, material_index)
                material_cache[material_index] = material
                if material is not self.ground_texture:
                    materials.append(material)
            meshes.append(Mesh(self.shaders[0], material, position_vector.copy(), submesh["vertices"]))
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
        position_vector = numpy.array(position, dtype=numpy.float32)
        for submesh in model_data["submeshes"]:
            material_index = submesh["material_index"]
            material = material_cache.get(material_index)
            if material is None:
                material = self._get_glb_material(file_path, material_index)
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
                position_vector.copy(),
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
                        position_vector.copy(),
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

    def _get_glb_material(self, file_path, material_index):
        if material_index < 0:
            return self.ground_texture

        cache_key = (file_path, int(material_index))
        material = self._glb_material_cache.get(cache_key)
        if material is not None:
            return material

        material_images = Mesh.load_glb_material_images(file_path, material_index=material_index)
        material = Material.from_compatible_glb_images(material_images)
        self._glb_material_cache[cache_key] = material
        self._owned_materials.append(material)
        return material

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
        for shader in self.shaders:
            glDeleteProgram(shader)
        self.crate_texture.destroy()
        self.ground_texture.destroy()
        for material in self._owned_materials:
            material.destroy()
        for mesh in self.terrain_meshes:
            mesh.destroy()
        for mesh in self.player_meshes:
            mesh.destroy()
        for mesh in self.npc_draw_meshes:
            mesh.destroy()
        for gizmo in self.npc_vision_gizmos:
            gizmo.destroy()
        for mesh in self._scene_draw_meshes:
            mesh.destroy()
        for cube in self.cubes:
            cube.destroy()
        for lamp in self.lampada:
            lamp.destroy()
        self.sky.destroy()

    def on_draw(self):
        with self.profiler.section("draw.total"):
            self.clear()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            with self.profiler.section("draw.sky"):
                glDepthMask(GL_FALSE)
                self.sky.draw()
                glDepthMask(GL_TRUE)

            with self.profiler.section("draw.terrain"):
                for mesh in self.terrain_meshes:
                    mesh.draw()

            with self.profiler.section("draw.player"):
                for mesh in self.player_meshes:
                    mesh.draw()

            with self.profiler.section("draw.npcs"):
                for mesh in self.npc_draw_meshes:
                    mesh.draw()

            if self.npc_vision_gizmos:
                with self.profiler.section("draw.npc_debug"):
                    for gizmo in self.npc_vision_gizmos:
                        gizmo.draw()

            with self.profiler.section("draw.cubes"):
                for cube in self.cubes:
                    cube.draw()

            with self.profiler.section("draw.scene"):
                for mesh in self._scene_draw_meshes:
                    mesh.draw()

            with self.profiler.section("draw.lamps"):
                for lamp in self.lampada:
                    lamp.draw()

        self.profiler.maybe_print()

    def on_update(self, delta_time):
        with self.profiler.section("update.total"):
            with self.profiler.section("update.lights"):
                for light in self.dynamic_lights:
                    light.update()

            with self.profiler.section("update.player"):
                self.player.update(delta_time)

            with self.profiler.section("update.npcs"):
                for npc in self.npcs:
                    npc.update(delta_time)
                for npc, gizmo in zip(self.npcs, self.npc_vision_gizmos):
                    gizmo.position[0] = float(npc.position[0])
                    gizmo.position[1] = float(npc.position[1])
                    gizmo.position[2] = float(npc.position[2])
                    forward_vector = npc.get_debug_cone_forward_vector()
                    gizmo.set_direction_2d(forward_vector[0], forward_vector[1])

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
