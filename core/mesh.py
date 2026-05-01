import hashlib
import json
import pickle
import struct
from io import BytesIO
from pathlib import Path
from typing import Optional

from OpenGL.GL import *
import numpy, pyrr
from PIL import Image


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


class TerrainGridSampler:
    def __init__(self, vertices, clamp_outside=True) -> None:
        self.clamp_outside = clamp_outside
        self.vertices = numpy.array(vertices, dtype=numpy.float32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("TerrainGridSampler expects Nx3 vertices.")

        self.x_values = numpy.unique(numpy.round(self.vertices[:, 0], decimals=6))
        self.y_values = numpy.unique(numpy.round(self.vertices[:, 1], decimals=6))
        self.cols = len(self.x_values)
        self.rows = len(self.y_values)

        if self.cols < 2 or self.rows < 2 or self.cols * self.rows != len(self.vertices):
            raise ValueError("TerrainGridSampler could not reconstruct a regular grid from OBJ vertices.")

        self.min_x = float(self.x_values[0])
        self.max_x = float(self.x_values[-1])
        self.min_y = float(self.y_values[0])
        self.max_y = float(self.y_values[-1])

        self.height_grid = numpy.zeros((self.rows, self.cols), dtype=numpy.float32)
        x_lookup = {round(float(value), 6): index for index, value in enumerate(self.x_values)}
        y_lookup = {round(float(value), 6): index for index, value in enumerate(self.y_values)}
        for x, y, z in self.vertices:
            self.height_grid[y_lookup[round(float(y), 6)], x_lookup[round(float(x), 6)]] = z

    @classmethod
    def from_obj(cls, file_name):
        vertices = []
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                if not line.startswith("v "):
                    continue
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
        return cls(vertices)

    @classmethod
    def from_glb(cls, file_name):
        vertices = Mesh.extract_glb_unique_positions(file_name)
        return cls(vertices)

    def sample_height(self, x, y):
        local_x = float(x)
        local_y = float(y)

        if self.clamp_outside:
            local_x = min(max(local_x, self.min_x), self.max_x)
            local_y = min(max(local_y, self.min_y), self.max_y)
        elif local_x < self.min_x or local_x > self.max_x or local_y < self.min_y or local_y > self.max_y:
            return 0.0

        col = int(numpy.searchsorted(self.x_values, local_x, side="right") - 1)
        row = int(numpy.searchsorted(self.y_values, local_y, side="right") - 1)
        col = max(0, min(col, self.cols - 2))
        row = max(0, min(row, self.rows - 2))

        x0 = float(self.x_values[col])
        x1 = float(self.x_values[col + 1])
        y0 = float(self.y_values[row])
        y1 = float(self.y_values[row + 1])

        z00 = float(self.height_grid[row, col])
        z10 = float(self.height_grid[row, col + 1])
        z11 = float(self.height_grid[row + 1, col + 1])
        z01 = float(self.height_grid[row + 1, col])

        tx = 0.0 if x1 == x0 else (local_x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (local_y - y0) / (y1 - y0)

        if tx >= ty:
            return self._sample_triangle(
                (x0, y0, z00),
                (x1, y0, z10),
                (x1, y1, z11),
                local_x,
                local_y,
            )

        return self._sample_triangle(
            (x1, y1, z11),
            (x0, y1, z01),
            (x0, y0, z00),
            local_x,
            local_y,
        )

    def contains_point(self, x, y, padding=0.0):
        return (
            self.min_x + padding <= float(x) <= self.max_x - padding
            and self.min_y + padding <= float(y) <= self.max_y - padding
        )

    def contains_circle(self, x, y, radius):
        return self.contains_point(x, y, padding=float(radius))

    @staticmethod
    def _sample_triangle(a, b, c, px, py):
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c

        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1e-6:
            return az

        w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
        w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
        w3 = 1.0 - w1 - w2
        return (w1 * az) + (w2 * bz) + (w3 * cz)


class Mesh:
    CACHE_DIR = Path(".cache") / "obj"
    PREPARED_CACHE_DIR = Path(".cache") / "obj_prepared"
    PREPARED_SUBMESH_CACHE_DIR = Path(".cache") / "glb_prepared"
    MATERIAL_CACHE_DIR = Path(".cache") / "glb_materials"
    UNIQUE_POSITIONS_CACHE_DIR = Path(".cache") / "glb_positions"
    SKINNED_CACHE_DIR = Path(".cache") / "glb_skinned"

    def __init__(self, shader, material, position, vertices: Optional[tuple] = None, faces: Optional[tuple] = None, scale=1.0) -> None:
        self.material = material
        self.shader = shader
        self.position = position
        self.scale = scale
        self.collider_mode = "aabb"
        self.collider_radius_scale = 0.45
        self.collider_radius_padding = 0.0
        self.collider_height_padding = 0.0
        self.rotation = [0, 0, 0]
        self.identity = pyrr.matrix44.create_identity(dtype=numpy.float32)
        self.model = None
        self._bounds_cache = None
        self._ground_footprint_cache = None
        self._bounds_cache_key = None
        glUseProgram(self.shader)
        #x, y, z, s, t, nx, ny, nz
        if vertices != None:
            self.vertices = vertices
        else:
            self.vertices = (
                -0.5, -0.5, -0.5, 0, 0, 0, 0, -1,
                 0.5, -0.5, -0.5, 1, 0, 0, 0, -1,
                 0.5,  0.5, -0.5, 1, 1, 0, 0, -1,

                 0.5,  0.5, -0.5, 1, 1, 0, 0, -1,
                -0.5,  0.5, -0.5, 0, 1, 0, 0, -1,
                -0.5, -0.5, -0.5, 0, 0, 0, 0, -1,

                -0.5, -0.5,  0.5, 0, 0, 0, 0,  1,
                 0.5, -0.5,  0.5, 1, 0, 0, 0,  1,
                 0.5,  0.5,  0.5, 1, 1, 0, 0,  1,

                 0.5,  0.5,  0.5, 1, 1, 0, 0,  1,
                -0.5,  0.5,  0.5, 0, 1, 0, 0,  1,
                -0.5, -0.5,  0.5, 0, 0, 0, 0,  1,

                -0.5,  0.5,  0.5, 1, 0, -1, 0,  0,
                -0.5,  0.5, -0.5, 1, 1, -1, 0,  0,
                -0.5, -0.5, -0.5, 0, 1, -1, 0,  0,

                -0.5, -0.5, -0.5, 0, 1, -1, 0,  0,
                -0.5, -0.5,  0.5, 0, 0, -1, 0,  0,
                -0.5,  0.5,  0.5, 1, 0, -1, 0,  0,

                 0.5,  0.5,  0.5, 1, 0, 1, 0,  0,
                 0.5,  0.5, -0.5, 1, 1, 1, 0,  0,
                 0.5, -0.5, -0.5, 0, 1, 1, 0,  0,

                 0.5, -0.5, -0.5, 0, 1, 1, 0,  0,
                 0.5, -0.5,  0.5, 0, 0, 1, 0,  0,
                 0.5,  0.5,  0.5, 1, 0, 1, 0,  0,

                -0.5, -0.5, -0.5, 0, 1, 0, -1,  0,
                 0.5, -0.5, -0.5, 1, 1, 0, -1,  0,
                 0.5, -0.5,  0.5, 1, 0, 0, -1,  0,

                 0.5, -0.5,  0.5, 1, 0, 0, -1,  0,
                -0.5, -0.5,  0.5, 0, 0, 0, -1,  0,
                -0.5, -0.5, -0.5, 0, 1, 0, -1,  0,

                -0.5,  0.5, -0.5, 0, 1, 0, 1,  0,
                 0.5,  0.5, -0.5, 1, 1, 0, 1,  0,
                 0.5,  0.5,  0.5, 1, 0, 0, 1,  0,

                 0.5,  0.5,  0.5, 1, 0, 0, 1,  0,
                -0.5,  0.5,  0.5, 0, 0, 0, 1,  0,
                -0.5,  0.5, -0.5, 0, 1, 0, 1,  0
            )
        if faces != None:
            self.faces = faces

        if len(self.vertices) % 11 != 0:
            self.vertices = Mesh.append_tangents(self.vertices, vertex_size=8)
        self.vertex_size = 11
        self.vertex_count = len(self.vertices)//self.vertex_size
        self.vertices = numpy.array(self.vertices, dtype=numpy.float32)
        self.local_bounds = self._compute_local_bounds()

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        stride = self.vertex_size * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))

    @staticmethod
    def load_obj(file_name):
        print(f'Carregando modelo: {file_name}')
        source_path = Path(file_name)
        cache_file = Mesh._get_cache_path(source_path)
        source_stat = source_path.stat()

        cached_vertices = Mesh._load_cached_vertices(cache_file, source_stat)
        if cached_vertices is not None:
            print(f'cache carregado: {file_name}')
            return cached_vertices

        vertices, v, vn, vt = [], [], [], []

        def emit_vertex(face_token):
            values = face_token.split('/')
            vertex_index = int(values[0]) - 1
            tex_index = None
            normal_index = None

            if len(values) > 1 and values[1] != '':
                tex_index = int(values[1]) - 1
            if len(values) > 2 and values[2] != '':
                normal_index = int(values[2]) - 1

            [vertices.append(value) for value in v[vertex_index]]

            if tex_index is not None and tex_index < len(vt):
                [vertices.append(value) for value in vt[tex_index]]
            else:
                vertices.extend((0.0, 0.0))

            if normal_index is not None and normal_index < len(vn):
                [vertices.append(value) for value in vn[normal_index]]
            else:
                vertices.extend((0.0, 0.0, 0.0))

        with open(file_name, 'r') as file:
            data = file.read().splitlines()
        for line in data:
            if line.startswith('v '):
                v.append([float(x) for x in line[1:].strip().split(' ')])
            elif line.startswith('vn '):
                vn.append([float(x) for x in line[2:].strip().split(' ')])
            elif line.startswith('vt '):
                vt.append([float(x) for x in line[2:].strip().split(' ')])
            elif line.startswith('f '):
                f_line = line[1:].strip().split(' ')
                if len(f_line) < 3:
                    continue

                # OBJ faces can come as v, v/vt, v//vn or v/vt/vn.
                # We triangulate polygons with a simple fan.
                triangles = [(0, index, index + 1) for index in range(1, len(f_line) - 1)]
                for triangle in triangles:
                    for index in triangle:
                        emit_vertex(f_line[index])

        Mesh._store_cached_vertices(cache_file, source_stat, vertices)
        print(f'modelo carregado: {file_name}')
        return vertices


    @staticmethod
    def load_glb(file_name):
        print(f'Carregando modelo GLB: {file_name}')
        source_path = Path(file_name)
        cache_file = Mesh._get_cache_path(source_path)
        source_stat = source_path.stat()

        cached_vertices = Mesh._load_cached_vertices(cache_file, source_stat)
        if cached_vertices is not None:
            print(f'cache carregado: {file_name}')
            return cached_vertices

        vertices = []
        for submesh in Mesh.load_glb_submeshes(file_name):
            vertices.extend(submesh["vertices"])

        Mesh._store_cached_vertices(cache_file, source_stat, vertices)
        print(f'modelo GLB carregado: {file_name}')
        return vertices


    @staticmethod
    def load_glb_submeshes(file_name):
        document, binary_chunk = Mesh._read_glb(Path(file_name))
        nodes = document.get("nodes", [])
        scene_index = document.get("scene", 0)
        scenes = document.get("scenes", [])
        if scenes and 0 <= scene_index < len(scenes):
            root_nodes = scenes[scene_index].get("nodes", [])
        else:
            root_nodes = list(range(len(nodes)))

        submeshes = []
        identity = numpy.identity(4, dtype=numpy.float32)

        def visit(node_index, parent_matrix):
            node = nodes[node_index]
            local_matrix = Mesh._glb_node_matrix(node)
            world_matrix = numpy.matmul(parent_matrix, local_matrix)

            if "mesh" in node:
                mesh_index = node["mesh"]
                mesh_def = document["meshes"][mesh_index]
                for primitive_index, primitive in enumerate(mesh_def.get("primitives", [])):
                    if primitive.get("mode", 4) != 4:
                        continue
                    vertices = Mesh._build_glb_primitive_vertices(document, binary_chunk, primitive, world_matrix)
                    material_index = primitive.get("material")
                    submeshes.append(
                        {
                            "name": f'{mesh_def.get("name", f"mesh_{mesh_index}")}_primitive_{primitive_index}',
                            "material_index": -1 if material_index is None else int(material_index),
                            "vertices": vertices,
                        }
                    )

            for child_index in node.get("children", []):
                visit(child_index, world_matrix)

        for root_index in root_nodes:
            visit(root_index, identity)

        return submeshes


    @staticmethod
    def load_obj_prepared(
        file_name,
        invert_texcoord=True,
        st_pos=4,
        vertex_size=8,
        target_size=3.5,
        normalize=True,
        rotation_degrees=(0.0, 0.0, 0.0),
    ):
        source_path = Path(file_name)
        cache_file = Mesh._get_prepared_cache_path(
            source_path,
            invert_texcoord=invert_texcoord,
            st_pos=st_pos,
            vertex_size=vertex_size,
            target_size=target_size,
            normalize=normalize,
            rotation_degrees=rotation_degrees,
        )
        source_stat = source_path.stat()

        cached_vertices = Mesh._load_cached_vertices(cache_file, source_stat)
        if cached_vertices is not None:
            print(f'cache preparado carregado: {file_name}')
            return cached_vertices

        vertices = Mesh.load_obj(file_name)
        if invert_texcoord:
            vertices = Mesh.invert_s_or_t(vertices, st_pos, vertex_size)
        if rotation_degrees != (0.0, 0.0, 0.0):
            vertices = Mesh.rotate_vertices(vertices, rotation_degrees=rotation_degrees, vertex_size=vertex_size)
        if normalize:
            vertices = Mesh.normalize_vertices(vertices, vertex_size=vertex_size, target_size=target_size)
        vertices = Mesh.append_tangents(vertices, vertex_size=vertex_size)
        Mesh._store_cached_vertices(cache_file, source_stat, vertices)
        return vertices


    @staticmethod
    def load_glb_submeshes_prepared(
        file_name,
        invert_texcoord=False,
        st_pos=4,
        vertex_size=8,
        target_size=3.5,
        normalize=True,
        rotation_degrees=(0.0, 0.0, 0.0),
    ):
        source_path = Path(file_name)
        cache_file = Mesh._get_prepared_submesh_cache_path(
            source_path,
            invert_texcoord=invert_texcoord,
            st_pos=st_pos,
            vertex_size=vertex_size,
            target_size=target_size,
            normalize=normalize,
            rotation_degrees=rotation_degrees,
        )
        source_stat = source_path.stat()

        cached_submeshes = Mesh._load_cached_submeshes(cache_file, source_stat)
        if cached_submeshes is not None:
            print(f'cache preparado carregado: {file_name}')
            return cached_submeshes

        prepared = Mesh.load_glb_submeshes(file_name)
        vertex_batches = [list(submesh["vertices"]) for submesh in prepared]

        if invert_texcoord:
            vertex_batches = [
                Mesh.invert_s_or_t(batch, st_pos, vertex_size)
                for batch in vertex_batches
            ]
        if rotation_degrees != (0.0, 0.0, 0.0):
            vertex_batches = [
                Mesh.rotate_vertices(batch, rotation_degrees=rotation_degrees, vertex_size=vertex_size)
                for batch in vertex_batches
            ]
        if normalize:
            vertex_batches = Mesh.normalize_vertex_batches(
                vertex_batches,
                vertex_size=vertex_size,
                target_size=target_size,
            )

        results = []
        for submesh, vertices in zip(prepared, vertex_batches):
            results.append(
                {
                    "name": submesh["name"],
                    "material_index": submesh["material_index"],
                    "vertices": Mesh.append_tangents(vertices, vertex_size=vertex_size),
                }
            )
        Mesh._store_cached_submeshes(cache_file, source_stat, results)
        return results


    @staticmethod
    def load_glb_prepared(
        file_name,
        invert_texcoord=False,
        st_pos=4,
        vertex_size=8,
        target_size=3.5,
        normalize=True,
        rotation_degrees=(0.0, 0.0, 0.0),
    ):
        source_path = Path(file_name)
        cache_file = Mesh._get_prepared_cache_path(
            source_path,
            invert_texcoord=invert_texcoord,
            st_pos=st_pos,
            vertex_size=vertex_size,
            target_size=target_size,
            normalize=normalize,
            rotation_degrees=rotation_degrees,
        )
        source_stat = source_path.stat()

        cached_vertices = Mesh._load_cached_vertices(cache_file, source_stat)
        if cached_vertices is not None:
            print(f'cache preparado carregado: {file_name}')
            return cached_vertices

        vertices = Mesh.load_glb(file_name)
        if invert_texcoord:
            vertices = Mesh.invert_s_or_t(vertices, st_pos, vertex_size)
        if rotation_degrees != (0.0, 0.0, 0.0):
            vertices = Mesh.rotate_vertices(vertices, rotation_degrees=rotation_degrees, vertex_size=vertex_size)
        if normalize:
            vertices = Mesh.normalize_vertices(vertices, vertex_size=vertex_size, target_size=target_size)
        vertices = Mesh.append_tangents(vertices, vertex_size=vertex_size)
        Mesh._store_cached_vertices(cache_file, source_stat, vertices)
        return vertices


    @staticmethod
    def load_glb_material_images(file_name, material_index=0):
        source_path = Path(file_name)
        cache_file = Mesh._get_material_cache_path(source_path, material_index)
        source_stat = source_path.stat()

        cached_images = Mesh._load_cached_material_images(cache_file, source_stat)
        if cached_images is not None:
            print(f'cache material carregado: {file_name} [{material_index}]')
            return cached_images

        document, binary_chunk = Mesh._read_glb(source_path)
        materials = document.get("materials", [])
        if material_index >= len(materials):
            raise ValueError(f"GLB material index {material_index} out of range for {file_name}")

        material = materials[material_index]
        pbr = material.get("pbrMetallicRoughness", {})

        diffuse = Mesh._read_glb_texture_image(document, binary_chunk, pbr.get("baseColorTexture", {}).get("index"))
        if diffuse is None:
            diffuse_factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            diffuse = Image.new("RGBA", (1, 1), Mesh._factor_to_rgba(diffuse_factor))

        specular_info = material.get("extensions", {}).get("KHR_materials_specular", {})
        specular = Mesh._read_glb_texture_image(document, binary_chunk, specular_info.get("specularColorTexture", {}).get("index"))
        if specular is None:
            specular_factor = float(specular_info.get("specularFactor", 1.0))
            specular_color = specular_info.get("specularColorFactor", [1.0, 1.0, 1.0])
            specular_rgb = [value * specular_factor for value in specular_color]
            specular = Image.new("RGBA", (1, 1), Mesh._factor_to_rgba([*specular_rgb, 1.0]))

        normal = Mesh._read_glb_texture_image(document, binary_chunk, material.get("normalTexture", {}).get("index"))

        images = {
            "diffuse": diffuse,
            "specular": specular,
            "normal": normal,
        }
        Mesh._store_cached_material_images(cache_file, source_stat, images)
        return images


    @staticmethod
    def _get_cache_path(source_path: Path) -> Path:
        Mesh.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return Mesh.CACHE_DIR / f"{source_path.stem}.{source_key}.npz"


    @staticmethod
    def _get_prepared_cache_path(source_path: Path, **config) -> Path:
        Mesh.PREPARED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config_key = "|".join(f"{key}={config[key]}" for key in sorted(config))
        cache_key = hashlib.sha1(
            f"{source_path.resolve()}|{config_key}".encode("utf-8")
        ).hexdigest()[:12]
        return Mesh.PREPARED_CACHE_DIR / f"{source_path.stem}.{cache_key}.npz"


    @staticmethod
    def _get_prepared_submesh_cache_path(source_path: Path, **config) -> Path:
        Mesh.PREPARED_SUBMESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config_key = "|".join(f"{key}={config[key]}" for key in sorted(config))
        cache_key = hashlib.sha1(
            f"{source_path.resolve()}|submeshes|{config_key}".encode("utf-8")
        ).hexdigest()[:12]
        return Mesh.PREPARED_SUBMESH_CACHE_DIR / f"{source_path.stem}.{cache_key}.npz"


    @staticmethod
    def _get_material_cache_path(source_path: Path, material_index: int) -> Path:
        Mesh.MATERIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(
            f"{source_path.resolve()}|material={material_index}|v2".encode("utf-8")
        ).hexdigest()[:12]
        return Mesh.MATERIAL_CACHE_DIR / f"{source_path.stem}.{cache_key}.npz"


    @staticmethod
    def _get_unique_positions_cache_path(source_path: Path) -> Path:
        Mesh.UNIQUE_POSITIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(
            f"{source_path.resolve()}|unique_positions".encode("utf-8")
        ).hexdigest()[:12]
        return Mesh.UNIQUE_POSITIONS_CACHE_DIR / f"{source_path.stem}.{cache_key}.npz"


    @staticmethod
    def _get_skinned_cache_path(source_path: Path, **config) -> Path:
        Mesh.SKINNED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config_key = "|".join(f"{key}={config[key]}" for key in sorted(config))
        cache_key = hashlib.sha1(
            f"{source_path.resolve()}|skinned|{config_key}".encode("utf-8")
        ).hexdigest()[:12]
        return Mesh.SKINNED_CACHE_DIR / f"{source_path.stem}.{cache_key}.pkl"


    @staticmethod
    def _load_cached_vertices(cache_file: Path, source_stat):
        if not cache_file.exists():
            return None

        try:
            with numpy.load(cache_file, allow_pickle=False) as cache_data:
                cached_mtime_ns = int(cache_data["mtime_ns"][0])
                cached_size = int(cache_data["file_size"][0])
                if cached_mtime_ns != source_stat.st_mtime_ns or cached_size != source_stat.st_size:
                    return None

                return cache_data["vertices"].astype(numpy.float32).tolist()
        except (OSError, KeyError, ValueError):
            return None


    @staticmethod
    def _load_cached_submeshes(cache_file: Path, source_stat):
        if not cache_file.exists():
            return None

        try:
            with numpy.load(cache_file, allow_pickle=False) as cache_data:
                cached_mtime_ns = int(cache_data["mtime_ns"][0])
                cached_size = int(cache_data["file_size"][0])
                if cached_mtime_ns != source_stat.st_mtime_ns or cached_size != source_stat.st_size:
                    return None

                names = cache_data["names"].tolist()
                material_indices = cache_data["material_indices"].astype(numpy.int32).tolist()
                vertices = cache_data["vertices"].astype(numpy.float32)
                offsets = cache_data["offsets"].astype(numpy.int64)

                submeshes = []
                for index, start in enumerate(offsets[:-1]):
                    end = offsets[index + 1]
                    submeshes.append(
                        {
                            "name": str(names[index]),
                            "material_index": int(material_indices[index]),
                            "vertices": vertices[start:end].tolist(),
                        }
                    )
                return submeshes
        except (OSError, KeyError, ValueError):
            return None


    @staticmethod
    def _load_cached_material_images(cache_file: Path, source_stat):
        if not cache_file.exists():
            return None

        try:
            with numpy.load(cache_file, allow_pickle=False) as cache_data:
                cached_mtime_ns = int(cache_data["mtime_ns"][0])
                cached_size = int(cache_data["file_size"][0])
                if cached_mtime_ns != source_stat.st_mtime_ns or cached_size != source_stat.st_size:
                    return None

                return {
                    "diffuse": Mesh._decode_cached_image(cache_data, "diffuse"),
                    "specular": Mesh._decode_cached_image(cache_data, "specular"),
                    "normal": Mesh._decode_cached_image(cache_data, "normal"),
                }
        except (OSError, KeyError, ValueError):
            return None


    @staticmethod
    def _load_cached_unique_positions(cache_file: Path, source_stat):
        if not cache_file.exists():
            return None

        try:
            with numpy.load(cache_file, allow_pickle=False) as cache_data:
                cached_mtime_ns = int(cache_data["mtime_ns"][0])
                cached_size = int(cache_data["file_size"][0])
                if cached_mtime_ns != source_stat.st_mtime_ns or cached_size != source_stat.st_size:
                    return None

                return cache_data["positions"].astype(numpy.float32)
        except (OSError, KeyError, ValueError):
            return None


    @staticmethod
    def _load_cached_skinned_data(cache_file: Path, source_stat):
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as file:
                cache_data = pickle.load(file)
        except (OSError, ValueError, pickle.PickleError):
            return None

        cached_mtime_ns = int(cache_data.get("mtime_ns", -1))
        cached_size = int(cache_data.get("file_size", -1))
        if cached_mtime_ns != source_stat.st_mtime_ns or cached_size != source_stat.st_size:
            return None

        return cache_data.get("data")


    @staticmethod
    def _store_cached_vertices(cache_file: Path, source_stat, vertices):
        numpy.savez_compressed(
            cache_file,
            vertices=numpy.array(vertices, dtype=numpy.float32),
            mtime_ns=numpy.array([source_stat.st_mtime_ns], dtype=numpy.int64),
            file_size=numpy.array([source_stat.st_size], dtype=numpy.int64),
        )


    @staticmethod
    def _store_cached_submeshes(cache_file: Path, source_stat, submeshes):
        names = []
        material_indices = []
        offsets = [0]
        flat_vertices = []

        for submesh in submeshes:
            names.append(submesh["name"])
            material_indices.append(int(submesh["material_index"]))
            flat_vertices.extend(submesh["vertices"])
            offsets.append(len(flat_vertices))

        numpy.savez_compressed(
            cache_file,
            names=numpy.array(names),
            material_indices=numpy.array(material_indices, dtype=numpy.int32),
            vertices=numpy.array(flat_vertices, dtype=numpy.float32),
            offsets=numpy.array(offsets, dtype=numpy.int64),
            mtime_ns=numpy.array([source_stat.st_mtime_ns], dtype=numpy.int64),
            file_size=numpy.array([source_stat.st_size], dtype=numpy.int64),
        )


    @staticmethod
    def _store_cached_material_images(cache_file: Path, source_stat, images):
        diffuse_data = Mesh._encode_cached_image(images.get("diffuse"))
        specular_data = Mesh._encode_cached_image(images.get("specular"))
        normal_data = Mesh._encode_cached_image(images.get("normal"))

        numpy.savez_compressed(
            cache_file,
            diffuse_pixels=diffuse_data["pixels"],
            diffuse_size=diffuse_data["size"],
            specular_pixels=specular_data["pixels"],
            specular_size=specular_data["size"],
            normal_pixels=normal_data["pixels"],
            normal_size=normal_data["size"],
            mtime_ns=numpy.array([source_stat.st_mtime_ns], dtype=numpy.int64),
            file_size=numpy.array([source_stat.st_size], dtype=numpy.int64),
        )


    @staticmethod
    def _store_cached_unique_positions(cache_file: Path, source_stat, positions):
        numpy.savez_compressed(
            cache_file,
            positions=numpy.asarray(positions, dtype=numpy.float32),
            mtime_ns=numpy.array([source_stat.st_mtime_ns], dtype=numpy.int64),
            file_size=numpy.array([source_stat.st_size], dtype=numpy.int64),
        )


    @staticmethod
    def _store_cached_skinned_data(cache_file: Path, source_stat, data):
        with open(cache_file, "wb") as file:
            pickle.dump(
                {
                    "mtime_ns": int(source_stat.st_mtime_ns),
                    "file_size": int(source_stat.st_size),
                    "data": data,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


    @staticmethod
    def _encode_cached_image(image):
        if image is None:
            return {
                "pixels": numpy.array([], dtype=numpy.uint8),
                "size": numpy.array([0, 0], dtype=numpy.int32),
            }

        rgba = image.convert("RGBA")
        return {
            "pixels": numpy.frombuffer(rgba.tobytes(), dtype=numpy.uint8),
            "size": numpy.array([rgba.size[0], rgba.size[1]], dtype=numpy.int32),
        }


    @staticmethod
    def _decode_cached_image(cache_data, prefix):
        size = cache_data[f"{prefix}_size"].astype(numpy.int32)
        width, height = int(size[0]), int(size[1])
        if width == 0 or height == 0:
            return None
        pixels = cache_data[f"{prefix}_pixels"].astype(numpy.uint8).tobytes()
        return Image.frombytes("RGBA", (width, height), pixels)


    @staticmethod
    def _factor_to_rgba(values):
        rgba = list(values[:4]) + [1.0] * max(0, 4 - len(values))
        return tuple(
            int(max(0.0, min(1.0, float(component))) * 255.0)
            for component in rgba[:4]
        )


    @staticmethod
    def invert_s_or_t(vertices, st_pos, vertice_size):
        vertices_len = len(vertices) // vertice_size
        for x in range(vertices_len):
            vertices[x * vertice_size + st_pos] = 1 - vertices[x * vertice_size + st_pos]
        return vertices


    @staticmethod
    def invert_st(vertices, s_pos, t_pos, vertice_size):
        return Mesh.invert_s_or_t(Mesh.invert_s_or_t(vertices, s_pos, vertice_size), t_pos, vertice_size)


    @staticmethod
    def normalize_vertices(vertices, vertex_size=8, target_size=3.0):
        if not vertices:
            return vertices

        normalized = list(vertices)
        positions = numpy.array(
            [normalized[index:index + 3] for index in range(0, len(normalized), vertex_size)],
            dtype=numpy.float32,
        )

        min_corner = positions.min(axis=0)
        max_corner = positions.max(axis=0)
        center_xy = (min_corner[:2] + max_corner[:2]) / 2
        min_z = min_corner[2]
        size = max_corner - min_corner
        max_dimension = max(float(size.max()), 1e-6)
        scale = target_size / max_dimension

        for index in range(0, len(normalized), vertex_size):
            normalized[index] = float((normalized[index] - center_xy[0]) * scale)
            normalized[index + 1] = float((normalized[index + 1] - center_xy[1]) * scale)
            normalized[index + 2] = float((normalized[index + 2] - min_z) * scale)

        return normalized


    @staticmethod
    def normalize_vertex_batches(vertex_batches, vertex_size=8, target_size=3.0):
        if not vertex_batches:
            return vertex_batches

        all_positions = []
        for vertices in vertex_batches:
            if not vertices:
                continue
            all_positions.extend(
                vertices[index:index + 3]
                for index in range(0, len(vertices), vertex_size)
            )

        if not all_positions:
            return [list(vertices) for vertices in vertex_batches]

        positions = numpy.array(all_positions, dtype=numpy.float32)
        min_corner = positions.min(axis=0)
        max_corner = positions.max(axis=0)
        center_xy = (min_corner[:2] + max_corner[:2]) / 2
        min_z = min_corner[2]
        size = max_corner - min_corner
        max_dimension = max(float(size.max()), 1e-6)
        scale = target_size / max_dimension

        normalized_batches = []
        for vertices in vertex_batches:
            normalized = list(vertices)
            for index in range(0, len(normalized), vertex_size):
                normalized[index] = float((normalized[index] - center_xy[0]) * scale)
                normalized[index + 1] = float((normalized[index + 1] - center_xy[1]) * scale)
                normalized[index + 2] = float((normalized[index + 2] - min_z) * scale)
            normalized_batches.append(normalized)

        return normalized_batches


    @staticmethod
    def append_tangents(vertices, vertex_size=8):
        if not vertices:
            return vertices
        if vertex_size == 11 and len(vertices) % 11 == 0:
            return list(vertices)

        expanded = []
        triangle_stride = vertex_size * 3
        for triangle_start in range(0, len(vertices), triangle_stride):
            tri = vertices[triangle_start:triangle_start + triangle_stride]
            if len(tri) < triangle_stride:
                break

            p0 = numpy.array(tri[0:3], dtype=numpy.float32)
            uv0 = numpy.array(tri[3:5], dtype=numpy.float32)
            n0 = numpy.array(tri[5:8], dtype=numpy.float32)

            p1 = numpy.array(tri[vertex_size:vertex_size + 3], dtype=numpy.float32)
            uv1 = numpy.array(tri[vertex_size + 3:vertex_size + 5], dtype=numpy.float32)

            p2 = numpy.array(tri[vertex_size * 2:vertex_size * 2 + 3], dtype=numpy.float32)
            uv2 = numpy.array(tri[vertex_size * 2 + 3:vertex_size * 2 + 5], dtype=numpy.float32)

            edge1 = p1 - p0
            edge2 = p2 - p0
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            denominator = (delta_uv1[0] * delta_uv2[1]) - (delta_uv2[0] * delta_uv1[1])

            if abs(float(denominator)) < 1e-6:
                tangent = numpy.cross(n0, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32))
                if numpy.linalg.norm(tangent) < 1e-6:
                    tangent = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
            else:
                inv = 1.0 / denominator
                tangent = inv * ((delta_uv2[1] * edge1) - (delta_uv1[1] * edge2))

            tangent_norm = max(float(numpy.linalg.norm(tangent)), 1e-6)
            tangent = tangent / tangent_norm

            for vertex_index in range(3):
                offset = vertex_index * vertex_size
                expanded.extend(tri[offset:offset + vertex_size])
                expanded.extend((float(tangent[0]), float(tangent[1]), float(tangent[2])))

        return expanded


    @staticmethod
    def rotate_vertices(vertices, rotation_degrees=(0.0, 0.0, 0.0), vertex_size=8):
        if not vertices:
            return vertices

        rotation_x, rotation_y, rotation_z = rotation_degrees
        rotation_matrix = pyrr.matrix44.create_identity(dtype=numpy.float32)
        rotation_matrix = pyrr.matrix44.multiply(
            rotation_matrix,
            pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(rotation_x), dtype=numpy.float32),
        )
        rotation_matrix = pyrr.matrix44.multiply(
            rotation_matrix,
            pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(rotation_y), dtype=numpy.float32),
        )
        rotation_matrix = pyrr.matrix44.multiply(
            rotation_matrix,
            pyrr.matrix44.create_from_z_rotation(theta=numpy.radians(rotation_z), dtype=numpy.float32),
        )

        rotated = list(vertices)
        for index in range(0, len(rotated), vertex_size):
            position = numpy.array([rotated[index], rotated[index + 1], rotated[index + 2], 1.0], dtype=numpy.float32)
            rotated_position = pyrr.matrix44.apply_to_vector(rotation_matrix, position)
            rotated[index] = float(rotated_position[0])
            rotated[index + 1] = float(rotated_position[1])
            rotated[index + 2] = float(rotated_position[2])

            normal_index = index + 5
            if normal_index + 2 < index + vertex_size:
                normal = numpy.array([rotated[normal_index], rotated[normal_index + 1], rotated[normal_index + 2], 0.0], dtype=numpy.float32)
                rotated_normal = pyrr.matrix44.apply_to_vector(rotation_matrix, normal)
                rotated[normal_index] = float(rotated_normal[0])
                rotated[normal_index + 1] = float(rotated_normal[1])
                rotated[normal_index + 2] = float(rotated_normal[2])

        return rotated


    @staticmethod
    def _read_glb(source_path: Path):
        data = source_path.read_bytes()
        magic, version, _ = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f"Unsupported GLB file: {source_path}")

        offset = 12
        json_chunk = None
        binary_chunk = b""
        while offset < len(data):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk_data = data[offset:offset + chunk_length]
            offset += chunk_length

            if chunk_type == 0x4E4F534A:
                json_chunk = json.loads(chunk_data.decode("utf-8").rstrip("\x00"))
            elif chunk_type == 0x004E4942:
                binary_chunk = chunk_data

        if json_chunk is None:
            raise ValueError(f"GLB without JSON chunk: {source_path}")
        return json_chunk, binary_chunk


    @staticmethod
    def _read_glb_accessor(document, binary_chunk, accessor_index):
        accessor = document["accessors"][accessor_index]
        buffer_view = document["bufferViews"][accessor["bufferView"]]
        component_type = accessor["componentType"]
        accessor_type = accessor["type"]
        count = accessor["count"]
        component_count = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[accessor_type]
        dtype = {
            5120: numpy.int8,
            5121: numpy.uint8,
            5122: numpy.int16,
            5123: numpy.uint16,
            5125: numpy.uint32,
            5126: numpy.float32,
        }[component_type]

        component_size = numpy.dtype(dtype).itemsize
        element_size = component_size * component_count
        byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        byte_stride = buffer_view.get("byteStride", element_size)

        if byte_stride == element_size:
            flat = numpy.frombuffer(binary_chunk, dtype=dtype, count=count * component_count, offset=byte_offset)
            return flat.reshape(count, component_count).astype(numpy.float32)

        values = numpy.empty((count, component_count), dtype=numpy.float32)
        for index in range(count):
            start = byte_offset + index * byte_stride
            chunk = binary_chunk[start:start + element_size]
            values[index] = numpy.frombuffer(chunk, dtype=dtype, count=component_count).astype(numpy.float32)
        return values


    @staticmethod
    def _glb_node_matrix(node):
        if "matrix" in node:
            return numpy.array(node["matrix"], dtype=numpy.float32).reshape((4, 4))

        translation = numpy.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=numpy.float32)
        rotation = numpy.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=numpy.float32)
        scale = numpy.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=numpy.float32)

        translation_matrix = pyrr.matrix44.create_from_translation(translation, dtype=numpy.float32)
        rotation_matrix = pyrr.matrix44.create_from_quaternion(rotation, dtype=numpy.float32)
        scale_matrix = pyrr.matrix44.create_from_scale(scale, dtype=numpy.float32)
        return numpy.matmul(numpy.matmul(translation_matrix, rotation_matrix), scale_matrix)

    @staticmethod
    def _glb_node_matrix_cpu(node):
        if "matrix" in node:
            return numpy.array(node["matrix"], dtype=numpy.float32).reshape((4, 4)).T

        translation = numpy.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=numpy.float32)
        rotation = numpy.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=numpy.float32)
        scale = numpy.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=numpy.float32)
        x, y, z, w = rotation
        rotation_matrix = numpy.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y), 0.0],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x), 0.0],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=numpy.float32,
        )
        translation_matrix = numpy.identity(4, dtype=numpy.float32)
        translation_matrix[:3, 3] = translation
        scale_matrix = numpy.diag([scale[0], scale[1], scale[2], 1.0]).astype(numpy.float32)
        return translation_matrix @ rotation_matrix @ scale_matrix


    @staticmethod
    def _build_glb_primitive_vertices(document, binary_chunk, primitive, world_matrix):
        attributes = primitive.get("attributes", {})
        positions = Mesh._read_glb_accessor(document, binary_chunk, attributes["POSITION"])
        texcoords = None
        normals = None
        if "TEXCOORD_0" in attributes:
            texcoords = Mesh._read_glb_accessor(document, binary_chunk, attributes["TEXCOORD_0"])
        if "NORMAL" in attributes:
            normals = Mesh._read_glb_accessor(document, binary_chunk, attributes["NORMAL"])

        if "indices" in primitive:
            indices = Mesh._read_glb_accessor(document, binary_chunk, primitive["indices"]).reshape(-1)
        else:
            indices = numpy.arange(len(positions), dtype=numpy.int32)

        normal_matrix = numpy.linalg.inv(world_matrix[:3, :3]).T
        vertices = []

        for vertex_index in indices:
            local_position = positions[int(vertex_index)]
            position = world_matrix @ numpy.array(
                [local_position[0], local_position[1], local_position[2], 1.0],
                dtype=numpy.float32,
            )
            vertices.extend((float(position[0]), float(position[1]), float(position[2])))

            if texcoords is not None:
                uv = texcoords[int(vertex_index)]
                vertices.extend((float(uv[0]), float(uv[1])))
            else:
                vertices.extend((0.0, 0.0))

            if normals is not None:
                local_normal = normals[int(vertex_index)]
                normal = normal_matrix @ local_normal
                normal /= max(float(numpy.linalg.norm(normal)), 1e-6)
                vertices.extend((float(normal[0]), float(normal[1]), float(normal[2])))
            else:
                vertices.extend((0.0, 0.0, 1.0))

        return vertices


    @staticmethod
    def _read_glb_texture_image(document, binary_chunk, texture_index):
        if texture_index is None:
            return None

        texture = document["textures"][texture_index]
        image_def = document["images"][texture["source"]]

        if "bufferView" in image_def:
            buffer_view = document["bufferViews"][image_def["bufferView"]]
            byte_offset = buffer_view.get("byteOffset", 0)
            byte_length = buffer_view["byteLength"]
            image_bytes = binary_chunk[byte_offset:byte_offset + byte_length]
            return Image.open(BytesIO(image_bytes)).convert("RGBA")

        if "uri" in image_def:
            image_path = Path(image_def["uri"])
            return Image.open(image_path).convert("RGBA")

        return None


    @staticmethod
    def extract_glb_unique_positions(file_name):
        source_path = Path(file_name)
        cache_file = Mesh._get_unique_positions_cache_path(source_path)
        source_stat = source_path.stat()

        cached_positions = Mesh._load_cached_unique_positions(cache_file, source_stat)
        if cached_positions is not None:
            return cached_positions

        document, binary_chunk = Mesh._read_glb(source_path)
        nodes = document.get("nodes", [])
        scene_index = document.get("scene", 0)
        scenes = document.get("scenes", [])
        if scenes and 0 <= scene_index < len(scenes):
            root_nodes = scenes[scene_index].get("nodes", [])
        else:
            root_nodes = list(range(len(nodes)))

        unique_positions = {}
        identity = numpy.identity(4, dtype=numpy.float32)

        def visit(node_index, parent_matrix):
            node = nodes[node_index]
            local_matrix = Mesh._glb_node_matrix(node)
            world_matrix = numpy.matmul(parent_matrix, local_matrix)

            if "mesh" in node:
                mesh_def = document["meshes"][node["mesh"]]
                for primitive in mesh_def.get("primitives", []):
                    attributes = primitive.get("attributes", {})
                    if "POSITION" not in attributes:
                        continue
                    positions = Mesh._read_glb_accessor(document, binary_chunk, attributes["POSITION"])
                    for local_position in positions:
                        position = world_matrix @ numpy.array(
                            [local_position[0], local_position[1], local_position[2], 1.0],
                            dtype=numpy.float32,
                        )
                        key = (
                            round(float(position[0]), 6),
                            round(float(position[1]), 6),
                            round(float(position[2]), 6),
                        )
                        unique_positions[key] = key

            for child_index in node.get("children", []):
                visit(child_index, world_matrix)

        for root_index in root_nodes:
            visit(root_index, identity)

        positions = numpy.array(list(unique_positions.values()), dtype=numpy.float32)
        Mesh._store_cached_unique_positions(cache_file, source_stat, positions)
        return positions


    @staticmethod
    def load_glb_skinned_data(
        file_name,
        invert_texcoord=False,
        target_size=1.0,
        normalize=True,
    ):
        source_path = Path(file_name)
        cache_file = Mesh._get_skinned_cache_path(
            source_path,
            invert_texcoord=invert_texcoord,
            target_size=target_size,
            normalize=normalize,
        )
        source_stat = source_path.stat()

        cached_data = Mesh._load_cached_skinned_data(cache_file, source_stat)
        if cached_data is not None:
            print(f'cache skinned carregado: {file_name}')
            return cached_data

        document, binary_chunk = Mesh._read_glb(source_path)
        nodes = document.get("nodes", [])
        node_parents = [-1] * len(nodes)
        root_nodes = Mesh._get_glb_root_nodes(document)
        submeshes = []

        for parent_index, node in enumerate(nodes):
            for child_index in node.get("children", []):
                node_parents[child_index] = parent_index

        default_local_matrices = [Mesh._glb_node_matrix_cpu(node) for node in nodes]
        bind_world_matrices = Mesh._build_node_world_matrices(default_local_matrices, node_parents)

        def visit(node_index):
            node = nodes[node_index]
            if "mesh" in node:
                mesh_index = node["mesh"]
                mesh_def = document["meshes"][mesh_index]
                for primitive_index, primitive in enumerate(mesh_def.get("primitives", [])):
                    if primitive.get("mode", 4) != 4:
                        continue
                    submeshes.append(
                        {
                            "name": f'{mesh_def.get("name", f"mesh_{mesh_index}")}_primitive_{primitive_index}',
                            "material_index": -1 if primitive.get("material") is None else int(primitive["material"]),
                            "mesh_node_index": node_index,
                            "mesh_bind_matrix": bind_world_matrices[node_index].copy(),
                            "skin_index": node.get("skin"),
                            "vertices": Mesh._build_glb_skinned_primitive_vertices(
                                document,
                                binary_chunk,
                                primitive,
                                invert_texcoord=invert_texcoord,
                            ),
                        }
                    )

            for child_index in node.get("children", []):
                visit(child_index)

        for root_index in root_nodes:
            visit(root_index)

        skins = []
        for skin in document.get("skins", []):
            joints = [int(joint_index) for joint_index in skin.get("joints", [])]
            if "inverseBindMatrices" in skin:
                inverse_bind_matrices = Mesh._read_glb_accessor(
                    document,
                    binary_chunk,
                    skin["inverseBindMatrices"],
                ).reshape((-1, 4, 4)).transpose((0, 2, 1))
            else:
                inverse_bind_matrices = numpy.repeat(
                    numpy.identity(4, dtype=numpy.float32)[numpy.newaxis, :, :],
                    len(joints),
                    axis=0,
                )
            skins.append(
                {
                    "joints": joints,
                    "inverse_bind_matrices": inverse_bind_matrices.astype(numpy.float32),
                }
            )

        animations = {}
        for animation_index, animation in enumerate(document.get("animations", [])):
            channels_by_node = {}
            duration = 0.0
            for channel in animation.get("channels", []):
                sampler = animation["samplers"][channel["sampler"]]
                node_index = int(channel["target"]["node"])
                path = channel["target"]["path"]
                times = Mesh._read_glb_accessor(document, binary_chunk, sampler["input"]).reshape(-1)
                values = Mesh._read_glb_accessor(document, binary_chunk, sampler["output"])
                channels_by_node.setdefault(node_index, {})[path] = {
                    "times": times.astype(numpy.float32),
                    "values": values.astype(numpy.float32),
                    "interpolation": sampler.get("interpolation", "LINEAR"),
                }
                if len(times) > 0:
                    duration = max(duration, float(times[-1]))

            animation_name = animation.get("name") or f"animation_{animation_index}"
            animations[animation_name] = {
                "name": animation_name,
                "duration": duration,
                "channels": channels_by_node,
            }

        node_transforms = []
        for node in nodes:
            node_transforms.append(
                {
                    "name": node.get("name"),
                    "translation": numpy.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=numpy.float32),
                    "rotation": numpy.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=numpy.float32),
                    "scale": numpy.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=numpy.float32),
                }
            )

        result = {
            "submeshes": submeshes,
            "skins": skins,
            "animations": animations,
            "node_transforms": node_transforms,
            "node_parents": node_parents,
            "transform": Mesh._calculate_skinned_model_transform(
                submeshes,
                target_size=target_size,
                normalize=normalize,
            ),
        }
        Mesh._store_cached_skinned_data(cache_file, source_stat, result)
        return result


    @staticmethod
    def _get_glb_root_nodes(document):
        nodes = document.get("nodes", [])
        scene_index = document.get("scene", 0)
        scenes = document.get("scenes", [])
        if scenes and 0 <= scene_index < len(scenes):
            return scenes[scene_index].get("nodes", [])
        return list(range(len(nodes)))


    @staticmethod
    def _build_node_world_matrices(local_matrices, node_parents):
        world_matrices = [None] * len(local_matrices)

        def resolve(node_index):
            if world_matrices[node_index] is not None:
                return world_matrices[node_index]

            parent_index = node_parents[node_index]
            local_matrix = local_matrices[node_index]
            if parent_index >= 0:
                world_matrices[node_index] = numpy.matmul(resolve(parent_index), local_matrix)
            else:
                world_matrices[node_index] = local_matrix
            return world_matrices[node_index]

        for node_index in range(len(local_matrices)):
            resolve(node_index)

        return [matrix.astype(numpy.float32) for matrix in world_matrices]


    @staticmethod
    def _build_glb_skinned_primitive_vertices(document, binary_chunk, primitive, invert_texcoord=False):
        attributes = primitive.get("attributes", {})
        positions = Mesh._read_glb_accessor(document, binary_chunk, attributes["POSITION"])
        texcoords = Mesh._read_glb_accessor(document, binary_chunk, attributes["TEXCOORD_0"]) if "TEXCOORD_0" in attributes else None
        normals = Mesh._read_glb_accessor(document, binary_chunk, attributes["NORMAL"]) if "NORMAL" in attributes else None
        joints = Mesh._read_glb_accessor(document, binary_chunk, attributes["JOINTS_0"]) if "JOINTS_0" in attributes else None
        weights = Mesh._read_glb_accessor(document, binary_chunk, attributes["WEIGHTS_0"]) if "WEIGHTS_0" in attributes else None

        if "indices" in primitive:
            indices = Mesh._read_glb_accessor(document, binary_chunk, primitive["indices"]).reshape(-1).astype(numpy.int32)
        else:
            indices = numpy.arange(len(positions), dtype=numpy.int32)

        vertices = []
        for triangle_start in range(0, len(indices), 3):
            triangle_indices = indices[triangle_start:triangle_start + 3]
            if len(triangle_indices) < 3:
                continue

            tangent = Mesh._calculate_triangle_tangent(
                positions[int(triangle_indices[0])],
                positions[int(triangle_indices[1])],
                positions[int(triangle_indices[2])],
                texcoords[int(triangle_indices[0])] if texcoords is not None else numpy.array([0.0, 0.0], dtype=numpy.float32),
                texcoords[int(triangle_indices[1])] if texcoords is not None else numpy.array([0.0, 0.0], dtype=numpy.float32),
                texcoords[int(triangle_indices[2])] if texcoords is not None else numpy.array([0.0, 0.0], dtype=numpy.float32),
                normals[int(triangle_indices[0])] if normals is not None else numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32),
            )

            for vertex_index in triangle_indices:
                vertex_index = int(vertex_index)
                position = positions[vertex_index]
                uv = texcoords[vertex_index] if texcoords is not None else numpy.array([0.0, 0.0], dtype=numpy.float32)
                if invert_texcoord:
                    uv = numpy.array([uv[0], 1.0 - uv[1]], dtype=numpy.float32)
                normal = normals[vertex_index] if normals is not None else numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)
                normal_norm = max(float(numpy.linalg.norm(normal)), 1e-6)
                normal = normal / normal_norm

                joint_values = joints[vertex_index] if joints is not None else numpy.array([0.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
                weight_values = weights[vertex_index] if weights is not None else numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
                weight_sum = float(numpy.sum(weight_values))
                if weight_sum > 1e-6:
                    weight_values = weight_values / weight_sum
                else:
                    weight_values = numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)

                vertices.extend((float(position[0]), float(position[1]), float(position[2])))
                vertices.extend((float(uv[0]), float(uv[1])))
                vertices.extend((float(normal[0]), float(normal[1]), float(normal[2])))
                vertices.extend((float(tangent[0]), float(tangent[1]), float(tangent[2])))
                vertices.extend(float(value) for value in joint_values[:4])
                vertices.extend(float(value) for value in weight_values[:4])

        return vertices


    @staticmethod
    def _calculate_triangle_tangent(p0, p1, p2, uv0, uv1, uv2, normal):
        edge1 = p1 - p0
        edge2 = p2 - p0
        delta_uv1 = uv1 - uv0
        delta_uv2 = uv2 - uv0
        denominator = (delta_uv1[0] * delta_uv2[1]) - (delta_uv2[0] * delta_uv1[1])

        if abs(float(denominator)) < 1e-6:
            tangent = numpy.cross(normal, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32))
            if numpy.linalg.norm(tangent) < 1e-6:
                tangent = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        else:
            tangent = ((delta_uv2[1] * edge1) - (delta_uv1[1] * edge2)) / denominator

        tangent_norm = max(float(numpy.linalg.norm(tangent)), 1e-6)
        return tangent.astype(numpy.float32) / tangent_norm


    @staticmethod
    def _calculate_skinned_model_transform(submeshes, target_size=1.0, normalize=True):
        if not submeshes:
            return {
                "scale": 1.0,
                "origin_offset": numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32),
            }

        positions = []
        for submesh in submeshes:
            vertices = submesh["vertices"]
            mesh_bind_matrix = numpy.array(submesh["mesh_bind_matrix"], dtype=numpy.float32)
            positions.extend(
                (mesh_bind_matrix @ numpy.array([*vertices[index:index + 3], 1.0], dtype=numpy.float32))[:3]
                for index in range(0, len(vertices), 19)
            )

        if not positions:
            return 1.0

        positions = numpy.array(positions, dtype=numpy.float32)
        min_corner = positions.min(axis=0)
        max_corner = positions.max(axis=0)
        if not normalize:
            return {
                "scale": 1.0,
                "origin_offset": numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32),
            }
        size = max_corner - min_corner
        max_dimension = max(float(size.max()), 1e-6)
        center_xy = (min_corner[:2] + max_corner[:2]) / 2.0
        return {
            "scale": float(target_size) / max_dimension,
            "origin_offset": numpy.array(
                [-center_xy[0], -center_xy[1], -min_corner[2]],
                dtype=numpy.float32,
            ),
        }


    @staticmethod
    def create_plane(width=40.0, depth=30.0, uv_scale=8.0):
        half_width = width / 2
        half_depth = depth / 2
        return (
            -half_width, -half_depth, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
             half_width, -half_depth, 0.0, uv_scale, 0.0, 0.0, 0.0, 1.0,
             half_width,  half_depth, 0.0, uv_scale, uv_scale, 0.0, 0.0, 1.0,

             half_width,  half_depth, 0.0, uv_scale, uv_scale, 0.0, 0.0, 1.0,
            -half_width,  half_depth, 0.0, 0.0, uv_scale, 0.0, 0.0, 1.0,
            -half_width, -half_depth, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )


    @staticmethod
    def create_terrain(width=40.0, depth=30.0, rows=40, cols=40, uv_scale=6.0, height_fn=None):
        def sample_height(x, y):
            if height_fn is None:
                return 0.0
            return float(height_fn(x, y))

        def sample_normal(x, y):
            epsilon = 0.25
            h_l = sample_height(x - epsilon, y)
            h_r = sample_height(x + epsilon, y)
            h_d = sample_height(x, y - epsilon)
            h_u = sample_height(x, y + epsilon)
            normal = numpy.array([h_l - h_r, h_d - h_u, 2.0], dtype=numpy.float32)
            normal /= max(numpy.linalg.norm(normal), 1e-6)
            return normal

        vertices = []
        half_width = width / 2
        half_depth = depth / 2
        step_x = width / cols
        step_y = depth / rows

        for row in range(rows):
            y0 = -half_depth + row * step_y
            y1 = y0 + step_y
            v0 = (row / rows) * uv_scale
            v1 = ((row + 1) / rows) * uv_scale

            for col in range(cols):
                x0 = -half_width + col * step_x
                x1 = x0 + step_x
                u0 = (col / cols) * uv_scale
                u1 = ((col + 1) / cols) * uv_scale

                p00 = (x0, y0, sample_height(x0, y0), u0, v0, *sample_normal(x0, y0))
                p10 = (x1, y0, sample_height(x1, y0), u1, v0, *sample_normal(x1, y0))
                p11 = (x1, y1, sample_height(x1, y1), u1, v1, *sample_normal(x1, y1))
                p01 = (x0, y1, sample_height(x0, y1), u0, v1, *sample_normal(x0, y1))

                vertices.extend(p00)
                vertices.extend(p10)
                vertices.extend(p11)
                vertices.extend(p11)
                vertices.extend(p01)
                vertices.extend(p00)

        return tuple(vertices)


    @staticmethod
    def export_grid_obj(file_name, width=40.0, depth=30.0, rows=40, cols=40, uv_scale=6.0, height_fn=None):
        def sample_height(x, y):
            if height_fn is None:
                return 0.0
            return float(height_fn(x, y))

        def sample_normal(x, y):
            epsilon = 0.25
            h_l = sample_height(x - epsilon, y)
            h_r = sample_height(x + epsilon, y)
            h_d = sample_height(x, y - epsilon)
            h_u = sample_height(x, y + epsilon)
            normal = numpy.array([h_l - h_r, h_d - h_u, 2.0], dtype=numpy.float32)
            normal /= max(numpy.linalg.norm(normal), 1e-6)
            return normal

        output_path = Path(file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        half_width = width / 2
        half_depth = depth / 2
        step_x = width / cols
        step_y = depth / rows

        vertices = []
        texcoords = []
        normals = []

        for row in range(rows + 1):
            y = -half_depth + row * step_y
            v = (row / rows) * uv_scale
            for col in range(cols + 1):
                x = -half_width + col * step_x
                u = (col / cols) * uv_scale
                vertices.append((x, y, sample_height(x, y)))
                texcoords.append((u, v))
                normals.append(tuple(sample_normal(x, y)))

        def grid_index(row, col):
            return row * (cols + 1) + col + 1

        lines = ["# generated terrain", "g terrain"]
        for x, y, z in vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for u, v in texcoords:
            lines.append(f"vt {u:.6f} {v:.6f}")
        for nx, ny, nz in normals:
            lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

        for row in range(rows):
            for col in range(cols):
                p00 = grid_index(row, col)
                p10 = grid_index(row, col + 1)
                p11 = grid_index(row + 1, col + 1)
                p01 = grid_index(row + 1, col)
                lines.append(f"f {p00}/{p00}/{p00} {p10}/{p10}/{p10} {p11}/{p11}/{p11}")
                lines.append(f"f {p11}/{p11}/{p11} {p01}/{p01}/{p01} {p00}/{p00}/{p00}")

        content = "\n".join(lines) + "\n"
        if output_path.exists():
            existing_content = output_path.read_text(encoding="utf-8")
            if existing_content == content:
                return

        output_path.write_text(content, encoding="utf-8")


    def _compute_local_bounds(self):
        positions = self.vertices.reshape(-1, self.vertex_size)[:, :3]
        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0)
        return minimum, maximum


    def get_world_bounds(self):
        position = self.position
        cache_key = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(self.scale),
        )
        if self._bounds_cache_key == cache_key and self._bounds_cache is not None:
            return self._bounds_cache

        minimum, maximum = self.local_bounds
        scaled_min = minimum * self.scale
        scaled_max = maximum * self.scale
        position_vector = numpy.asarray(position, dtype=numpy.float32)
        self._bounds_cache_key = cache_key
        self._bounds_cache = (scaled_min + position_vector, scaled_max + position_vector)
        self._ground_footprint_cache = None
        return self._bounds_cache

    def set_collider(self, mode="aabb", radius_scale=None, radius_padding=None, height_padding=None):
        self.collider_mode = mode
        if radius_scale is not None:
            self.collider_radius_scale = float(radius_scale)
        if radius_padding is not None:
            self.collider_radius_padding = float(radius_padding)
        if height_padding is not None:
            self.collider_height_padding = float(height_padding)
        return self

    def get_ground_footprint(self):
        if self._ground_footprint_cache is not None and self._bounds_cache_key is not None:
            return self._ground_footprint_cache

        bounds_min, bounds_max = self.get_world_bounds()
        center_x = float((bounds_min[0] + bounds_max[0]) / 2.0)
        center_y = float((bounds_min[1] + bounds_max[1]) / 2.0)
        half_width = float((bounds_max[0] - bounds_min[0]) / 2.0)
        half_depth = float((bounds_max[1] - bounds_min[1]) / 2.0)
        radius = max(half_width, half_depth) * self.collider_radius_scale + self.collider_radius_padding
        self._ground_footprint_cache = (center_x, center_y, radius)
        return self._ground_footprint_cache

    def collides_with_circle(self, position, radius, probe_z):
        bounds_min, bounds_max = self.get_world_bounds()
        if probe_z < bounds_min[2] - self.collider_height_padding or probe_z > bounds_max[2] + 2.0 + self.collider_height_padding:
            return False

        if self.collider_mode == "circle":
            center_x, center_y, obstacle_radius = self.get_ground_footprint()
            delta_x = float(position[0] - center_x)
            delta_y = float(position[1] - center_y)
            total_radius = float(radius + obstacle_radius)
            return delta_x * delta_x + delta_y * delta_y < total_radius * total_radius

        closest_x = min(max(position[0], bounds_min[0]), bounds_max[0])
        closest_y = min(max(position[1], bounds_min[1]), bounds_max[1])
        delta_x = position[0] - closest_x
        delta_y = position[1] - closest_y
        return delta_x * delta_x + delta_y * delta_y < radius * radius


    def _build_model_matrix(self, rotation_matrix):
        scale_vector = numpy.array([self.scale, self.scale, self.scale], dtype=numpy.float32)
        scale_matrix = pyrr.matrix44.create_from_scale(scale_vector, dtype=numpy.float32)
        translation_matrix = pyrr.matrix44.create_from_translation(vec=numpy.array(self.position), dtype=numpy.float32)
        model = pyrr.matrix44.multiply(scale_matrix, rotation_matrix)
        return pyrr.matrix44.multiply(model, translation_matrix)

    def _compose_rotation_matrix(self):
        model = pyrr.matrix44.multiply(
            self.identity,
            pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(self.rotation[0]), dtype=numpy.float32),
        )
        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(self.rotation[1]), dtype=numpy.float32),
        )
        return pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_z_rotation(theta=numpy.radians(self.rotation[2]), dtype=numpy.float32),
        )


    def translate(self, model):
        self.model = self._build_model_matrix(model)


    def rotate_x(self, angle):
        self.rotation[0] = (self.rotation[0] + angle) % 360
        model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(self.rotation[0]), dtype=numpy.float32))
        self.translate(model)


    def rotate_y(self, angle):
        self.rotation[1] = (self.rotation[1] + angle) % 360
        model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(self.rotation[1]), dtype=numpy.float32))
        self.translate(model)


    def rotate_z(self, angle):
        self.rotation[2] = (self.rotation[2] + angle) % 360
        model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_z_rotation(theta=numpy.radians(self.rotation[2]), dtype=numpy.float32))
        self.translate(model)


    def rotate_xy(self, angle):
        self.rotation[0] = (self.rotation[0] + angle) % 360
        self.rotation[1] = (self.rotation[1] + angle) % 360
        model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(self.rotation[0]), dtype=numpy.float32))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(self.rotation[1]), dtype=numpy.float32))
        self.translate(model)
    

    def rotate_xyz(self, angle):
        self.rotation[0] = (self.rotation[0] + angle) % 360
        self.rotation[1] = (self.rotation[1] + angle) % 360
        self.rotation[2] = (self.rotation[2] + angle) % 360
        self.translate(self._compose_rotation_matrix())

    def set_rotation(self, x=None, y=None, z=None):
        new_x = self.rotation[0] if x is None else float(x) % 360
        new_y = self.rotation[1] if y is None else float(y) % 360
        new_z = self.rotation[2] if z is None else float(z) % 360
        if x is not None:
            self.rotation[0] = new_x
        if y is not None:
            self.rotation[1] = new_y
        if z is not None:
            self.rotation[2] = new_z
        self.translate(self._compose_rotation_matrix())


    def draw(self):
        glUseProgram(self.shader)
        self.material.use()
        if type(self.model) == type(None):
            self.model = self._build_model_matrix(self.identity)
        glUniformMatrix4fv(get_uniform_location(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))



class SkinnedAnimator:
    def __init__(self, node_transforms, node_parents, animations, skins) -> None:
        self.node_transforms = node_transforms
        self.node_parents = node_parents
        self.animations = animations
        self.skins = skins
        self.node_count = len(node_transforms)
        self.current_animation_name = None
        self.current_animation = None
        self.current_channels = None
        self.current_channel_items = ()
        self.current_time = 0.0
        self.loop = True
        self.paused = False
        self.revision = 0
        self._skin_matrix_cache = {}
        self.base_translations = [
            numpy.array(transform.get("translation", [0.0, 0.0, 0.0]), dtype=numpy.float32)
            for transform in node_transforms
        ]
        self.base_rotations = [
            numpy.array(transform.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=numpy.float32)
            for transform in node_transforms
        ]
        self.base_scales = [
            numpy.array(transform.get("scale", [1.0, 1.0, 1.0]), dtype=numpy.float32)
            for transform in node_transforms
        ]
        self.base_local_matrices = [
            self._compose_trs_matrix(
                self.base_translations[node_index],
                self.base_rotations[node_index],
                self.base_scales[node_index],
            )
            for node_index in range(self.node_count)
        ]
        self.topological_order = self._build_topological_order()
        self._prepare_animations()
        self.world_matrices = self._compute_world_matrices()

    def has_animation(self, name):
        return name in self.animations

    def play(self, name, loop=True, paused=False, restart=False, hold_time=None):
        if name not in self.animations:
            return False

        if restart or self.current_animation_name != name:
            self.current_time = 0.0 if hold_time is None else float(hold_time)

        self.current_animation_name = name
        self.current_animation = self.animations[name]
        self.current_channels = self.current_animation["channels"]
        self.current_channel_items = self.current_animation["channel_items"]
        self.loop = bool(loop)
        self.paused = bool(paused)
        self._reset_animation_cursors()

        if hold_time is not None:
            self.current_time = float(hold_time)

        self.world_matrices = self._compute_world_matrices()
        self._skin_matrix_cache.clear()
        self.revision += 1
        return True

    def update(self, delta_time):
        world_changed = False
        if self.current_animation is not None and not self.paused:
            duration = float(self.current_animation["duration"])
            if duration > 0.0:
                previous_time = self.current_time
                self.current_time += float(delta_time)
                if self.loop:
                    did_wrap = self.current_time >= duration
                    self.current_time %= duration
                    if did_wrap:
                        self._reset_animation_cursors()
                else:
                    self.current_time = min(self.current_time, duration)
                world_changed = abs(self.current_time - previous_time) > 1e-8

        if world_changed:
            self.world_matrices = self._compute_world_matrices()
            self._skin_matrix_cache.clear()
            self.revision += 1

    def get_skin_matrices(self, skin_index, mesh_bind_matrix, mode="standard"):
        if skin_index is None or skin_index < 0 or skin_index >= len(self.skins):
            return numpy.identity(4, dtype=numpy.float32).reshape((1, 4, 4))

        cached = self._skin_matrix_cache.get(skin_index)
        if cached is not None:
            return cached

        skin = self.skins[skin_index]
        joint_matrices = []
        for joint_node_index, inverse_bind_matrix in zip(skin["joints"], skin["inverse_bind_matrices"]):
            joint_world = self.world_matrices[joint_node_index]
            joint_matrices.append(numpy.matmul(joint_world, inverse_bind_matrix))
        result = numpy.array(joint_matrices, dtype=numpy.float32)
        self._skin_matrix_cache[skin_index] = result
        return result

    def _compute_world_matrices(self):
        local_matrices = list(self.base_local_matrices)
        if self.current_channel_items:
            for node_index, channels in self.current_channel_items:
                translation = self.base_translations[node_index]
                rotation = self.base_rotations[node_index]
                scale = self.base_scales[node_index]

                if "translation" in channels:
                    translation = self._sample_channel(channels["translation"], self.current_time)
                if "rotation" in channels:
                    rotation = self._sample_channel(channels["rotation"], self.current_time)
                    rotation_norm = max(float(numpy.linalg.norm(rotation)), 1e-6)
                    rotation = rotation / rotation_norm
                if "scale" in channels:
                    scale = self._sample_channel(channels["scale"], self.current_time)

                local_matrices[node_index] = self._compose_trs_matrix(translation, rotation, scale)

        world_matrices = [None] * self.node_count
        for node_index in self.topological_order:
            parent_index = self.node_parents[node_index]
            local_matrix = local_matrices[node_index]
            if parent_index >= 0:
                world_matrices[node_index] = numpy.matmul(world_matrices[parent_index], local_matrix)
            else:
                world_matrices[node_index] = local_matrix
        return world_matrices

    def _prepare_animations(self):
        for animation in self.animations.values():
            channel_items = []
            for node_index, channels in animation["channels"].items():
                for channel in channels.values():
                    channel["_cursor"] = 0
                channel_items.append((node_index, channels))
            animation["channel_items"] = tuple(channel_items)

    def _reset_animation_cursors(self):
        if self.current_animation is None:
            return
        for _node_index, channels in self.current_channel_items:
            for channel in channels.values():
                channel["_cursor"] = 0

    def _build_topological_order(self):
        children = [[] for _ in range(self.node_count)]
        roots = []
        for node_index, parent_index in enumerate(self.node_parents):
            if parent_index >= 0:
                children[parent_index].append(node_index)
            else:
                roots.append(node_index)

        order = []
        stack = list(reversed(roots))
        while stack:
            node_index = stack.pop()
            order.append(node_index)
            node_children = children[node_index]
            for child_index in reversed(node_children):
                stack.append(child_index)

        if len(order) != self.node_count:
            return list(range(self.node_count))
        return order

    @staticmethod
    def _compose_trs_matrix(translation, rotation, scale):
        x, y, z, w = rotation
        rotation_matrix = numpy.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y), 0.0],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x), 0.0],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=numpy.float32,
        )
        result = rotation_matrix.copy()
        result[0, :3] *= float(scale[0])
        result[1, :3] *= float(scale[1])
        result[2, :3] *= float(scale[2])
        result[:3, 3] = translation
        return result

    def _sample_channel(self, channel, time_value):
        times = channel["times"]
        values = channel["values"]
        interpolation = channel.get("interpolation", "LINEAR")

        if len(times) == 0:
            return values[0]
        if len(times) == 1 or time_value <= float(times[0]):
            channel["_cursor"] = 0
            return values[0]
        if time_value >= float(times[-1]):
            channel["_cursor"] = max(len(times) - 2, 0)
            return values[-1]

        prev_index = int(channel.get("_cursor", 0))
        max_prev_index = len(times) - 2
        if prev_index > max_prev_index:
            prev_index = max_prev_index

        while prev_index < max_prev_index and time_value > float(times[prev_index + 1]):
            prev_index += 1
        while prev_index > 0 and time_value < float(times[prev_index]):
            prev_index -= 1

        channel["_cursor"] = prev_index
        next_index = prev_index + 1

        prev_time = float(times[prev_index])
        next_time = float(times[next_index])
        if next_time <= prev_time:
            return values[next_index]

        factor = (float(time_value) - prev_time) / (next_time - prev_time)
        if interpolation == "STEP":
            return values[prev_index]

        if values.shape[1] == 4:
            return self._slerp(values[prev_index], values[next_index], factor)
        return ((1.0 - factor) * values[prev_index]) + (factor * values[next_index])

    @staticmethod
    def _slerp(start, end, factor):
        start = start.astype(numpy.float32)
        end = end.astype(numpy.float32)
        dot = float(numpy.dot(start, end))

        if dot < 0.0:
            end = -end
            dot = -dot

        if dot > 0.9995:
            result = start + factor * (end - start)
            result_norm = max(float(numpy.linalg.norm(result)), 1e-6)
            return result / result_norm

        theta_0 = numpy.arccos(dot)
        sin_theta_0 = numpy.sin(theta_0)
        theta = theta_0 * factor
        sin_theta = numpy.sin(theta)

        s0 = numpy.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * start) + (s1 * end)


class SkinnedMesh(Mesh):
    def __init__(
        self,
        shader,
        material,
        position,
        vertices,
        animator,
        skin_index,
        mesh_bind_matrix,
        skinning_mode="standard",
        post_skinning_transform=None,
        origin_offset=None,
        scale=1.0,
    ) -> None:
        self.material = material
        self.shader = shader
        self.position = position
        self.scale = scale
        self.animator = animator
        self.skin_index = skin_index
        self.mesh_bind_matrix = numpy.array(mesh_bind_matrix, dtype=numpy.float32)
        self.skinning_mode = skinning_mode
        if post_skinning_transform is None:
            post_skinning_transform = numpy.identity(4, dtype=numpy.float32)
        self.post_skinning_transform = numpy.array(post_skinning_transform, dtype=numpy.float32)
        if origin_offset is None:
            origin_offset = [0.0, 0.0, 0.0]
        self.origin_offset = numpy.array(origin_offset, dtype=numpy.float32)
        self.collider_mode = "aabb"
        self.collider_radius_scale = 0.45
        self.collider_radius_padding = 0.0
        self.collider_height_padding = 0.0
        self.rotation = [0, 0, 0]
        self.identity = pyrr.matrix44.create_identity(dtype=numpy.float32)
        self.model = None
        self._bounds_cache = None
        self._ground_footprint_cache = None
        self._bounds_cache_key = None
        glUseProgram(self.shader)

        self.base_vertex_size = 19
        self.vertex_size = 11
        self.base_vertices = numpy.array(vertices, dtype=numpy.float32).reshape(-1, self.base_vertex_size)
        self.base_positions = self.base_vertices[:, 0:3]
        self.base_uvs = self.base_vertices[:, 3:5]
        self.base_normals = self.base_vertices[:, 5:8]
        self.base_tangents = self.base_vertices[:, 8:11]
        self.base_joint_ids = self.base_vertices[:, 11:15].astype(numpy.int32)
        self.base_joint_weights = self.base_vertices[:, 15:19].astype(numpy.float32)
        self.position4 = numpy.concatenate(
            [self.base_positions, numpy.ones((len(self.base_vertices), 1), dtype=numpy.float32)],
            axis=1,
        )
        self.vertex_count = len(self.base_vertices)
        self.vertices = numpy.array(vertices, dtype=numpy.float32).reshape(-1, self.base_vertex_size)
        self.bone_matrices = self._compute_bone_matrices()
        self._last_animator_revision = self.animator.revision
        self.local_bounds = self._compute_bind_local_bounds()
        self.local_bounds = (
            self.local_bounds[0] + self.origin_offset,
            self.local_bounds[1] + self.origin_offset,
        )

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        initial_buffer = numpy.ascontiguousarray(self.vertices, dtype=numpy.float32).ravel()
        glBufferData(GL_ARRAY_BUFFER, initial_buffer.nbytes, initial_buffer, GL_DYNAMIC_DRAW)

        stride = self.base_vertex_size * 4

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))

        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(44))

        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(60))

    def draw(self):
        glUseProgram(self.shader)
        self.material.use()
        if self.model is None:
            self.model = self._build_model_matrix(self.identity)

        glUniformMatrix4fv(get_uniform_location(self.shader, "model"), 1, GL_FALSE, self.model)
        glUniformMatrix4fv(
            get_uniform_location(self.shader, "meshBindMatrix"),
            1,
            GL_FALSE,
            numpy.ascontiguousarray(self.mesh_bind_matrix.T, dtype=numpy.float32),
        )
        glUniformMatrix4fv(
            get_uniform_location(self.shader, "postSkinningTransform"),
            1,
            GL_FALSE,
            numpy.ascontiguousarray(self.post_skinning_transform.T, dtype=numpy.float32),
        )
        glUniform1i(get_uniform_location(self.shader, "boneCount"), int(len(self.bone_matrices)))
        if len(self.bone_matrices) > 0:
            glUniformMatrix4fv(
                get_uniform_location(self.shader, "boneMatrices"),
                len(self.bone_matrices),
                GL_FALSE,
                numpy.ascontiguousarray(self.bone_matrices.transpose((0, 2, 1)), dtype=numpy.float32),
            )
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def _build_model_matrix(self, rotation_matrix):
        scale_vector = numpy.array([self.scale, self.scale, self.scale], dtype=numpy.float32)
        scale_matrix = pyrr.matrix44.create_from_scale(scale_vector, dtype=numpy.float32)
        local_offset_matrix = pyrr.matrix44.create_from_translation(self.origin_offset, dtype=numpy.float32)
        translation_matrix = pyrr.matrix44.create_from_translation(vec=numpy.array(self.position), dtype=numpy.float32)
        model = pyrr.matrix44.multiply(local_offset_matrix, rotation_matrix)
        model = pyrr.matrix44.multiply(scale_matrix, model)
        return pyrr.matrix44.multiply(model, translation_matrix)

    def _compute_bind_local_bounds(self):
        bind_positions = self._apply_skinning_to_positions(self.bone_matrices)
        positions = bind_positions[:, :3]
        return positions.min(axis=0), positions.max(axis=0)

    def update_skinning(self):
        if self._last_animator_revision == self.animator.revision:
            return
        self.bone_matrices = self._compute_bone_matrices()
        self._last_animator_revision = self.animator.revision

    def _compute_bone_matrices(self):
        return self.animator.get_skin_matrices(
            self.skin_index,
            self.mesh_bind_matrix,
            mode=self.skinning_mode,
        )

    def _apply_skinning_to_positions(self, bone_matrices):
        selected_bones = bone_matrices[self.base_joint_ids]
        skin_matrices = (selected_bones * self.base_joint_weights[:, :, numpy.newaxis, numpy.newaxis]).sum(axis=1)
        skinned_positions = numpy.einsum("nij,nj->ni", skin_matrices, self.position4)
        bind_positions = numpy.einsum("ij,nj->ni", self.mesh_bind_matrix, skinned_positions)
        corrected_positions = numpy.einsum("ij,nj->ni", self.post_skinning_transform, bind_positions)
        return corrected_positions


class SkinnedModel:
    def __init__(self, meshes, materials, animator) -> None:
        self.meshes = meshes
        self.materials = materials
        self.animator = animator

    def has_animation(self, name):
        return self.animator.has_animation(name)

    def play(self, name, loop=True, paused=False, restart=False, hold_time=None):
        return self.animator.play(
            name,
            loop=loop,
            paused=paused,
            restart=restart,
            hold_time=hold_time,
        )

    def update(self, delta_time):
        self.animator.update(delta_time)
        for mesh in self.meshes:
            if hasattr(mesh, "update_skinning"):
                mesh.update_skinning()


class MeshRGB:
    def __init__(
        self,
        shader,
        position,
        vertices=None,
        color=[1, 1, 1],
        scale=1.0,
        billboard_target=None,
    ) -> None:
        self.shader = shader
        self.position = position
        self.scale = scale
        self.billboard_target = billboard_target
        self.identity = pyrr.matrix44.create_identity(dtype=numpy.float32)
        self.model = None
        self.rotation = [0.0, 0.0, 0.0]
        glUseProgram(self.shader)
        #x, y, z, r, g, b
        if vertices != None:
            self.vertices = vertices
        else:
            self.vertices = (
                -0.1, 0.1, -0.1, *color, 
                0.1, 0.1, 0.1, *color,   
                0.1, 0.1, -0.1, *color,
                0.1, 0.1, 0.1, *color,
                -0.1, -0.1, 0.1, *color,
                0.1, -0.1, 0.1, *color,
                -0.1, 0.1, 0.1, *color,
                -0.1, -0.1, -0.1, *color,
                -0.1, -0.1, 0.1, *color,
                0.1, -0.1, -0.1, *color,
                -0.1, -0.1, 0.1, *color,
                -0.1, -0.1, -0.1, *color,
                0.1, 0.1, -0.1, *color,
                0.1, -0.1, 0.1, *color,
                0.1, -0.1, -0.1, *color,
                -0.1, 0.1, -0.1,*color,
                0.1, -0.1, -0.1,*color,
                -0.1, -0.1, -0.1,*color,
                -0.1, 0.1, -0.1,*color,
                -0.1, 0.1, 0.1,*color,
                0.1, 0.1, 0.1,*color,
                0.1, 0.1, 0.1,*color,
                -0.1, 0.1, 0.1,*color,
                -0.1, -0.1, 0.1,*color,
                -0.1, 0.1, 0.1,*color,
                -0.1, 0.1, -0.1,*color,
                -0.1, -0.1, -0.1,*color,
                0.1, -0.1, -0.1,*color,
                0.1, -0.1, 0.1,*color,
                -0.1, -0.1, 0.1,*color,
                0.1, 0.1, -0.1,*color,
                0.1, 0.1, 0.1,*color,
                0.1, -0.1, 0.1,*color,
                -0.1, 0.1, -0.1,*color,
                0.1, 0.1, -0.1,*color,
                0.1, -0.1, -0.1,*color,
            )
        self.vertex_count = len(self.vertices)//6
        self.vertices = numpy.array(self.vertices, dtype=numpy.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))


    @staticmethod
    def create_gradient_box(size=1.0, top_color=(0.55, 0.78, 0.98), bottom_color=(0.9, 0.82, 0.68)):
        half = size / 2
        top = list(top_color)
        bottom = list(bottom_color)
        return (
            -half,  half, -half, *top,
             half,  half,  half, *top,
             half,  half, -half, *top,
             half,  half,  half, *top,
            -half, -half,  half, *bottom,
             half, -half,  half, *bottom,
            -half,  half,  half, *top,
            -half, -half, -half, *bottom,
            -half, -half,  half, *bottom,
             half, -half, -half, *bottom,
            -half, -half,  half, *bottom,
            -half, -half, -half, *bottom,
             half,  half, -half, *top,
             half, -half,  half, *bottom,
             half, -half, -half, *bottom,
            -half,  half, -half, *top,
             half, -half, -half, *bottom,
            -half, -half, -half, *bottom,
            -half,  half, -half, *top,
            -half,  half,  half, *top,
             half,  half,  half, *top,
             half,  half,  half, *top,
            -half,  half,  half, *top,
            -half, -half,  half, *bottom,
            -half,  half,  half, *top,
            -half,  half, -half, *top,
            -half, -half, -half, *bottom,
             half, -half, -half, *bottom,
             half, -half,  half, *bottom,
            -half, -half,  half, *bottom,
             half,  half, -half, *top,
             half,  half,  half, *top,
             half, -half,  half, *bottom,
            -half,  half, -half, *top,
             half,  half, -half, *top,
             half, -half, -half, *bottom,
        )


    @staticmethod
    def create_disc(radius=1.0, segments=24, color=(1.0, 0.9, 0.65)):
        vertices = []
        center = [0.0, 0.0, 0.0, *color]

        for index in range(segments):
            angle0 = (2 * numpy.pi * index) / segments
            angle1 = (2 * numpy.pi * (index + 1)) / segments
            p0 = [numpy.cos(angle0) * radius, numpy.sin(angle0) * radius, 0.0, *color]
            p1 = [numpy.cos(angle1) * radius, numpy.sin(angle1) * radius, 0.0, *color]
            vertices.extend(center)
            vertices.extend(p0)
            vertices.extend(p1)

        return tuple(vertices)

    @staticmethod
    def create_sector(radius=1.0, angle_degrees=75.0, segments=16, color=(1.0, 0.4, 0.25), z_offset=0.02):
        vertices = []
        center = [0.0, 0.0, z_offset, *color]
        half_angle = numpy.radians(angle_degrees * 0.5)

        for index in range(segments):
            t0 = index / segments
            t1 = (index + 1) / segments
            angle0 = -half_angle + (2.0 * half_angle * t0)
            angle1 = -half_angle + (2.0 * half_angle * t1)
            p0 = [numpy.cos(angle0) * radius, numpy.sin(angle0) * radius, z_offset, *color]
            p1 = [numpy.cos(angle1) * radius, numpy.sin(angle1) * radius, z_offset, *color]
            vertices.extend(center)
            vertices.extend(p0)
            vertices.extend(p1)

        return tuple(vertices)

    def _compose_rotation_matrix(self):
        model = pyrr.matrix44.multiply(
            self.identity,
            pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(self.rotation[0]), dtype=numpy.float32),
        )
        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(self.rotation[1]), dtype=numpy.float32),
        )
        return pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_z_rotation(theta=numpy.radians(self.rotation[2]), dtype=numpy.float32),
        )

    def _build_model_matrix(self, position, rotation_matrix):
        scale_matrix = pyrr.matrix44.create_from_scale(
            numpy.array([self.scale, self.scale, self.scale], dtype=numpy.float32),
            dtype=numpy.float32,
        )
        translation_matrix = pyrr.matrix44.create_from_translation(vec=position, dtype=numpy.float32)
        model = pyrr.matrix44.multiply(scale_matrix, rotation_matrix)
        return pyrr.matrix44.multiply(model, translation_matrix)

    def set_rotation(self, x=None, y=None, z=None):
        if x is not None:
            self.rotation[0] = float(x) % 360.0
        if y is not None:
            self.rotation[1] = float(y) % 360.0
        if z is not None:
            self.rotation[2] = float(z) % 360.0
        position = self.position.position if hasattr(self.position, "position") else self.position
        position = numpy.array(position, dtype=numpy.float32)
        self.model = self._build_model_matrix(position, self._compose_rotation_matrix())

    def set_direction_2d(self, direction_x, direction_y):
        length_sq = float(direction_x * direction_x + direction_y * direction_y)
        if length_sq <= 1e-8:
            return
        angle = float(numpy.degrees(numpy.arctan2(direction_y, direction_x)))
        self.set_rotation(z=angle)


    def draw(self):
        glUseProgram(self.shader)
        position = self.position.position if hasattr(self.position, "position") else self.position
        position = numpy.array(position, dtype=numpy.float32)

        if self.billboard_target is not None:
            target_position = self.billboard_target.position if hasattr(self.billboard_target, "position") else self.billboard_target
            target_position = numpy.array(target_position, dtype=numpy.float32)
            forward = target_position - position
            forward_norm = numpy.linalg.norm(forward)
            if forward_norm < 1e-6:
                forward = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
            else:
                forward = forward / forward_norm

            world_up = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)
            if abs(float(numpy.dot(forward, world_up))) > 0.98:
                world_up = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)

            right = numpy.cross(world_up, forward)
            right = right / max(numpy.linalg.norm(right), 1e-6)
            up = numpy.cross(forward, right)
            up = up / max(numpy.linalg.norm(up), 1e-6)

            self.model = numpy.array(
                [
                    [right[0] * self.scale, up[0] * self.scale, forward[0] * self.scale, position[0]],
                    [right[1] * self.scale, up[1] * self.scale, forward[1] * self.scale, position[1]],
                    [right[2] * self.scale, up[2] * self.scale, forward[2] * self.scale, position[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=numpy.float32,
            )
        else:
            self.model = self._build_model_matrix(position, self._compose_rotation_matrix())

        glUniformMatrix4fv(get_uniform_location(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

