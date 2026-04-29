import hashlib
from pathlib import Path
from typing import Optional

from OpenGL.GL import *
import numpy, pyrr


class Mesh:
    CACHE_DIR = Path(".cache") / "obj"
    PREPARED_CACHE_DIR = Path(".cache") / "obj_prepared"

    def __init__(self, shader, material, position, vertices: Optional[tuple] = None, faces: Optional[tuple] = None, scale=1.0) -> None:
        self.material = material
        self.shader = shader
        self.position = position
        self.scale = scale
        self.rotation = [0, 0, 0]
        self.identity = pyrr.matrix44.create_identity(dtype=numpy.float32)
        self.model = None
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

        self.vertex_count = len(self.vertices)//8
        self.vertices = numpy.array(self.vertices, dtype=numpy.float32)
        self.local_bounds = self._compute_local_bounds()

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

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
    def load_obj_prepared(file_name, invert_texcoord=True, st_pos=4, vertex_size=8, target_size=3.5):
        source_path = Path(file_name)
        cache_file = Mesh._get_prepared_cache_path(
            source_path,
            invert_texcoord=invert_texcoord,
            st_pos=st_pos,
            vertex_size=vertex_size,
            target_size=target_size,
        )
        source_stat = source_path.stat()

        cached_vertices = Mesh._load_cached_vertices(cache_file, source_stat)
        if cached_vertices is not None:
            print(f'cache preparado carregado: {file_name}')
            return cached_vertices

        vertices = Mesh.load_obj(file_name)
        if invert_texcoord:
            vertices = Mesh.invert_s_or_t(vertices, st_pos, vertex_size)
        vertices = Mesh.normalize_vertices(vertices, vertex_size=vertex_size, target_size=target_size)
        Mesh._store_cached_vertices(cache_file, source_stat, vertices)
        return vertices


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
    def _store_cached_vertices(cache_file: Path, source_stat, vertices):
        numpy.savez_compressed(
            cache_file,
            vertices=numpy.array(vertices, dtype=numpy.float32),
            mtime_ns=numpy.array([source_stat.st_mtime_ns], dtype=numpy.int64),
            file_size=numpy.array([source_stat.st_size], dtype=numpy.int64),
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


    def _compute_local_bounds(self):
        positions = self.vertices.reshape(-1, 8)[:, :3]
        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0)
        return minimum, maximum


    def get_world_bounds(self):
        minimum, maximum = self.local_bounds
        scaled_min = minimum * self.scale
        scaled_max = maximum * self.scale
        position = numpy.array(self.position, dtype=numpy.float32)
        return scaled_min + position, scaled_max + position


    def _build_model_matrix(self, rotation_matrix):
        scale_vector = numpy.array([self.scale, self.scale, self.scale], dtype=numpy.float32)
        scale_matrix = pyrr.matrix44.create_from_scale(scale_vector, dtype=numpy.float32)
        translation_matrix = pyrr.matrix44.create_from_translation(vec=numpy.array(self.position), dtype=numpy.float32)
        model = pyrr.matrix44.multiply(scale_matrix, rotation_matrix)
        return pyrr.matrix44.multiply(model, translation_matrix)


    def translate(self, model):
        self.model = self._build_model_matrix(model)
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.model)


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
        model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_x_rotation(theta=numpy.radians(self.rotation[0]), dtype=numpy.float32))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_y_rotation(theta=numpy.radians(self.rotation[1]), dtype=numpy.float32))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_z_rotation(theta=numpy.radians(self.rotation[2]), dtype=numpy.float32))
        self.translate(model)


    def draw(self):
        glUseProgram(self.shader)
        self.material.use()
        if type(self.model) == type(None):
            self.model = self._build_model_matrix(self.identity)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))



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
            scale_matrix = pyrr.matrix44.create_from_scale(
                numpy.array([self.scale, self.scale, self.scale], dtype=numpy.float32),
                dtype=numpy.float32,
            )
            translation_matrix = pyrr.matrix44.create_from_translation(vec=position, dtype=numpy.float32)
            self.model = pyrr.matrix44.multiply(scale_matrix, translation_matrix)

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

