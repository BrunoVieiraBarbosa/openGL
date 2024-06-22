from typing import Optional
from OpenGL.GL import *
import numpy, pyrr


class Mesh:
    def __init__(self, shader, material, position, vertices: Optional[tuple] = None, faces: Optional[tuple] = None) -> None:
        self.material = material
        self.shader = shader
        self.position = position
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
        vertices, v, vn, vt, f = [], [], [], [], []

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
                #Get vertex position
                f_line = line[1:].strip().split(' ')
                if '//' in line:
                    print('Em desenvolvimento')
                    exit(0)
                    faces = [x.split('//') for x in f_line]
                    faces = [[int(y)-1 for y in x] for x in faces]
                    for x in range(3):
                        [vertices.append(y) for y in v[faces[x][0]]]
                        if vt_count > 0:
                            [vertices.append(y) for y in vt[faces[x][1]]]
                            vt_count -= 1
                        elif vn_count > 0:
                            [vertices.append(y) for y in vn[faces[x][1]]]
                            vn_count -= 1
                    if (len(f_line) == 4):
                        vt_count -= 1
                        vn_count -= 1
                        for x in (0, 2, 3):
                            [vertices.append(y) for y in v[faces[x][0]]]
                            if vt_count > 0:
                                [vertices.append(y) for y in vt[faces[x][1]]]
                                [vertices.append(y) for y in range(3)]
                            elif vn_count > 0:
                                [vertices.append(y) for y in range(2)]
                                [vertices.append(y) for y in vn[faces[x][1]]]

                elif '/' in line:
                    faces = [x.split('/') for x in f_line]
                    faces = [[int(y)-1 for y in x] for x in faces]
                    for x in range(3):
                        [vertices.append(y) for y in v[faces[x][0]]]
                        [vertices.append(y) for y in vt[faces[x][1]]]
                        [vertices.append(y) for y in vn[faces[x][2]]]
                    if (len(f_line) == 4):
                        for x in (0, 2, 3):
                            [vertices.append(y) for y in v[faces[x][0]]]
                            [vertices.append(y) for y in vt[faces[x][1]]]
                            [vertices.append(y) for y in vn[faces[x][2]]]

        print(f'modelo carregado: {file_name}')
        return vertices


    @staticmethod
    def invert_s_or_t(vertices, st_pos, vertice_size):
        vertices_len = len(vertices) // vertice_size
        for x in range(vertices_len):
            vertices[x * vertice_size + st_pos] = 1 - vertices[x * vertice_size + st_pos]
        return vertices


    @staticmethod
    def invert_st(vertices, s_pos, t_pos, vertice_size):
        return Mesh.invert_s_or_t(Mesh.invert_s_or_t(vertices, s_pos, vertice_size), t_pos, vertice_size)


    def translate(self, model):
        self.model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_translation(vec=numpy.array(self.position), dtype=numpy.float32))
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
            self.model = pyrr.matrix44.create_from_translation(vec=numpy.array(self.position), dtype=numpy.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))



class MeshRGB:
    def __init__(self, shader, position, vertices=None, color = [1, 1, 1]) -> None:
        self.shader = shader
        self.position = position
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


    def draw(self):
        glUseProgram(self.shader)
        if type(self.model) == type(None):
            self.model = pyrr.matrix44.create_from_translation(vec=numpy.array(self.position.position), dtype=numpy.float32)

        self.model = pyrr.matrix44.multiply(self.identity, pyrr.matrix44.create_from_translation(vec=numpy.array(self.position.position), dtype=numpy.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, self.model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

