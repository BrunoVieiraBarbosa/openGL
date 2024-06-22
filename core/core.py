from core.light import Light
from typing import Union
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import *
import pygame, numpy, pyrr



class Shader:
    @staticmethod
    def create_shader(vertex_file_path, fragment_file_path):
        with open(vertex_file_path, 'r') as file:
            vertex = file.readlines()
        
        with open(fragment_file_path, 'r') as file:
            fragment = file.readlines()
    
        shader = compileProgram(compileShader(vertex, GL_VERTEX_SHADER),
                                compileShader(fragment, GL_FRAGMENT_SHADER))
        return shader



class Material:
    def __init__(self, file_path_diffuse: Union[str, pygame.Surface], 
                        file_path_specular: Union[str, pygame.Surface],
                        file_path_normal: Union[str, pygame.Surface]) -> None:
        def texture(v, image):
            glBindTexture(GL_TEXTURE_2D, v)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glGenerateMipmap(GL_TEXTURE_2D)
            image_w, image_h = image.get_size()
            img_data = pygame.image.tostring(image, 'RGBA')
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_w, image_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        if (type(file_path_diffuse) == str):
            image_diffuse = pygame.image.load(file_path_diffuse).convert_alpha()
        else:
            image_diffuse = file_path_diffuse
        if (type(file_path_specular) == str):
            image_specular = pygame.image.load(file_path_specular).convert_alpha()
        else:
            image_specular = file_path_specular
        if (type(file_path_normal) == str):
            image_normal = pygame.image.load(file_path_normal).convert_alpha()
        else:
            image_normal = file_path_normal

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
        glBindTexture(GL_TEXTURE_2D,self.specular_texture)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D,self.normal_texture)


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
        look_at_matrix = pyrr.matrix44.create_look_at(self.position, self.position+self.forward, up, dtype=numpy.float32)

        for shader in shaders:
            glUseProgram(shader)
            glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, look_at_matrix)
            glUniform3fv(glGetUniformLocation(shader, "cameraPos"), 1, self.position)


class App:
    def __init__(self, size, ambient_color = (.1, .1, .1, 1)) -> None:
        self.size = size
        self.ambient_color = ambient_color
        self.shaders = []
        pygame.display.set_mode(self.size, pygame.OPENGL | pygame.DOUBLEBUF)
        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.ambient_color)
    

    def add_shader(self, name, shader):
        self.shaders.append(shader)
    

    def start_(self):
        glUseProgram(self.shaders[0])

        glUniform3fv(glGetUniformLocation(self.shaders[0],"ambient"), 1, numpy.array(self.ambient_color[:3], dtype=numpy.float32))
        #Diz para o OpenGL colocar na posição 0 e 1 da struct da textura
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.diffuse"), 0)
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.specular"), 1)
        glUniform1i(glGetUniformLocation(self.shaders[0], "material.normal"), 2)

        #Cria a matrix de projeção e envia para o OpenGL
        projection_transform = pyrr.matrix44.create_perspective_projection(45, self.size[0]/self.size[1], .1, 100, numpy.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shaders[0], "projection"), 1, GL_FALSE, projection_transform)

        glUseProgram(self.shaders[1])
        glUniformMatrix4fv(glGetUniformLocation(self.shaders[1], "projection"), 1, GL_FALSE, projection_transform)

        Light.reset_lights([self.shaders[0]])