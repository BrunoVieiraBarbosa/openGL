from core.mesh import Mesh, MeshRGB
from core.light import DirectionalLight, FlashLight, Light, PointLight
from core.core import *
from core.utils import *
from OpenGL.GL import *
import pygame, os, random

def quit_engine(shader, shader_basic, texture, texture2, monkey, cubes):
    pygame.quit()
    glDeleteProgram(shader)
    glDeleteProgram(shader_basic)
    texture.destroy()
    texture2.destroy()
    [x.destroy() for x in monkey]
    [x.destroy() for x in cubes]
    exit(0)


def main():
    size = (1280, 720)
    engine = App(size)
    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos((size[0]//2, size[1]//2))

    engine.add_shader('first', Shader.create_shader('shaders/vertex.c', 'shaders/fragment.c'))
    engine.add_shader('simple', Shader.create_shader('shaders/vertex_rgb.c', 'shaders/fragment_rgb.c'))

    engine.start_()
    #[-0.2, -1.0, -0.3]
    #Objetos da cena
    #light = [Light([engine.shaders[0], engine.shaders[1]], [1.0, 1.0, 1.0], [2, 5, 2], 16, 0, [True, False])]
    #light = [DirectionalLight([engine.shaders[0], engine.shaders[1]], [-0.2, -1.0, -0.3], [1, 1, 1], 16, 0,[True, False])]
    #light = [FlashLight([engine.shaders[0], engine.shaders[1]], [5, 5, 2], [-0.2, -1.0, -0.3], [.8, .8, .8], 16,
    #                                                        0, 15, 20, [True, False])]
    light = [PointLight([engine.shaders[0], engine.shaders[1]], [15, 14, 15], [1, 1, 1], 8, 0, [True, False]),
            FlashLight([engine.shaders[0], engine.shaders[1]], [15, 0, 15], [-0.2, -1.0, -0.3], [.8, .8, .8], 8,
                                                                                1, 15, 20, [True, False]),
            FlashLight([engine.shaders[0], engine.shaders[1]], [15, 7, 15], [1, .2, .1], [.8, .8, .8], 10,
                                                                                1, 15, 20, [True, False])
                                                                                ]

    lampada = [MeshRGB(engine.shaders[1], light[0], color=[1, 1, 1]),
                MeshRGB(engine.shaders[1], light[1], color=[0, 1, 0]),
                MeshRGB(engine.shaders[1], light[2], color=[0, 1, 1])]
    
    red = pygame.Surface((10, 10))
    red.fill((255, 0, 0))

    texture = Material(os.path.join('textures', 'teste.png'), os.path.join('textures', 'teste_specular.png'),
                                            os.path.join('textures', 'teste_specular.png'))
    texture2 = Material(os.path.join('textures', 'box.jpg'), os.path.join('textures', 'box_specular.jpg'), 
                                            os.path.join('textures', 'box_specular.jpg'))

    black = pygame.Surface((10, 10))
    black.fill((0, 0, 0))


    vertices = Mesh.load_obj('obj/nem.obj')
    vertices = Mesh.invert_s_or_t(vertices, 4, 8)
    monkey = [Mesh(engine.shaders[0], texture, [2, 7, 1], vertices)]
    #vertices = Mesh.load_obj('monkey.obj')
    #monkey.append(Mesh(shader, texture3, [8, 8, 1], vertices))
    

    cubes = [Mesh(engine.shaders[0], texture2, [random.randint(x, x*2), random.randint(y, y*2), 0]) for y in range(10) for x in range(10)]
    camera = CameraFirstPerson([-10, 7, 2])
    player = PlayerFirstPerson(size, camera, [engine.shaders[0], engine.shaders[1]])

    cubes_rotate = [random.randint(-5, 5)/10 for _ in cubes]


    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_engine(engine.shaders[0], engine.shaders[1], texture, texture2, monkey, cubes)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_engine(engine.shaders[0], engine.shaders[1], texture, texture2, monkey, cubes)
                    
        #Move camera
        light[0].position[0] -= .02
        light[0].position[1] -= .02
        light[0].position[2] -= .02

        light[1].position[0] -= .02
        light[1].position[1] += .02
        light[1].position[2] -= .02

        light[2].position[0] -= .02
        light[2].position[2] -= .02

        #Update
        [x.update() for x in light]
        [x.rotate_xyz(.5) for x in monkey]
        [x.rotate_xyz(cubes_rotate[i]) for i, x in enumerate(cubes)]
        player.update(1 / max(clock.get_fps(), 1))



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        [x.draw() for x in cubes]
        [x.draw() for x in monkey]
        [x.draw() for x in lampada]

        pygame.display.flip()

        clock.tick(60)

        pygame.display.set_caption(str(round(clock.get_fps(), 0)))
        



if (__name__ == '__main__'):
    main()