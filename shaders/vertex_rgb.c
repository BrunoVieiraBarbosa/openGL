#version 330 core
#extension GL_ARB_separate_shader_objects : enable

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location=0) out vec3 fragmentPos;
layout (location=1) out vec3 fragmentColor;


void main()
{
    gl_Position = projection * view * model * vec4(vertexPos, 1.0);
    fragmentPos = vec3(model * vec4(vertexPos, 1.0));
    fragmentColor = vertexColor;
}