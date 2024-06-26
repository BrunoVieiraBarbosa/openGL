#version 330 core
#extension GL_ARB_separate_shader_objects : enable

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;
layout (location=2) in vec3 vertexNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location=0) out vec3 fragmentPos;
layout (location=1) out vec2 fragmentTexCoord;
layout (location=2) out vec3 fragmentNormal;


void main()
{
    gl_Position = projection * view * model * vec4(vertexPos, 1.0);
    fragmentPos = vec3(model * vec4(vertexPos, 1.0));
    fragmentTexCoord = vertexTexCoord;
    fragmentNormal = mat3(model) * vertexNormal;
}