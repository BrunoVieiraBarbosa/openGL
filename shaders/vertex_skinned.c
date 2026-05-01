#version 330 core
#extension GL_ARB_separate_shader_objects : enable

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;
layout (location=2) in vec3 vertexNormal;
layout (location=3) in vec3 vertexTangent;
layout (location=4) in vec4 vertexJointIds;
layout (location=5) in vec4 vertexJointWeights;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 meshBindMatrix;
uniform mat4 postSkinningTransform;
uniform int boneCount;
uniform mat4 boneMatrices[128];

layout (location=0) out vec3 fragmentPos;
layout (location=1) out vec2 fragmentTexCoord;
layout (location=2) out vec3 fragmentNormal;
layout (location=3) out vec3 fragmentTangent;

mat4 sampleSkinMatrix()
{
    ivec4 joints = ivec4(vertexJointIds);
    vec4 weights = vertexJointWeights;
    mat4 skinMatrix = mat4(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < 4; i++) {
        int jointIndex = joints[i];
        if (jointIndex >= 0 && jointIndex < boneCount && weights[i] > 0.0) {
            skinMatrix += boneMatrices[jointIndex] * weights[i];
            totalWeight += weights[i];
        }
    }

    if (totalWeight <= 0.0) {
        return mat4(1.0);
    }
    return skinMatrix;
}

void main()
{
    mat4 skinMatrix = sampleSkinMatrix();
    mat4 skinTransform = postSkinningTransform * meshBindMatrix * skinMatrix;
    vec4 skinnedPosition = skinTransform * vec4(vertexPos, 1.0);
    vec3 skinnedNormal = mat3(skinTransform) * vertexNormal;
    vec3 skinnedTangent = mat3(skinTransform) * vertexTangent;

    gl_Position = projection * view * model * skinnedPosition;
    fragmentPos = vec3(model * skinnedPosition);
    fragmentTexCoord = vertexTexCoord;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    fragmentNormal = normalize(normalMatrix * skinnedNormal);
    fragmentTangent = normalize(normalMatrix * skinnedTangent);
}
