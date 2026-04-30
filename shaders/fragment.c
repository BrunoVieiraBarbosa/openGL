#version 330 core
#extension GL_ARB_separate_shader_objects : enable
#define MAX_LIGHT_COUNT 8

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D normal;
    float shininess;
};


struct Light {
    //Type Light
    //0 - Directional Light
    //1 - Point lights
    //2 - Spotlight - Flashlight

    int type;
    
    vec3 pos;
    vec3 color;

    //Directional Light
    vec3 dir;

    //Flashlight
    float cutOff;
    float outerCutOff;

    float strength;

    float constant;
    float linear;
    float quadratic;

    bool enable;
};


vec3 CalculatePointLight(Light light, vec3 cameraPosition, vec3 fragmentPosition, vec3 normal, vec3 diffuseColor, vec3 specularColor, vec3 ambientColor)
{
    //ambient
    vec3 ambientCol = ambientColor * light.color * diffuseColor;

    vec3 norm = normalize(normal);

    vec3 lightDir = vec3(0.0);

    if (light.type == 0){
        lightDir = normalize(-light.dir);
    } else {
        lightDir = normalize(light.pos - fragmentPosition);
    }

    //diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.color * diff * diffuseColor;

    //specular
    vec3 viewDir = normalize(cameraPosition - fragmentPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), light.strength);
    vec3 specular = light.color * spec * specularColor;

    if (light.type == 1){
        float distance = length(light.pos - fragmentPosition);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
    
        ambientCol *= attenuation;
        diffuse *= attenuation;
        specular *= attenuation;
    }

    if(light.type == 2){
        float theta = dot(lightDir, normalize(-light.dir));
        float epsilon = light.cutOff - light.outerCutOff;
        float intensity = 1 - clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        diffuse *= intensity;
        specular *= intensity;
    }

    vec3 result = ambientCol + diffuse + specular;
    
    return result;
}


layout (location=0) in vec3 fragmentPos;
layout (location=1) in vec2 fragmentTexCoord;
layout (location=2) in vec3 fragmentNormal;
layout (location=3) in vec3 fragmentTangent;
layout (location=4) in vec4 fragmentLightSpacePos;

uniform Material material;
uniform Light lights[MAX_LIGHT_COUNT];
uniform vec3 cameraPos;
uniform vec3 ambient;
uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
uniform sampler2D shadowMap;
uniform bool shadowsEnabled;
uniform int shadowDebugMode;
uniform float shadowStrength;
uniform bool shadowDisableFog;

layout (location=0) out vec4 color;

float CalculateShadow(vec4 lightSpacePosition, vec3 normal, vec3 lightDir)
{
    vec3 projectedCoords = lightSpacePosition.xyz / max(lightSpacePosition.w, 0.0001);
    projectedCoords = projectedCoords * 0.5 + 0.5;

    if (projectedCoords.z > 1.0 || projectedCoords.x < 0.0 || projectedCoords.x > 1.0 || projectedCoords.y < 0.0 || projectedCoords.y > 1.0) {
        return 0.0;
    }

    float bias = max(0.0055 * (1.0 - dot(normalize(normal), normalize(lightDir))), 0.0012);
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float closestDepth = texture(shadowMap, projectedCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (projectedCoords.z - bias) > closestDepth ? 1.0 : 0.0;
        }
    }

    return shadow / 9.0;
}

vec3 ProjectToLightSpace(vec4 lightSpacePosition)
{
    vec3 projectedCoords = lightSpacePosition.xyz / max(lightSpacePosition.w, 0.0001);
    return projectedCoords * 0.5 + 0.5;
}

void main()
{
    vec3 tangent = normalize(fragmentTangent - dot(fragmentTangent, fragmentNormal) * fragmentNormal);
    vec3 bitangent = normalize(cross(fragmentNormal, tangent));
    mat3 tbn = mat3(tangent, bitangent, normalize(fragmentNormal));

    vec3 sampledNormal = texture(material.normal, fragmentTexCoord).xyz * 2.0 - 1.0;
    vec3 worldNormal = normalize(tbn * sampledNormal);
    vec3 diffuseColor = texture(material.diffuse, fragmentTexCoord).rgb;
    vec3 specularColor = texture(material.specular, fragmentTexCoord).rgb;

    //ambient
    vec3 lightLevel = vec3(0.0, 0.0, 0.0);
    float directionalShadow = 0.0;
    bool shadowComputed = false;
    vec3 lightSpaceCoords = ProjectToLightSpace(fragmentLightSpacePos);

    for (int i = 0; i < MAX_LIGHT_COUNT; i++)
    {
        if (lights[i].enable) {
            vec3 shadowNormal = worldNormal;
            vec3 currentLightDir = lights[i].type == 0 ? normalize(-lights[i].dir) : normalize(lights[i].pos - fragmentPos);
            float shadowFactor = 0.0;
            if (!shadowComputed && shadowsEnabled && lights[i].type == 0) {
                directionalShadow = CalculateShadow(fragmentLightSpacePos, shadowNormal, currentLightDir);
                directionalShadow = clamp(directionalShadow * shadowStrength, 0.0, 1.0);
                shadowComputed = true;
            }
            lightLevel += CalculatePointLight(
                lights[i],
                cameraPos,
                fragmentPos,
                worldNormal,
                diffuseColor,
                specularColor,
                ambient
            ) * (lights[i].type == 0 ? (1.0 - directionalShadow) : 1.0);
        }
    }

    if (shadowDebugMode == 1) {
        color = vec4(vec3(directionalShadow), 1.0);
        return;
    }

    if (shadowDebugMode == 2) {
        float sampledDepth = texture(shadowMap, lightSpaceCoords.xy).r;
        color = vec4(vec3(sampledDepth), 1.0);
        return;
    }

    if (shadowDebugMode == 3) {
        vec3 coverage = vec3(
            lightSpaceCoords.x,
            lightSpaceCoords.y,
            clamp(lightSpaceCoords.z, 0.0, 1.0)
        );
        color = vec4(coverage, 1.0);
        return;
    }

    if (!shadowDisableFog) {
        float fogDistance = distance(cameraPos, fragmentPos);
        float fogFactor = clamp((fogFar - fogDistance) / max(fogFar - fogNear, 0.001), 0.0, 1.0);
        lightLevel = mix(fogColor, lightLevel, fogFactor);
    }

    //return pixel color
	color = vec4(lightLevel, 1.0);
}
