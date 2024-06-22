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


vec3 CalculatePointLight(Light light, vec3 cameraPosition, vec3 fragmentPosition, vec3 normal, Material fragmentMaterial, vec2 texCoord, vec3 ambientColor)
{
    //ambient
    vec3 ambientCol = ambientColor * light.color * vec3(texture(fragmentMaterial.diffuse, texCoord));

    vec3 norm = normalize(normal);

    vec3 lightDir = vec3(0.0);

    if (light.type == 0){
        lightDir = normalize(-light.dir);
    } else {
        lightDir = normalize(light.pos - fragmentPosition);
    }

    //diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.color * diff * vec3(texture(fragmentMaterial.diffuse, texCoord));

    //specular
    vec3 viewDir = normalize(cameraPosition - fragmentPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), light.strength);
    vec3 specular = light.color * spec * vec3(texture(fragmentMaterial.specular, texCoord));

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

uniform Material material;
uniform Light lights[MAX_LIGHT_COUNT];
uniform vec3 cameraPos;
uniform vec3 ambient;

layout (location=0) out vec4 color;

void main()
{
    //ambient
    vec3 lightLevel = vec3(0.0, 0.0, 0.0);

    for (int i = 0; i < MAX_LIGHT_COUNT; i++)
    {
        if (lights[i].enable) {
            lightLevel += CalculatePointLight(lights[i], cameraPos, fragmentPos, fragmentNormal, material, fragmentTexCoord, ambient);
        }
    }

    //return pixel color
	color = vec4(lightLevel, 1.0);
}