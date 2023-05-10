#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	float depth = texture(samplerColor, inUV).r;
	outFragColor = vec4(vec3(depth), 1.0);
}