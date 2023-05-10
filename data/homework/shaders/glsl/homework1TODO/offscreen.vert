#version 450

layout (location = 0) in vec3 inPos;
layout (location = 5)  in uint  inNodeIndex; //for index which FkMatrix to use

layout (binding = 0) uniform UBO 
{
	mat4 lightViewProjection;
} ubo;

//forward kinematics transformation matrices
layout(set=1, binding=0) readonly buffer FkMatrices{
	mat4 fkMatririces[];
}models;


out gl_PerVertex 
{
    vec4 gl_Position;   
};

 
void main()
{
	gl_Position =  ubo.lightViewProjection * models.fkMatririces[inNodeIndex] * vec4(inPos, 1.0);
}