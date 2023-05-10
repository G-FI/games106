#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inTangent;
layout (location = 5)  in uint  inNodeIndex; //for index which FkMatrix to use

layout (set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
} uboScene;

layout (set = 3, binding = 0) uniform  UBO 
{
	mat4 lightViewProjection;
} ubo;


layout(push_constant) uniform PushConsts {
	mat4 model;
} primitive;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;
layout (location = 5) out uint outNodeIndex;
layout (location = 6) out vec4 outTangent;
layout (location = 7) out vec4 outPos;

//forward kinematics transformation matrices
layout(set=2, binding=0) readonly buffer FkMatrices{
	mat4 fkMatririces[];
};

void main() 
{

	
	outNodeIndex = inNodeIndex;

	//transformation matrix
	mat4 skeletonMat = fkMatririces[inNodeIndex];

	vec4 worldPos = skeletonMat * vec4(inPos,1.0);
	gl_Position = uboScene.projection * uboScene.view  * worldPos;
	outNormal = mat3(skeletonMat) * inNormal;

	outColor = inColor.rgb;
	outUV = inUV;	
	outTangent = inTangent;

	vec3 pos = worldPos.xyz / worldPos.w;

	outPos = worldPos;
	outViewVec = uboScene.viewPos.xyz - pos;
	outLightVec = uboScene.lightPos.xzy - pos;
}