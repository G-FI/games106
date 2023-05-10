#version 450
layout (set = 1, binding = 0) uniform sampler2D albedoMap;
layout (set = 1, binding = 1) uniform sampler2D normalMap;
layout (set = 1, binding = 2) uniform sampler2D aoMap;
layout (set = 1, binding = 3) uniform sampler2D metallicRoughnessMap;
layout (set = 1, binding = 4) uniform sampler2D emissiveMap;

layout (set = 3, binding = 0) uniform  UBO 
{
	mat4 lightViewProjection;
} ubo;

layout (set = 3, binding = 1) uniform sampler2D shadowMap;



layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 5) flat in uint inNodeIndex;
layout (location = 6) in vec4 inTangent;
layout (location = 7) in vec4 inPos;
layout (location = 8) in vec4 inTmp;

layout (location = 0) out vec4 outFragColor;


const float PI = 3.14159265359;
#define  ALBEDO  pow(texture(albedoMap, inUV).rgb, vec3(2.2))


//void main() 
//{
//	vec4 color = texture(albedoMap, inUV) * vec4(inColor, 1.0);
//
//	vec3 N = normalize(inNormal);
//	vec3 L = normalize(inLightVec);
//	vec3 V = normalize(inViewVec);
//	vec3 R = reflect(L, N);
//	vec3 diffuse = max(dot(N, L), 0.15) * inColor;
//	vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(0.75);
//	outFragColor = vec4(diffuse * color.rgb + specular, 1.0);		
//}
//

vec3 materialcolor()
{
	return texture(albedoMap, inUV).rgb;
}

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, float metallic)
{
	vec3 F0 = mix(vec3(0.04), materialcolor(), metallic); // * material.specular
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); 
	return F;    
}

// Specular BRDF composition --------------------------------------------

vec3 specularContribution(vec3 L, vec3 V, vec3 N, float metallic, float roughness)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float dotLH = clamp(dot(L, H), 0.0, 1.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0);

	vec3 color = vec3(0.0);

	if (dotNL > 0.0)
	{

		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness); 
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, metallic);

		vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);

		vec3 Ks = F;
		vec3 Kd = (vec3(1.0)-Ks) * (1.0 - metallic);

		color += (Kd * ALBEDO / PI + spec) * dotNL;
	}

	return color;
}


vec3 calculateNormal()
{
	vec3 tangentNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);
	return normalize(TBN * tangentNormal);
}

float getShadow(vec4 shadowCoord){
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadowMap, shadowCoord.st).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z + 0.01) 
		{
			shadow = 0;
		}
	}
	return shadow;
}

void main()
{		  
	vec3 N = calculateNormal();
	vec3 V = normalize(inViewVec);
	vec3 L = normalize(inLightVec);

	float metallic = texture(metallicRoughnessMap, inUV).r;
	float roughness = texture(metallicRoughnessMap, inUV).g;

	

	// Specular contribution
	vec3 Lo = vec3(0.0);
	Lo += specularContribution(L, V, N, metallic, roughness);

	// Combine with ambient
	vec3 color = materialcolor() * 0.02 * texture(aoMap, inUV).rrr;
	color += Lo;

	//emissive
	color += texture(emissiveMap, inUV).rgb;


	//Tone mapping
	color = color / (color + vec3(1.0));

	// Gamma correct
	color = pow(color, vec3(0.4545));
	

	outFragColor = vec4(color, 1.0);

	vec4 pos = ubo.lightViewProjection * inPos;
	pos /= pos.w;	
	//show the depth of every shading point from light view

	//outFragColor = vec4(inTmp.z);
	//outFragColor = vec4(vec3(inTmp.z)/inTmp.w, 1.0); //depth from light view


	float shadow = getShadow(pos);
	outFragColor = vec4(vec3(shadow), 1.0);
}