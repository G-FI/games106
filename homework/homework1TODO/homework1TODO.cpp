/*
* Vulkan Example - glTF scene loading and rendering
*
* Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

#pragma once

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"

#include "vulkanexamplebase.h"

#define SHADOW_DIM 2048

#define ENABLE_VALIDATION true

// Contains everything required to render a glTF model in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice* vulkanDevice;
	VkQueue copyQueue;

	// The vertex layout for the samples' model
	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec4 color;
		glm::vec4 tangent;
		uint32_t  nodeIndex;
	};

	// Single vertex buffer for all primitives
	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required for this sample
	struct Node;

	// A primitive contains the data for a single draw call
	struct Primitive {
		uint32_t firstIndex;
		uint32_t indexCount;
		int32_t materialIndex;
	};

	//Animation
	struct Skeleton {
		std::vector<glm::mat4> fkMatrices;
		vks::Buffer ssbo;
		VkDescriptorSet descriptorSet;
	};

	struct AnimationChannel {
		uint32_t samplerIndex;
		std::string path;
		Node* node;
	};

	struct AnimationSampler {
		std::string interpolation;
		std::vector<float> inputs;  //key frames
		std::vector<glm::vec4> outputs;  //transformation data of corresponding key frames
	};

	struct Animation {
		std::string name;
		std::vector<AnimationChannel> channels;
		std::vector<AnimationSampler> samplers;
		float                         start = std::numeric_limits<float>::max();
		float                         end = std::numeric_limits<float>::min();
		float                         currentTime = 0.0f;
	};


	// Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
	struct Mesh {
		std::vector<Primitive> primitives;
	};

	// A node represents an object in the glTF scene graph
	struct Node {
		uint32_t index;
		Node* parent;
		std::vector<Node*> children;
		Mesh mesh;
		glm::mat4 matrix;
		glm::vec3 translation{};
		glm::quat rotation{};
		glm::vec3 scale{1.0f};

		glm::mat4 getLocalTransformation() {
			return glm::translate(glm::mat4(1.0f), translation)* glm::mat4(rotation)* glm::scale(glm::mat4(1.0f), scale)* matrix;
		}

		~Node() {
			for (auto& child : children) {
				delete child;
			}
		}
	};

	// A glTF material stores information in e.g. the texture that is attached to it and colors
	struct Material {
		glm::vec4 baseColorFactor = glm::vec4(1.0f);
		int baseColorTextureIndex;


		int metallicRoughnessTextureIndex;
		float	metallicFactor = 1.f;
		float roughnessFactor = 1.f;

		int normalTexureIndex;

		int occlusionTextureIndex;

		int emissiveTextureIndex = -1;
		glm::vec3 emissiveFactor = glm::vec3(1.f);

		VkDescriptorSet descriptorSet;
	};

	// Contains the texture for a single glTF image
	// Images may be reused by texture objects and are as such separated
	struct Image {

		vks::Texture2D texture;
		// We also store (and create) a descriptor set that's used to access this texture from the fragment shader
		VkDescriptorSet descriptorSet;
	};

	// A glTF texture stores a reference to the image and a sampler
	// In this sample, we are only interested in the image
	struct Texture {
		int32_t imageIndex;
	};
	uint32_t nodeSize;
	/*
		Model data
	*/
	std::vector<Image> images;
	std::vector<Texture> textures;
	std::vector<Material> materials;
	std::vector<Node*> nodes;
	std::vector<Animation> animations;
	


	Skeleton skeleton;

	~VulkanglTFModel()
	{
		for (auto node : nodes) {
			delete node;
		}
		// Release all Vulkan resources allocated for the model
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
		for (Image image : images) {
			vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
			vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
			vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
			vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
		}

		skeleton.ssbo.destroy();
		
	}

	// Helper functions for locating glTF nodes

	Node* findNode(Node* parent, uint32_t index)
	{
		Node* nodeFound = nullptr;
		if (parent->index == index)
		{
			return parent;
		}
		for (auto& child : parent->children)
		{
			nodeFound = findNode(child, index);
			if (nodeFound)
			{
				break;
			}
		}
		return nodeFound;
	}

	Node* nodeFromIndex(uint32_t index)
	{
		Node* nodeFound = nullptr;
		for (auto& node : nodes)
		{
			nodeFound = findNode(node, index);
			if (nodeFound)
			{
				break;
			}
		}
		return nodeFound;
	}


	/*
		glTF loading functions

		The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own structure
	*/

	void loadImages(tinygltf::Model& input)
	{
		// Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
		// loading them from disk, we fetch them from the glTF loader and upload the buffers
		images.resize(input.images.size());
		for (size_t i = 0; i < input.images.size(); i++) {
			tinygltf::Image& glTFImage = input.images[i];
			// Get the image data from the glTF loader
			unsigned char* buffer = nullptr;
			VkDeviceSize bufferSize = 0;
			bool deleteBuffer = false;
			// We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
			if (glTFImage.component == 3) {
				bufferSize = glTFImage.width * glTFImage.height * 4;
				buffer = new unsigned char[bufferSize];
				unsigned char* rgba = buffer;
				unsigned char* rgb = &glTFImage.image[0];
				for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i) {
					memcpy(rgba, rgb, sizeof(unsigned char) * 3);
					rgba += 4;
					rgb += 3;
				}
				deleteBuffer = true;
			}
			else {
				buffer = &glTFImage.image[0];
				bufferSize = glTFImage.image.size();
			}
			// Load texture from image buffer
			images[i].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, glTFImage.width, glTFImage.height, vulkanDevice, copyQueue);
			if (deleteBuffer) {
				delete[] buffer;
			}
		}
	}

	void loadTextures(tinygltf::Model& input)
	{
		textures.resize(input.textures.size());
		for (size_t i = 0; i < input.textures.size(); i++) {
			textures[i].imageIndex = input.textures[i].source;
		}
	}

	void loadMaterials(tinygltf::Model& input)
	{
		materials.resize(input.materials.size());
		for (size_t i = 0; i < input.materials.size(); i++) {
			// We only read the most basic properties required for our sample
			tinygltf::Material glTFMaterial = input.materials[i];
			// Get the base color factor
			//if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end()) {
			//	materials[i].baseColorFactor = glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
			//}
			//// Get base color texture index
			//if (glTFMaterial.values.find("baseColorTexture") != glTFMaterial.values.end()) {
			//	materials[i].baseColorTextureIndex = glTFMaterial.values["baseColorTexture"].TextureIndex();
			//}
			materials[i].baseColorTextureIndex = glTFMaterial.pbrMetallicRoughness.baseColorTexture.index;
			materials[i].metallicRoughnessTextureIndex = glTFMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
			materials[i].normalTexureIndex = glTFMaterial.normalTexture.index;
			materials[i].occlusionTextureIndex = glTFMaterial.occlusionTexture.index;
			materials[i].emissiveTextureIndex = glTFMaterial.emissiveTexture.index;

		}
	}

	void loadNode(const tinygltf::Node& inputNode, const tinygltf::Model& input, VulkanglTFModel::Node* parent, int nodeIndex, std::vector<uint32_t>& indexBuffer, std::vector<VulkanglTFModel::Vertex>& vertexBuffer)
	{
		VulkanglTFModel::Node* node = new VulkanglTFModel::Node{};
		node->matrix = glm::mat4(1.0f);
		node->index = nodeIndex; //added
		node->parent = parent;

		// Get the local node matrix
		// It's either made up from translation, rotation, scale or a 4x4 matrix
		if (inputNode.translation.size() == 3) {
			node->translation = glm::make_vec3(inputNode.translation.data());
		}
		if (inputNode.rotation.size() == 4) {
			node->rotation = glm::make_quat(inputNode.rotation.data());
		}
		if (inputNode.scale.size() == 3) {
			node->scale = glm::make_vec3(inputNode.scale.data());
		}
		if (inputNode.matrix.size() == 16) {
			node->matrix = glm::make_mat4x4(inputNode.matrix.data());
		};

		// Load node's children
		if (inputNode.children.size() > 0) {
			for (size_t i = 0; i < inputNode.children.size(); i++) {
				loadNode(input.nodes[inputNode.children[i]], input , node, inputNode.children[i], indexBuffer, vertexBuffer);
			}
		}

		// If the node contains mesh data, we load vertices and indices from the buffers
		// In glTF this is done via accessors and buffer views
		if (inputNode.mesh > -1) {
			const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
			// Iterate through all primitives of this node's mesh
			for (size_t i = 0; i < mesh.primitives.size(); i++) {
				const tinygltf::Primitive& glTFPrimitive = mesh.primitives[i];
				uint32_t firstIndex = static_cast<uint32_t>(indexBuffer.size());
				uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
				uint32_t indexCount = 0;
				// Vertices
				{
					const float* positionBuffer = nullptr;
					const float* normalsBuffer = nullptr;
					const float* texCoordsBuffer = nullptr;
					const float* tangentBufferer = nullptr;
					size_t vertexCount = 0;

					// Get buffer data for vertex positions
					if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						positionBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
						vertexCount = accessor.count;
					}
					// Get buffer data for vertex normals
					if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						normalsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}
					// Get buffer data for vertex texture coordinates
					// glTF supports multiple sets, we only load the first one
					if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						texCoordsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}
					if (glTFPrimitive.attributes.find("TANGENT") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						tangentBufferer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}

					// Append data to model's vertex buffer
					for (size_t v = 0; v < vertexCount; v++) {
						Vertex vert{};
						vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
						vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
						vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
						vert.color = glm::vec4(1.0f);
						vert.tangent = tangentBufferer ? glm::make_vec4(&tangentBufferer[v * 4]) : glm::vec4(0.f);
						vert.nodeIndex = nodeIndex; //used for index fk matrixs
						vertexBuffer.push_back(vert);
					}
				}
				// Indices
				{
					const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.indices];
					const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

					indexCount += static_cast<uint32_t>(accessor.count);

					// glTF supports different component types of indices
					switch (accessor.componentType) {
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
						const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
						const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
						const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					default:
						std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
						return;
					}
				}
				Primitive primitive{};
				primitive.firstIndex = firstIndex;
				primitive.indexCount = indexCount;
				primitive.materialIndex = glTFPrimitive.material;
				node->mesh.primitives.push_back(primitive);
			}
		}

		if (parent) {
			parent->children.push_back(node);
		}
		else {
			nodes.push_back(node);
		}
	}

	void loadAnimation(tinygltf::Model& input) {
		animations.resize(input.animations.size());
		for (int i = 0 ; i < animations.size(); ++i) {
			tinygltf::Animation& srcAnimation = input.animations[i];
			Animation& dstAnimation = animations[i];

			dstAnimation.name = srcAnimation.name;
			

			//Channels
			dstAnimation.channels.resize(srcAnimation.channels.size());
			for(int j = 0; j <srcAnimation.channels.size(); ++j){
				tinygltf::AnimationChannel& srcChannel = srcAnimation.channels[j];
				AnimationChannel& dstChannel = animations[i].channels[j];
				dstChannel.path = srcChannel.target_path;
				dstChannel.samplerIndex = srcChannel.sampler;
				dstChannel.node = nodeFromIndex(srcChannel.target_node); //find Node* from nodes tree by index
			}

			//Samplers
			dstAnimation.samplers.resize(srcAnimation.samplers.size());
			for (int j = 0; j < srcAnimation.samplers.size(); ++j) {
				tinygltf::AnimationSampler& srcSampler = srcAnimation.samplers[j];
				AnimationSampler& dstSampler = dstAnimation.samplers[j];
				dstSampler.interpolation = srcSampler.interpolation;

				// Read sampler keyframe input time values
				{
					const tinygltf::Accessor& accessor = input.accessors[srcSampler.input];
					const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

					const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
					const float* buf = static_cast<const float*>(dataPtr);
					dstSampler.inputs.resize(accessor.count);
					for (size_t index = 0; index < accessor.count; index++)
					{
						dstSampler.inputs[index] = buf[index];
					}
					// Adjust animation's start and end times
					for (auto input : dstSampler.inputs)
					{
						if (input < dstAnimation.start)
						{
							dstAnimation.start = input;
						};
						if (input > dstAnimation.end)
						{
							dstAnimation.end = input;
						}
					}
				}

				// Read sampler keyframe output translate/rotate/scale values
				{
					const tinygltf::Accessor& accessor = input.accessors[srcSampler.output];
					const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];
					const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
					switch (accessor.type)
					{
					case TINYGLTF_TYPE_VEC3: {
						const glm::vec3* buf = static_cast<const glm::vec3*>(dataPtr);

						dstSampler.outputs.resize(accessor.count);
						for (size_t index = 0; index < accessor.count; index++)
						{
							dstSampler.outputs[index]  = glm::vec4(buf[index], 0.0f);// from pointer to value
						}
						break;
					}
					case TINYGLTF_TYPE_VEC4: {
						const glm::vec4* buf = static_cast<const glm::vec4*>(dataPtr);
						dstSampler.outputs.resize(accessor.count);
						for (size_t index = 0; index < accessor.count; index++)
						{
							dstSampler.outputs[index] = buf[index]; // from pointer to value
						}
						break;
					}
					default: {
						std::cout << "unknown type" << std::endl;
						break;
					}
					}
				}
				
			}
		
		}



		//fk matrices
		skeleton.fkMatrices.resize(input.nodes.size(), glm::mat4(1.f));

		updateFKmatrices(nodes[0], glm::mat4(1.f)); 

		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&skeleton.ssbo,
			sizeof(glm::mat4) * skeleton.fkMatrices.size(),
			skeleton.fkMatrices.data()));
		VK_CHECK_RESULT(skeleton.ssbo.map()); 
	}

	//update fk matirces
	void updateFKmatrices(Node* node, const glm::mat4& parentTransformation) {
		if (node == nullptr) {
			return;
		}
		const glm::mat4& currentTransformation = parentTransformation * node->getLocalTransformation(); 

		skeleton.fkMatrices[node->index] = currentTransformation;

		for (auto&& child : node->children) {
			updateFKmatrices(child, currentTransformation);
		}
	}

	void updateAnimation(float deltaTime) {
		Animation& animation = animations[0];

		animation.currentTime += deltaTime;

		if (animation.currentTime > animation.end) {
			animation.currentTime -= (animation.end - animation.start);
		}

		//interpolate to get the local transformation of each node in the current frame.

		for (auto && channel: animation.channels) {
			auto&& sampler = animation.samplers[channel.samplerIndex];

			float alpha = 0.f;
			int i = 0;
			for (int n = sampler.inputs.size(); i < n - 1; ++i) {
				if (animation.currentTime >= sampler.inputs[i] && animation.currentTime <= sampler.inputs[i + 1]) {
					alpha = (animation.currentTime - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
					break;
				}
			}

			if (sampler.interpolation == "LINEAR") {
				if (channel.path == "translation") {
					channel.node->translation = glm::mix(sampler.outputs[i], sampler.outputs[i + 1], alpha);
				}
				else if (channel.path == "rotation") {
					glm::quat q1;
					q1.x = sampler.outputs[i].x;
					q1.y = sampler.outputs[i].y;
					q1.z = sampler.outputs[i].z;
					q1.w = sampler.outputs[i].w;

					glm::quat q2;
					q2.x = sampler.outputs[i + 1].x;
					q2.y = sampler.outputs[i + 1].y;
					q2.z = sampler.outputs[i + 1].z;
					q2.w = sampler.outputs[i + 1].w;

					channel.node->rotation = glm::normalize(glm::slerp(q1, q2, alpha));
				}
				else if (channel.path == "scale") {
					channel.node->scale = glm::mix(sampler.outputs[i], sampler.outputs[i + 1], alpha);
				}
			}
			else {
				throw std::runtime_error("interpolation not support");
			}

		}

		updateFKmatrices(nodes[0], glm::mat4(1.f));
		skeleton.ssbo.copyTo(skeleton.fkMatrices.data(), skeleton.fkMatrices.size() * sizeof(glm::mat4));
	}
	/*
		glTF rendering functions
	*/

	// Draw a single node including child nodes (if present)
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node)
	{
		if (node->mesh.primitives.size() > 0) {
			// Pass the node's matrix via push constants
			// Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
			/*glm::mat4 nodeMatrix = node->matrix;
			VulkanglTFModel::Node* currentParent = node->parent;
			while (currentParent) {
				nodeMatrix = currentParent->matrix * nodeMatrix;
				currentParent = currentParent->parent;
			}*/
			// Pass the final matrix to the vertex shader using push constants
			//vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &nodeMatrix);

			for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
				if (primitive.indexCount > 0) {
					// Get the texture index for this primitive
					//VulkanglTFModel::Texture texture = textures[materials[primitive.materialIndex].baseColorTextureIndex];
					// Bind the descriptor for the current primitive's texture
					//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &images[texture.imageIndex].descriptorSet, 0, nullptr);
					vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materials[primitive.materialIndex].descriptorSet, 0, nullptr);
					vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
				}
			}
		}
		for (auto& child : node->children) {
			drawNode(commandBuffer, pipelineLayout, child);
		}
	}
	void drawNodeShadow(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node)
	{
		if (node->mesh.primitives.size() > 0) {

			for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
				if (primitive.indexCount > 0) {
					// Get the texture index for this primitive
					//VulkanglTFModel::Texture texture = textures[materials[primitive.materialIndex].baseColorTextureIndex];
					// Bind the descriptor for the current primitive's texture
					//vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &images[texture.imageIndex].descriptorSet, 0, nullptr);
			
					// Pass the final matrix to the vertex shader using push constants
					//vkCmdPushConstants(commandBuffer, pipelineLayout,VK_SHADER_STAGE_VERTEX_BIT,  0, sizeof(glm::mat4), &skeleton.fkMatrices[node->index]);
					
					vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
				}
			}
		}
		for (auto& child : node->children) {
			drawNodeShadow(commandBuffer, pipelineLayout, child);
		}
	}

	// Draw the glTF scene starting at the top-level-nodes
	void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout)
	{
		// All vertices and indices are stored in single buffers, so we only need to bind once
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &skeleton.descriptorSet, 0, nullptr);
		// Render all nodes at top-level
		for (auto& node : nodes) {
			drawNode(commandBuffer, pipelineLayout, node);
		}
	}

	//generate depth map
	void drawShadow(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) {
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		//shader storage buffer中存储每个node的global model matrix
		vkCmdBindDescriptorSets(commandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &skeleton.descriptorSet, 0, nullptr);
		for (auto& node : nodes) {
			drawNodeShadow(commandBuffer, pipelineLayout, node);
		}
	}

};

class VulkanExample : public VulkanExampleBase
{
public:

	//for shadow map
	float zNear = 0.5f;
	float zFar = 5.f;
	float lightFov = 150.f;

	bool shadowDebug = false;

	struct {
		VkPipeline offscreen;
		VkPipeline debug;
	} shadowPipelines;

	//每个pipeline都有自己的descriptor set，但这些pipeline 的使用的layout相同，所两个pipeline使用同一个pipelinelayout
	VkPipelineLayout shadowPipelineLayout;
	struct {
		VkDescriptorSet offscreen;
		VkDescriptorSet debug;
	}shadowDescriptorSets;

	VkDescriptorPool shadowDescriptorPool;


	struct {
		glm::mat4 lightSpaceMatrix;
	} lightSpace;


	struct {
		vks::Buffer lightSpaceMatrixUBO;
	}ubos;




	struct {
		VkSampler sampler;
		VkImage image;
		VkDeviceMemory memory;
		VkImageView imageView;
	} shadowMap;


	VkFramebuffer offscreenFrameBuffer;
	VkRenderPass offscreenRenderPass;



	void initDefalutMap() {
		uint32_t width = 2048, height = 2048;
		std::vector<unsigned char> buffer(width * height * 4, 1);
		VkDeviceSize bufferSize = static_cast<VkDeviceSize>(buffer.size());
		defalutAOmap.fromBuffer(buffer.data(), bufferSize, VK_FORMAT_R8G8B8A8_UNORM, width, height, vulkanDevice, queue);

		buffer.resize(width * height * 4, 0);
		defalutEmissiveMap.fromBuffer(buffer.data(), bufferSize, VK_FORMAT_R8G8B8A8_UNORM, width, height, vulkanDevice, queue);
	}

	void destroyDefalutMap() {
		defalutAOmap.destroy();
		defalutEmissiveMap.destroy();
	}

	bool wireframe = false;

	VulkanglTFModel glTFModel;


	vks::Texture2D defalutAOmap;
	vks::Texture2D defalutEmissiveMap;

	//genrated at run time
	vks::Texture2D lutBrdf;
	vks::TextureCubeMap irradianceCube;
	vks::TextureCubeMap prefilteredCube;


	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec4 lightPos = glm::vec4(1.f, 1.f, -1.f, 1.0f);
			glm::vec4 viewPos;
		} values;
	} shaderData;

	struct Pipelines {
		VkPipeline solid;
		VkPipeline wireframe = VK_NULL_HANDLE;
	} pipelines;

	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
		VkDescriptorSetLayout fkMatrix;
		VkDescriptorSetLayout shadowMap;
	} descriptorSetLayouts;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "homework1";
		camera.type = Camera::CameraType::lookat;
		camera.flipY = true;
		camera.setPosition(glm::vec4(1.f, 1.f, -1.f, 1.0f));
		//camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
		camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note : Inherited destructor cleans up resources stored in base class
		vkDestroyPipeline(device, pipelines.solid, nullptr);
		if (pipelines.wireframe != VK_NULL_HANDLE) {
			vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		}

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.fkMatrix, nullptr);

		shaderData.buffer.destroy();
		destroyDefalutMap();
	}

	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
	}
	// Generate a BRDF integration map used as a look-up-table (stores roughness / NdotV)
	/*void generateBRDFLUT()
	{
		auto tStart = std::chrono::high_resolution_clock::now();

		const VkFormat format = VK_FORMAT_R16G16_SFLOAT;	// R16G16 is supported pretty much everywhere
		const int32_t dim = 512;

		// Image
		VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent.width = dim;
		imageCI.extent.height = dim;
		imageCI.extent.depth = 1;
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &lutBrdf.image));
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, lutBrdf.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &lutBrdf.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, lutBrdf.image, lutBrdf.deviceMemory, 0));
		// Image view
		VkImageViewCreateInfo viewCI = vks::initializers::imageViewCreateInfo();
		viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewCI.format = format;
		viewCI.subresourceRange = {};
		viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewCI.subresourceRange.levelCount = 1;
		viewCI.subresourceRange.layerCount = 1;
		viewCI.image = lutBrdf.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &lutBrdf.view));
		// Sampler
		VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = 1.0f;
		samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &lutBrdf.sampler));

		lutBrdf.descriptor.imageView = lutBrdf.view;
		lutBrdf.descriptor.sampler = lutBrdf.sampler;
		lutBrdf.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		lutBrdf.device = vulkanDevice;

		// FB, Att, RP, Pipe, etc.
		VkAttachmentDescription attDesc = {};
		// Color attachment
		attDesc.format = format;
		attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
		attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attDesc;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = 2;
		renderPassCI.pDependencies = dependencies.data();

		VkRenderPass renderpass;
		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderpass));

		VkFramebufferCreateInfo framebufferCI = vks::initializers::framebufferCreateInfo();
		framebufferCI.renderPass = renderpass;
		framebufferCI.attachmentCount = 1;
		framebufferCI.pAttachments = &lutBrdf.view;
		framebufferCI.width = dim;
		framebufferCI.height = dim;
		framebufferCI.layers = 1;

		VkFramebuffer framebuffer;
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCI, nullptr, &framebuffer));

		// Pipeline layout
		VkPipelineLayout pipelineLayout;
		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(nullptr, 0);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout))

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderpass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = &emptyInputState;

		// Look-up-table (from BRDF) pipeline
		shaderStages[0] = loadShader(getShadersPath() + "pbrtexture/genbrdflut.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "pbrtexture/genbrdflut.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VkPipeline pipeline;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

		// Render
		VkClearValue clearValues[1];
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderpass;
		renderPassBeginInfo.renderArea.extent.width = dim;
		renderPassBeginInfo.renderArea.extent.height = dim;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = framebuffer;

		VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		VkViewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
		VkRect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);
		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdDraw(cmdBuf, 3, 1, 0, 0);
		vkCmdEndRenderPass(cmdBuf);
		vulkanDevice->flushCommandBuffer(cmdBuf, queue);

		vkQueueWaitIdle(queue);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderpass, nullptr);
		vkDestroyFramebuffer(device, framebuffer, nullptr);;

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
	}
	*/
	// Prefilter environment cubemap
	/*void generatePrefilteredCube()
	{
		auto tStart = std::chrono::high_resolution_clock::now();

		const VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
		const int32_t dim = 512;
		const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

		// Pre-filtered cube map
		// Image
		VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = format;
		imageCI.extent.width = dim;
		imageCI.extent.height = dim;
		imageCI.extent.depth = 1;
		imageCI.mipLevels = numMips;
		imageCI.arrayLayers = 6;
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &prefilteredCube.image));
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, prefilteredCube.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &prefilteredCube.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, prefilteredCube.image, prefilteredCube.deviceMemory, 0));
		// Image view
		VkImageViewCreateInfo viewCI = vks::initializers::imageViewCreateInfo();
		viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewCI.format = format;
		viewCI.subresourceRange = {};
		viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewCI.subresourceRange.levelCount = numMips;
		viewCI.subresourceRange.layerCount = 6;
		viewCI.image = prefilteredCube.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &prefilteredCube.view));
		// Sampler
		VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = static_cast<float>(numMips);
		samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &prefilteredCube.sampler));

		prefilteredCube.descriptor.imageView = prefilteredCube.view;
		prefilteredCube.descriptor.sampler = prefilteredCube.sampler;
		prefilteredCube.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		prefilteredCube.device = vulkanDevice;

		// FB, Att, RP, Pipe, etc.
		VkAttachmentDescription attDesc = {};
		// Color attachment
		attDesc.format = format;
		attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
		attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Renderpass
		VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attDesc;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = 2;
		renderPassCI.pDependencies = dependencies.data();
		VkRenderPass renderpass;
		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderpass));

		struct {
			VkImage image;
			VkImageView view;
			VkDeviceMemory memory;
			VkFramebuffer framebuffer;
		} offscreen;

		// Offfscreen framebuffer
		{
			// Color attachment
			VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
			imageCreateInfo.format = format;
			imageCreateInfo.extent.width = dim;
			imageCreateInfo.extent.height = dim;
			imageCreateInfo.extent.depth = 1;
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &offscreen.image));

			VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
			VkMemoryRequirements memReqs;
			vkGetImageMemoryRequirements(device, offscreen.image, &memReqs);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreen.memory));
			VK_CHECK_RESULT(vkBindImageMemory(device, offscreen.image, offscreen.memory, 0));

			VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
			colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
			colorImageView.format = format;
			colorImageView.flags = 0;
			colorImageView.subresourceRange = {};
			colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			colorImageView.subresourceRange.baseMipLevel = 0;
			colorImageView.subresourceRange.levelCount = 1;
			colorImageView.subresourceRange.baseArrayLayer = 0;
			colorImageView.subresourceRange.layerCount = 1;
			colorImageView.image = offscreen.image;
			VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &offscreen.view));

			VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = renderpass;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.pAttachments = &offscreen.view;
			fbufCreateInfo.width = dim;
			fbufCreateInfo.height = dim;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreen.framebuffer));

			VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
			vks::tools::setImageLayout(
				layoutCmd,
				offscreen.image,
				VK_IMAGE_ASPECT_COLOR_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
			vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);
		}

		// Descriptors
		VkDescriptorSetLayout descriptorsetlayout;
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		};
		VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

		// Descriptor Pool
		std::vector<VkDescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
		VkDescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		VkDescriptorPool descriptorpool;
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorpool));

		// Descriptor sets
		VkDescriptorSet descriptorset;
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorset));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorset, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &textures.environmentCube.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

		// Pipeline layout
		struct PushBlock {
			glm::mat4 mvp;
			float roughness;
			uint32_t numSamples = 32u;
		} pushBlock;

		VkPipelineLayout pipelinelayout;
		std::vector<VkPushConstantRange> pushConstantRanges = {
			vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(PushBlock), 0),
		};
		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelinelayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.renderPass = renderpass;
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV });

		shaderStages[0] = loadShader(getShadersPath() + "pbrtexture/filtercube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "pbrtexture/prefilterenvmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VkPipeline pipeline;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

		// Render

		VkClearValue clearValues[1];
		clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 0.0f } };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		// Reuse render pass from example pass
		renderPassBeginInfo.renderPass = renderpass;
		renderPassBeginInfo.framebuffer = offscreen.framebuffer;
		renderPassBeginInfo.renderArea.extent.width = dim;
		renderPassBeginInfo.renderArea.extent.height = dim;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues;

		std::vector<glm::mat4> matrices = {
			// POSITIVE_X
			glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_X
			glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Y
			glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Y
			glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Z
			glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Z
			glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		};

		VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		VkViewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
		VkRect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);

		vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
		vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = numMips;
		subresourceRange.layerCount = 6;

		// Change image layout for all cubemap faces to transfer destination
		vks::tools::setImageLayout(
			cmdBuf,
			prefilteredCube.image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			subresourceRange);

		for (uint32_t m = 0; m < numMips; m++) {
			pushBlock.roughness = (float)m / (float)(numMips - 1);
			for (uint32_t f = 0; f < 6; f++) {
				viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
				viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
				vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

				// Render scene from cube face's point of view
				vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				// Update shader push constant block
				pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

				vkCmdPushConstants(cmdBuf, pipelinelayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlock), &pushBlock);

				vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
				vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 0, 1, &descriptorset, 0, NULL);

				skybox.draw(cmdBuf);

				vkCmdEndRenderPass(cmdBuf);

				vks::tools::setImageLayout(
					cmdBuf,
					offscreen.image,
					VK_IMAGE_ASPECT_COLOR_BIT,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

				// Copy region for transfer from framebuffer to cube face
				VkImageCopy copyRegion = {};

				copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				copyRegion.srcSubresource.baseArrayLayer = 0;
				copyRegion.srcSubresource.mipLevel = 0;
				copyRegion.srcSubresource.layerCount = 1;
				copyRegion.srcOffset = { 0, 0, 0 };

				copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				copyRegion.dstSubresource.baseArrayLayer = f;
				copyRegion.dstSubresource.mipLevel = m;
				copyRegion.dstSubresource.layerCount = 1;
				copyRegion.dstOffset = { 0, 0, 0 };

				copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
				copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
				copyRegion.extent.depth = 1;

				vkCmdCopyImage(
					cmdBuf,
					offscreen.image,
					VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					prefilteredCube.image,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1,
					&copyRegion);

				// Transform framebuffer color attachment back
				vks::tools::setImageLayout(
					cmdBuf,
					offscreen.image,
					VK_IMAGE_ASPECT_COLOR_BIT,
					VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
			}
		}

		vks::tools::setImageLayout(
			cmdBuf,
			prefilteredCube.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			subresourceRange);

		vulkanDevice->flushCommandBuffer(cmdBuf, queue);

		vkDestroyRenderPass(device, renderpass, nullptr);
		vkDestroyFramebuffer(device, offscreen.framebuffer, nullptr);
		vkFreeMemory(device, offscreen.memory, nullptr);
		vkDestroyImageView(device, offscreen.view, nullptr);
		vkDestroyImage(device, offscreen.image, nullptr);
		vkDestroyDescriptorPool(device, descriptorpool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorsetlayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelinelayout, nullptr);

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		std::cout << "Generating pre-filtered enivornment cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
	}
	*/
	//void buildShadowCommandBuffers() {
	void buildCommandBuffers(){
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		VkViewport viewport;
		VkRect2D scissor;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			/*
				First render pass: Generate shadow map by rendering the scene from light's POV
			*/
			{
				clearValues[0].depthStencil = { 1.f, 0 };
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.clearValueCount = 1;
				renderPassBeginInfo.pClearValues = clearValues;
				renderPassBeginInfo.framebuffer = offscreenFrameBuffer;
				renderPassBeginInfo.renderPass = offscreenRenderPass;
				renderPassBeginInfo.renderArea.extent = { SHADOW_DIM, SHADOW_DIM };
				renderPassBeginInfo.renderArea.offset = { 0, 0 };

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				
				viewport = vks::initializers::viewport(SHADOW_DIM, SHADOW_DIM, 0.f, 1.f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				scissor = vks::initializers::rect2D(SHADOW_DIM, SHADOW_DIM, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

				//TODO: set depth bias
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelines.offscreen);
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 1, &shadowDescriptorSets.offscreen, 0, nullptr);

				glTFModel.drawShadow(drawCmdBuffers[i], shadowPipelineLayout);

				vkCmdEndRenderPass(drawCmdBuffers[i]);

			}

			/*
				Note: Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
			*/

			/*
				Second pass: Scene rendering with applied shadow map
			*/

			{
				clearValues[0].color = defaultClearColor;
				clearValues[1].depthStencil = { 1.0f, 0 };

				clearValues[0].depthStencil = { 1.f, 0 };
				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.clearValueCount = 2;
				renderPassBeginInfo.pClearValues = clearValues;
				renderPassBeginInfo.framebuffer = frameBuffers[i];
				renderPassBeginInfo.renderPass = renderPass;
				renderPassBeginInfo.renderArea.extent = { width	, height };
				renderPassBeginInfo.renderArea.offset = { 0, 0 };

				vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
				
				//dynamic state
				viewport = vks::initializers::viewport(width, height, 0.f, 1.f);
				vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

				scissor = vks::initializers::rect2D(width, height, 0, 0);
				vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
				if (shadowDebug) {
					vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelines.debug);
					vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 1, &shadowDescriptorSets.debug, 0, nullptr);
					vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				}
				else {
					// Bind scene matrices descriptor to set 0
					vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
					// Bind shadow map descriptor to set 3
					vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 3, 1, &shadowDescriptorSets.debug, 0, nullptr);
					vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);
					glTFModel.draw(drawCmdBuffers[i], pipelineLayout);
				}

			
				drawUI(drawCmdBuffers[i]);
				vkCmdEndRenderPass(drawCmdBuffers[i]);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}
	void buildCommandBuffers_()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 1.0f } };;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);


			// Bind scene matrices descriptor to set 0
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);
			glTFModel.draw(drawCmdBuffers[i], pipelineLayout);
			drawUI(drawCmdBuffers[i]);
			vkCmdEndRenderPass(drawCmdBuffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadglTFFile(std::string filename)
	{
		tinygltf::Model glTFInput;
		tinygltf::TinyGLTF gltfContext;
		std::string error, warning;

		this->device = device;

#if defined(__ANDROID__)
		// On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset manager
		// We let tinygltf handle this, by passing the asset manager of our app
		tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
		bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

		// Pass some Vulkan resources required for setup and rendering to the glTF model loading class
		glTFModel.vulkanDevice = vulkanDevice;
		glTFModel.copyQueue = queue;

		std::vector<uint32_t> indexBuffer;
		std::vector<VulkanglTFModel::Vertex> vertexBuffer;

		if (fileLoaded) {
			glTFModel.loadImages(glTFInput);
			glTFModel.loadMaterials(glTFInput);
			glTFModel.loadTextures(glTFInput);
			glTFModel.nodeSize = glTFInput.nodes.size();


			const tinygltf::Scene& scene = glTFInput.scenes[0];
			for (size_t i = 0; i < scene.nodes.size(); i++) {
				const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
				glTFModel.loadNode(node, glTFInput, nullptr,i, indexBuffer, vertexBuffer);
			}

			//load animtaions
			glTFModel.loadAnimation(glTFInput);
		}

		else {
			vks::tools::exitFatal("Could not open the glTF file.\n\nThe file is part of the additional asset pack.\n\nRun \"download_assets.py\" in the repository root to download the latest version.", -1);
			return;
		}

		// Create and upload vertex and index buffer
		// We will be using one single vertex buffer and one single index buffer for the whole glTF scene
		// Primitives (of the glTF model) will then index into these using index offsets

		size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
		size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

		struct StagingBuffer {
			VkBuffer buffer;
			VkDeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create host visible staging buffers (source)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indexBuffer.data()));

		// Create device local buffers (target)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBufferSize,
			&glTFModel.vertices.buffer,
			&glTFModel.vertices.memory));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBufferSize,
			&glTFModel.indices.buffer,
			&glTFModel.indices.memory));

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			vertexStaging.buffer,
			glTFModel.vertices.buffer,
			1,
			&copyRegion);

		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			indexStaging.buffer,
			glTFModel.indices.buffer,
			1,
			&copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		// Free staging resources
		vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
		vkFreeMemory(device, vertexStaging.memory, nullptr);
		vkDestroyBuffer(device, indexStaging.buffer, nullptr);
		vkFreeMemory(device, indexStaging.memory, nullptr);
	}

	void loadAssets()
	{
		loadglTFFile(getAssetPath() + "buster_drone/busterDrone.gltf");
		//const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
		//skybox.loadFromFile(getAssetPath() + "models/cube.gltf", vulkanDevice, queue, glTFLoadingFlags);
	}

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
		*/
		
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
			// One combined image sampler per model image/texture
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(5*glTFModel.materials.size())) , 
			
			//ssbo for skeleton matrix
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
		};
		// One set for matrices and one per model image/texture and skeleton matrix
		const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.images.size()) + 2;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Descriptor set layout for passing matrices
		VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));
		
	
		//Descriptor set layout for passing forward knimatics matrix
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.fkMatrix));
		
		// Descriptor set layout for passing material textures
		std::vector<VkDescriptorSetLayoutBinding> texSetLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(texSetLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));

		//Descriport set layout for passing shadow map
		//shadow map generator pipeline 和 shadow debug pipeline共享同一个layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1), //生成shadow map时的 lightMatrixUBO数据
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1) //在debug时使用的shadow map sampler的位置
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.shadowMap ));


		// Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = material,  set 2 = fkMatrix, set 3 = shadowMap)
		std::array<VkDescriptorSetLayout, 4> setLayouts = { 
			descriptorSetLayouts.matrices, 
			descriptorSetLayouts.textures, 
			descriptorSetLayouts.fkMatrix,
			descriptorSetLayouts.shadowMap
			};
		VkPipelineLayoutCreateInfo pipelineLayoutCI= vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		// We will use push constants to push the local matrices of a primitive to the vertex shader
		//VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
		//// Push constant ranges are part of the pipeline layout
		//pipelineLayoutCI.pushConstantRangeCount = 1;
		//pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		// set 0 Descriptor set for scene matrices
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		
		// set 1 Descriptor sets for materials
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
		for (auto& material : glTFModel.materials) {

			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &material.descriptorSet));

			std::array<VkDescriptorImageInfo*, 5> descriptorImageInfos{nullptr};
			if (material.baseColorTextureIndex != -1) {
				descriptorImageInfos[0] = &glTFModel.images[material.baseColorTextureIndex].texture.descriptor;
			}
			if (material.normalTexureIndex != -1) {
				descriptorImageInfos[1] = &glTFModel.images[material.normalTexureIndex].texture.descriptor;
			}
			if (material.occlusionTextureIndex != -1) {
				descriptorImageInfos[2] = &glTFModel.images[material.occlusionTextureIndex].texture.descriptor;
			}
			else {
				descriptorImageInfos[2] = &defalutAOmap.descriptor;
			}
			if (material.metallicRoughnessTextureIndex != -1) {
				descriptorImageInfos[3] = &glTFModel.images[material.metallicRoughnessTextureIndex].texture.descriptor;
			}
			if (material.emissiveTextureIndex != -1) {
				descriptorImageInfos[4] = &glTFModel.images[material.emissiveTextureIndex].texture.descriptor;
			}
			else {
				descriptorImageInfos[4] = &defalutEmissiveMap.descriptor;
			}
			//update each binding point about the material descri
			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			for (int i = 0; i < descriptorImageInfos.size(); ++i) {
				if (descriptorImageInfos[i] != nullptr) {
					VkWriteDescriptorSet w =  vks::initializers::writeDescriptorSet(material.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, i, descriptorImageInfos[i],  1);
					writeDescriptorSets.emplace_back(w);
				}
			}

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(),0 , nullptr);
		}

		//set 2 Descriptor sets for fk matrices
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.fkMatrix, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &glTFModel.skeleton.descriptorSet));
		writeDescriptorSet = vks::initializers::writeDescriptorSet(glTFModel.skeleton.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &glTFModel.skeleton.ssbo.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		// Vertex input bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),	// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),	// Location 3: Color
			vks::initializers::vertexInputAttributeDescription(0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, tangent)), //location 4: tangent
			vks::initializers::vertexInputAttributeDescription(0, 5, VK_FORMAT_R32_UINT,		 offsetof(VulkanglTFModel::Vertex, nodeIndex)), //location 5: for index transformation
		};
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

		const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1TODO/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1TODO/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Solid rendering pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			rasterizationStateCI.lineWidth = 1.0f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}
	}


	void setupShadowDescriptor() {

		//descripotor Layout + pipeline layout

		//shadow map generator pipeline 和 shadow debug pipeline共享同一个layout
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1), //生成shadow map时的 lightMatrixUBO数据
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1, 1) //在debug时使用的shadow map sampler的位置
		};

		
		std::array<VkDescriptorSetLayout, 2>  setLayouts{};
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &setLayouts[0]));

		//多一个storage buffer 的descriptor set layout，但是并不创建descriptorSet，就是说，这个layout 是用来作为shader 的接口使用的，shader里面根据layout接口访问资源，而descriptor set真正资源实体，在运行时由gltfModel绑定
		VkDescriptorSetLayoutBinding binding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0, 1);
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&binding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &setLayouts[1]));


		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &shadowPipelineLayout));


		//descriptor pool: 总共两个descriptor set， 分别是(set=0, binding = 0)uniform buffer object 和(set=0, binding=1)sampler
		// (set=1, binnding=0) shader storage buffer
		//VkDescriptorPoolSize poolSize = vks::initializers::descriptorPoolSize(VK)
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2)
		};

		//2 Descriptor sets * 2，offscren 和 debug pipeline都使用同样的pipeline layout，该pipeline layout里有两个 sets
		VkDescriptorPoolCreateInfo poolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 4);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolCI, nullptr, &shadowDescriptorPool));

		//descriptor set

			//offscreen
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(shadowDescriptorPool, setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowDescriptorSets.offscreen));
		
		std::vector<VkWriteDescriptorSet>  writeSets = {
			//binding 0: light space matrix uniform buffer object
			vks::initializers::writeDescriptorSet(shadowDescriptorSets.offscreen, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &ubos.lightSpaceMatrixUBO.descriptor, 1)
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeSets.size()), writeSets.data(), 0, nullptr); //update descriptor set setting
	
	
			//debug, debug有 light view projection，可以用来，将shading point变换到light space
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &shadowDescriptorSets.debug));
		VkDescriptorImageInfo imageInfo = vks::initializers::descriptorImageInfo(shadowMap.sampler, shadowMap.imageView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
		writeSets = {
			//binding 0: light space view projectino matrix
			vks::initializers::writeDescriptorSet(shadowDescriptorSets.debug, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,  &ubos.lightSpaceMatrixUBO.descriptor, 1),
			//binding 1: depth map sampler
			vks::initializers::writeDescriptorSet(shadowDescriptorSets.debug, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &imageInfo, 1)
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeSets.size()), writeSets.data(), 0, nullptr);

	}

	VkImage createVkImage() {
		VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
		imageCI.extent = { SHADOW_DIM, SHADOW_DIM, 1 };
		imageCI.format = VK_FORMAT_D16_UNORM;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCI.mipLevels = 1; //depth map, no mipmap
		imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCI.arrayLayers = 1;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL; //指定图像的存储方式，按块存储，适用于顺序读写，(LINEAR按行存储适合随机读写)
		imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; //生成shadow map pass作为深度缓冲attachment，在渲染场景pass，作为smapler

		
		VkImage image{};
		VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &image));
		return image;
	}
	VkDeviceMemory createVkDeviceMemory(VkImage image) {
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		
		VkDeviceMemory memory{};
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo,nullptr, &memory));
		return memory;

	}
	VkImageView createVkImageView(VkImage image) {
		VkImageViewCreateInfo imageViewCI = vks::initializers::imageViewCreateInfo();
		imageViewCI.image = image;
		imageViewCI.format = VK_FORMAT_D16_UNORM;
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		imageViewCI.subresourceRange.baseArrayLayer = 0;
		imageViewCI.subresourceRange.layerCount = 1;
		imageViewCI.subresourceRange.baseMipLevel = 0;
		imageViewCI.subresourceRange.levelCount = 1;

		VkImageView imageView{};
		VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &imageView));
		return imageView;
	}

	void bindImageMemory(VkImage image, VkDeviceMemory memory) {
		VK_CHECK_RESULT(vkBindImageMemory(device, image, memory, 0));
		
	}
	VkSampler createVkSampler() {
		VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCI.addressModeV = samplerCI.addressModeU;
		samplerCI.addressModeW = samplerCI.addressModeU;
		samplerCI.anisotropyEnable = VK_FALSE;
		samplerCI.mipLodBias = 0.0f;
		samplerCI.maxAnisotropy = 1.0f;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = 1.0f;
		samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		VkFilter shadowMapFilter = vks::tools::formatIsFilterable(physicalDevice, VK_FORMAT_D16_UNORM, VK_IMAGE_TILING_OPTIMAL) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		samplerCI.minFilter = shadowMapFilter;
		samplerCI.magFilter = shadowMapFilter;	

		VkSampler sampler{};

		VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &sampler));
		return sampler;
	}

	//创建新的frame buffer，用于生成shadow mapping的render pass

	VkFramebuffer createFrameBuffer(VkImageView imageView, VkRenderPass renderPass) {
		VkFramebufferCreateInfo frameBufferCI = vks::initializers::framebufferCreateInfo();
		frameBufferCI.attachmentCount = 1;
		frameBufferCI.pAttachments = &imageView;
		frameBufferCI.width = SHADOW_DIM;
		frameBufferCI.height = SHADOW_DIM;
		frameBufferCI.layers = 1;
		frameBufferCI.renderPass = renderPass;
		
		VkFramebuffer frameBuffer{};
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCI, nullptr, &frameBuffer));
		return frameBuffer;
	}
	VkRenderPass createOffscreenRenderPass() {
		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = VK_FORMAT_D16_UNORM;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL; //当前shadow render pass 结束后，这个depth attachment应该作为纹理，下一个renderpass给读取，因此finallayout为read

		VkAttachmentReference attachmentRef{};
		attachmentRef.attachment = 0;
		attachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//subpass
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.pDepthStencilAttachment = &attachmentRef;


		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attachmentDescription;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpass;
		renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCI.pDependencies = dependencies.data();

		VkRenderPass renderPass;
		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderPass));
		return renderPass;
	}

	void prepareShadowImageSampler() {
		shadowMap.image = createVkImage();
		shadowMap.memory = createVkDeviceMemory(shadowMap.image);
		bindImageMemory(shadowMap.image, shadowMap.memory);
		shadowMap.imageView = createVkImageView(shadowMap.image);
		shadowMap.sampler = createVkSampler();
	}

	void prepareShadowFramebuffer() {
		offscreenRenderPass = createOffscreenRenderPass();

		offscreenFrameBuffer = createFrameBuffer(shadowMap.imageView, offscreenRenderPass);
	}

	
	void preparShadowPipeline() {
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo();
		pipelineCI.pVertexInputState = &emptyInputState;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = shaderStages.size();
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.layout = shadowPipelineLayout;
		pipelineCI.renderPass = renderPass;   //使用原来的renderPass
	
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1TODO/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getHomeworkShadersPath() + "homework1TODO/quad.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		/*debug pipeline*/
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &shadowPipelines.debug));
	

		// Vertex input bindings and attributes, 生成shadow map需要 vertex 信息
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),	// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),	// Location 3: Color
			vks::initializers::vertexInputAttributeDescription(0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, tangent)), //location 4: tangent
			vks::initializers::vertexInputAttributeDescription(0, 5, VK_FORMAT_R32_UINT,		 offsetof(VulkanglTFModel::Vertex, nodeIndex)), //location 5: for index transformation
		};
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

		shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1TODO/offscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		pipelineCI.stageCount = 1;
		pipelineCI.pVertexInputState = &vertexInputStateCI;

		//TODO 没有color attachmetn使用, colorBlendStateCI的参数式什么意思
		colorBlendStateCI.attachmentCount = 0;
		//所有的面都会参与深度计算
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		//TODO  Enable depht bias
		rasterizationStateCI.depthBiasEnable = VK_FALSE;
		/** {
			// Add depth bias to dynamic state, so we can change it at runtime
			dynamicStateEnables.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
			dynamicStateCI =
				vks::initializers::pipelineDynamicStateCreateInfo(
					dynamicStateEnables.data(),
					dynamicStateEnables.size(),
					0);
		}*/

		pipelineCI.renderPass = offscreenRenderPass;

		/*offscreen pipeline*/
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &shadowPipelines.offscreen));
		

	}


	//创建 生成depth map时的ubo buffer
	void prepareShadowUniformBuffers() {
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&ubos.lightSpaceMatrixUBO,
			sizeof(lightSpace), nullptr));
		//persistent mapping
		ubos.lightSpaceMatrixUBO.map();
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&shaderData.buffer,
			sizeof(shaderData.values)));

		// Map persistent
		VK_CHECK_RESULT(shaderData.buffer.map());

		updateUniformBuffers();
	}

	void updateLight()
	{
		// Animate the light source
		shaderData.values.lightPos.x = cos(glm::radians(timer * 360.0f)) *1.0f;
		shaderData.values.lightPos.y = -1.0f + sin(glm::radians(timer * 360.0f)) * 1.0f;
		shaderData.values.lightPos.z = 1.0f + sin(glm::radians(timer * 360.0f)) * 1.0f;
	}

	void updateUniformBuffers()
	{
		shaderData.values.projection = camera.matrices.perspective;
		shaderData.values.view = camera.matrices.view;
		shaderData.values.viewPos = camera.viewPos;
		memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));
	}
	void updateShadowUniformBuffers() {
		float top = zNear * std::tan(glm::radians(lightFov * 0.5f)), right = top;
		{
			glm::mat4 f =  glm::ortho(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
			auto a = f;
			glm::mat4 l = glm::lookAtRH(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
			glm::mat4 lr = glm::lookAtRH(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
			glm::mat4 ll = glm::lookAtLH(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

			std::cout << std::endl;

		}
		
		glm::mat4 lightviewProjectoin = glm::ortho(-right, right, -top, top, zNear, zFar)*glm::lookAt(glm::vec3(shaderData.values.lightPos), glm::vec3(0.f), glm::vec3(0.f, -1.f, 0.f));
		memcpy(ubos.lightSpaceMatrixUBO.mapped, &lightviewProjectoin, sizeof(lightviewProjectoin));
	}


	void prepare()
	{

		VulkanExampleBase::prepare();
		initDefalutMap();
		loadAssets();
		{//shadow map
			prepareShadowUniformBuffers();
			prepareShadowImageSampler();
			setupShadowDescriptor();

			prepareShadowFramebuffer();
			preparShadowPipeline();
		}

		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		renderFrame();
		//updateLight();
		updateUniformBuffers();
		updateShadowUniformBuffers();
		if (camera.updated) {
		
		}
		glTFModel.updateAnimation(frameTimer);
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Wireframe", &wireframe)) {
				buildCommandBuffers();
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
