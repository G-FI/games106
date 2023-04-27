1. gLTF文件
	1. glTF™ (GL Transmission Format) is a royalty-free specification for the efficient transmission and loading of 3D scenes and models by applications. glTF minimizes both the size of 3D assets, and the runtime processing needed to unpack and use those assets. glTF defines an extensible, common publishing format for 3D content tools and services that streamlines authoring workflows and enables interoperable use of content across the industry.
	2. image buffer引用渲染模型需要的外部数据，image纹理，buffer几何/动画数据
	3. node引用了一个mesh/carmra，并且有一个局部坐标变换矩阵
	4. vertex skining，mesh上的顶点会受到骨骼的影响
	5. animation: 
		1. channels: index引用一个node标识作用动画的target，path是一个变换，标识对target做什么变换，变换的数据通过引用的smapler获得，sampler通过引用的两帧数据进行滚插值得到数据
		2. smapler:引用input and output data，分别为accesssor的索引，accessor对应的数据为前后两帧的数据，并且interplotation指定插值方式
		
# 作业框架相关
1. logical device 被抽象成VulkanDevice，并且创建它时，query出logical device需要支持的队列簇的indices(例如graphics，compute，transfer队列簇)，并创建VkQueueCreateInfo，在创建Device时
    1. 将queueCreateInfo传递给DeviceCreateinfo,
	2. 将需要支持的extension传递给DeviceCreateInfo，
	3. 将device需要支持的Feature告诉DeviceCreateInfo
	3. 标识要创建的队列需要以及创建commandBufferPool，并且它的usage bit为reset，也就是commandBuffer可以进行重用，

2. RenderCompleteSemaphore:用于同步present操作，因为需要先render结束之后才能present，所以它结束之后semaphore调用V操作，此时present阶段才能进行展示颜色
3. PresentCompleteSemphore:用于同步vkAcquireNextImage，因为需要先将image渲染完成之后present之后他才能被acquire出来，然而vkAcquireNextImage调用会直接返回，若返回的nextImage index还没有被present，这时之后的vkQueueSubmit提交一个绘制命令绘制到这个acquire出来的图像上，那就会出错，因为它上一帧的图像还没有被呈现出来
4. VulkanExampelBase::vulkanDevice是一个逻辑设备，在创建它时创建了一个commandPool,作为vulkanDevice的成员，但是VulkanExampel子类，在prepare时，也创建了一个commandPool，这是为何？
5. 由于模型的vertices和vertexIndices数据不会改变，所以也使用staging buffer将它们copy到GPU local emory
6. 创建descriptor set时创建descriptor set layout，并且创建pipelineLayout，并且创建pipeline

# 作业记录
1. 修改VulkanglTFModel::Node结构，添加index用于之后递归根据索引查找对应Node，因为tinyglft中对于每个animation channel中指定的是Node的的索引，而VulkanglTFModel::Node是以树状结构组织Node的，无法直接通过索引获得对应Node， 修改loadNode函数签名，添加传入参数nodeIndex，以在加载Node是设置它在全局nodes中对应的索引
2. 添加nodeFromIndex函数用于DSF查找Node
3. 添加加载Animation函数
4. TODO 配置Joint信息
5. 学习清除如何创建animation信息的descriptor set，并在合适的位置将其发送到GPU
6. gltf配置每个顶点最多受4个joints的影响
7. VkDrawIndex()因为之前已将所有vertex和vertexIndices传到GPUbuffer中，所以对于不同的mesh primitive只需要指定这个primitive的顶顶点在indices中的起始位置，以及数量，就能绘制出这个primitive。

# 骨骼动画
1. 配置command buffer，在每次绘制一个node->mesh之前，1号binding point上使用Descriptor set描述fk matrix
2. 解释
    ```C++

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
	```
	通过当前时间找到前后两个key frame i，j
	目标骨骼为 AnimationChannel::Node* node;
	fk变化为 outputs i，j之间进行插值(根据channel::path指定的变化translation，rotation，scale)
3. 宏观描述：
    1. shader中传入所有node的transformation (node就代表了骨骼，由一个mesh即多个三角形组成)
	2. 配置顶点属性时，添加一项nodeIndex，来说明每一个个顶点属于哪一个骨骼
	3. 在shader中通过nodeIndex获取对应的transformation matrix，来进行变换
4. 直接updateAnimation，它会更新每个node的局部transformation，然后resetCommand重新构建命令，它会使用vkCmdPushConstant将model传递给GPU，问题就是每帧都得重建commandBuffer，直接将所有全局transformation传递到ssbo，然后每帧只需要CPU端计算然后memcpy到GPU即可
# PBR
1. 材质直接push constant到GPU中，因为在渲染过程中不会改变
2. light的信息和场景信息通过uniform 每帧更新
3. 需要将多个采样器传送给pipeline, 用来作为漫反射、法线等贴图
# TODO
1. createDescriptorPool的poolSize可能需要修改，因为新添加了一个ssbo
