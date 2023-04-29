# 骨骼动画

1. 将所有骨骼的世界坐标下的变化矩阵作为VK_BUFFER_USAGE_STORAGE_BUFFER_BIT类型的buffer，并将这些矩阵数据作为一个descriptor set，在绘制之前进行绑定

2. 在每个顶点中添加属性nodeIndex，说明该顶点是属于哪一个骨骼的，以便用来索引对应的变化矩阵

3. 根据前向动力学，跟新每个骨骼的在世界坐标系下单变化矩阵，然后将更新结果拷贝到GPU的buffer中

4. 在shader中根据nodelndex获取对应的变换矩阵，然后计算顶点的位置

   ```glsl
   
   layout (location = 5) out uint outNodeIndex;
   
   layout(set=2, binding=0) readonly buffer FkMatrices{
   	mat4 fkMatririces[];
   };
   
   void main(){
   	//transformation matrix
   	mat4 skeletonMat = fkMatririces[inNodeIndex];
   	gl_Position = uboScene.projection * uboScene.view  * skeletonMat * vec4(inPos.xyz, 1.0);
   }
   ```

# PBR

 1. 对于每个材质，都有一组对应的纹理，将每个材质作为一个descriptor set，当绘mesh时，根据其引用的primitive所使用的材质，绑定对应的descriptor set。

    ```glsl
    layout (set = 1, binding = 0) uniform sampler2D albedoMap;
    layout (set = 1, binding = 1) uniform sampler2D normalMap;
    layout (set = 1, binding = 2) uniform sampler2D aoMap;
    layout (set = 1, binding = 3) uniform sampler2D metallicRoughnessMap;
    layout (set = 1, binding = 4) uniform sampler2D emissiveMap;
    ```

 2. 更descriptor set配置时，对于其中部分材质没有aoMap，和emissiveMap，使用默认的纹理来代替，具体是在创建`VkWriteDescriptorSet`时，检测材质对应的纹理是否存在，若存在使用那个纹理对应的`VkDescriptorImageInfo`来为作为`VkWriteDescriptorSet::pImageInfo`的值来更新descriptorSet的配置，否则使用创建的默认纹理的`VkDescriptorImageInfo Texture::descriptor;` 作为它的值来更新descriptorSet的配置。