# 1. 概念以及使用

1. vk对象创建方法：

   1. 对象的描述VkObjectCreateInfo
   2. 创建vkCreateObject

2. 带有内存的对象创建

   1. 对象的描述VkObjectCreateInfo
   2. vkCreateBuffer(): VkBuffer句柄
   3. 获取内存要求：vkGetBufferMemoryRequirement(): Requirement句柄
   4. 分配内存：vkAllocateMemory(): VkDeviceMemory句柄
   5. 内存与Buffer句柄的绑定 vkBindBufferMemory()

3. subpass+attachmet

   ​	Vulkan中的**subpass使用VkAttachmentReference来引用attachment**。**每个subpass都会使用一组attachment来进行渲染**，并通过VkAttachmentReference结构体来指定subpass中使用的attachment的索引和使用方式。这些attachment的定义是在渲染流程中的VkRenderPass对象中进行的，而**subpass则指定了哪些attachment用于该子渲染过程以及如何使用它们**。这种设计**允许我们在不同的子渲染过程中重用相同的attachment**(attachmentRef引用的的同一些attachment)，从而提高了渲染效率。

4. 一个图像的布局表示该图像在内存中的存储方式和使用方式，`不同的布局可以影响 GPU 在读取和写入图像数据时的性能和行为。

5. command pool创建时

6. pipeline layout

   ![pipeline layout](https://pic1.zhimg.com/80/v2-b54e9aa0481d4565bcfc5a07e276b224_720w.webp?source=d16d100b)

7. DescriptorSetLayout 描述了 Shader 使用资源的布局，它主要包括了一个 binding 数组，数组中每项代表一个 Descriptor 信息，说明了这个 Descriptor 的类型、数量、对应的 Shader 阶段以及静态采样器列表。https://zhuanlan.zhihu.com/p/124251944

8. **Descriptor**是一个Descriptor set中的一项，他在一个集合中，如何来定位呢？就是使用**binding**，CPU端指定UBO为binding 0，texture sampler是binding 1，那么在GPU端的shader中，他就可以使用binding = 0/1，来拿到对应的资源。

9. pipeline barrirer同步资源的访问，比如保证图像在被读取之 前数据被写入。它也可以被用来变换图像布局。

10. 在 Vulkan 中，每个图像都必须被指定一个特定的布局，这个布局定义了图像可以用于哪些操作，并决定了图像数据在内存中的组织方式。例如，`VK_IMAGE_LAYOUT_UNDEFINED` 表示图像数据在第一次使用前没有被定义，`VK_IMAGE_LAYOUT_GENERAL` 表示图像可以被用于任何操作，`VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` 表示图像用于颜色附件，等等。

   在执行 `vkCmdCopyBufferToImage` 命令之前，需要确保目标图像已经被转换为可用于复制的布局。这是因为 `vkCmdCopyBufferToImage` 命令会将缓冲区中的数据复制到图像的指定区域。如果目标图像的布局不正确，可能会导致数据不正确或者出现未定义行为。

   因此，在调用 `vkCmdCopyBufferToImage` 之前，需要调用 `vkCmdPipelineBarrier` 命令，将目标图像的布局从未定义的布局转换为适合作为复制目标的布局。这可以通过将 `VK_IMAGE_LAYOUT_UNDEFINED` 布局转换为 `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` 来实现，从而指示图像将被用作传输目标。

11. aspect 是指图像的一个方面，如颜色、深度或模板等。aspectMask 参数用于指定要操作的图像方面，例如 VK_IMAGE_ASPECT_COLOR_BIT 表示操作颜色方面，VK_IMAGE_ASPECT_DEPTH_BIT 表示操作深度方面，VK_IMAGE_ASPECT_STENCIL_BIT 表示操作模板方面

    aspectMask 参数通常在创建 ImageView 和进行 Barrier 操作时使用，以指定对哪个方面进行操作。

12. 描述旧布局和新布局之间的**资源访问控制**，srcAccessMask, dstAccessMask指定了不同的布局之下，资源的访问方式，例如将image的layout从VK_IMAGE_LAYOUT_UNDEFINED转换为VK_IMAGE_LAYOUT_TRANSFER_DST_BIT，说明oldlayout的访问控制是无，**newlayout的访问控制是允许作为传输目标进行写**

    ```c++
    srcAccessMask = 0;
    dstAccessMask = VK_TRANSFER_WRITE_BIT;
    ```

13. Vulkan中的pipeline由多个stage组成，每个stage都有特定的功能和目的，下面是一个完整的pipeline中的所有stage（按照执行顺序排列）：

    1. VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT：表示pipeline开始执行的阶段，通常用于同步操作等待所有前置操作完成；
    2. VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT：表示处理绘制间接命令的阶段；
    3. VK_PIPELINE_STAGE_VERTEX_INPUT_BIT：表示顶点数据输入阶段；
    4. VK_PIPELINE_STAGE_VERTEX_SHADER_BIT：表示顶点着色器执行阶段；
    5. VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT：表示镶嵌细分控制着色器执行阶段；
    6. VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT：表示镶嵌细分评估着色器执行阶段；
    7. VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT：表示几何着色器执行阶段；
    8. VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT：表示片段着色器执行阶段；
    9. VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT：表示执行深度/模板测试的阶段；
    10. VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT：表示执行深度/模板测试的阶段；
    11. VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT：表示执行颜色混合和写入操作的阶段；
    12. VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT：表示计算着色器执行阶段；
    13. VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT：表示pipeline完成执行的阶段，通常用于同步操作等待所有后续操作完成。

    注意，上述stage不是全部都会在一个pipeline中使用，而是根据需要选择合适的stage组合。例如，一个简单的graphics pipeline通常只需要使用前面的几个stage，而不需要使用计算着色器和最后的bottom-of-pipe阶段。

14. RenderPasshttps://zhuanlan.zhihu.com/p/617374058

    1. A render pass represent a collection of attachements , subpasses , and dependencies between the subpasses and describes how the attachements are used over the course of subpasses. The use of a render pass in a command buffer is a render pass instance

    2. Renderpass object会和VkFrameBuffer object配合使用，framebuffer代表一系列即将用作attachement的images的集合，这些attachment将会在renderpass中使用到。
