// vk_conv_test.cpp -- Vulkan compute 单层 3x3 卷积，和 PyTorch 参考值对比。
//
// 这一步只验证"链路 + shader + 内存布局"三件事，不管性能。
//   testdata/x.bin   [C,H,W]      fp32
//   testdata/w.bin   [O,C,3,3]    fp32
//   testdata/b.bin   [O]          fp32
//   testdata/ref.bin [O,H,W]      fp32
// 用法: vk_conv_test.exe [slope]
#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult r_ = (x);                                                     \
        if (r_ != VK_SUCCESS) {                                                \
            std::printf("[vk] %s failed: %d (line %d)\n", #x, (int)r_, __LINE__); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("open failed: %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf((size_t)n);
    if (std::fread(buf.data(), 1, (size_t)n, f) != (size_t)n) std::exit(1);
    std::fclose(f);
    return buf;
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    // 关键：优先 DEVICE_LOCAL（否则会落到系统内存，所有访问都走 PCIe）。
    // ReBAR 机型上存在 DEVICE_LOCAL|HOST_VISIBLE 的堆，可以直接映射，省掉 staging。
    const VkMemoryPropertyFlags want_dl = want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want_dl) == want_dl) return i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    std::printf("no suitable memory type\n");
    std::exit(1);
}

static Buf make_buf(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size,
                    VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mem_type(pd, req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped));
    }
    return b;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const float slope = (argc > 1) ? (float)std::atof(argv[1]) : 0.1f;
    const int C = 64, H = 16, W = 20, O = 64;

    // ---- Vulkan 初始化（无 surface）----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{};
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice pd = devs[0];
    for (auto d : devs) {                      // 选独显：DEVICE_LOCAL 堆最大
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = d; break; }
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s\n", props.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qs.data());
    uint32_t qi = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev{};
    VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue{};
    vkGetDeviceQueue(dev, qi, 0, &queue);

    // ---- buffers ----
    const VkDeviceSize fmt = sizeof(float);
    auto host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buf bx = make_buf(dev, pd, (VkDeviceSize)C * H * W * fmt,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    Buf bw = make_buf(dev, pd, (VkDeviceSize)O * C * 9 * fmt,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    Buf bb = make_buf(dev, pd, (VkDeviceSize)O * fmt,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    Buf by = make_buf(dev, pd, (VkDeviceSize)O * H * W * fmt,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);

    auto load_into = [&](Buf& b, const char* path) {
        auto v = read_file(path);
        if (v.size() != b.size) {
            std::printf("size mismatch %s: %zu vs %llu\n", path, v.size(),
                        (unsigned long long)b.size);
            std::exit(1);
        }
        std::memcpy(b.mapped, v.data(), v.size());
    };
    load_into(bx, "testdata/x.bin");
    load_into(bw, "testdata/w.bin");
    load_into(bb, "testdata/b.bin");

    // ---- shader ----
    auto spv = read_file("shaders/conv3x3_c64.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm{};
    VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkDescriptorSetLayoutBinding binds[4]{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 4;
    dslci.pBindings = binds;
    VkDescriptorSetLayout dsl{};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = 5 * sizeof(int32_t);            // C,H,W,O,slope
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl{};
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe{};
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool dp{};
    VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds{};
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));

    VkDescriptorBufferInfo dbi[4] = {{bx.buf, 0, bx.size}, {bw.buf, 0, bw.size},
                                     {bb.buf, 0, bb.size}, {by.buf, 0, by.size}};
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 4, writes, 0, nullptr);

    // ---- 命令 ----
    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qi;
    VkCommandPool cp{};
    VK_CHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb{};
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    int32_t pc[4] = {C, H, W, O};
    float pcf[1] = {slope};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), sizeof(pcf), pcf);
    vkCmdDispatch(cb, (uint32_t)((W + 7) / 8), (uint32_t)((H + 7) / 8), 1u);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    // ---- 对比 ----
    auto ref = read_file("testdata/ref.bin");
    const float* got = (const float*)by.mapped;
    const float* want = (const float*)ref.data();
    double md = 0, mean = 0;
    const size_t n = (size_t)O * H * W;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)got[i] - (double)want[i]);
        md = d > md ? d : md;
        mean += d;
    }
    mean /= (double)n;
    std::printf("maxdiff %.3e  meandiff %.3e  %s\n", md, mean, md < 2e-3 ? "OK" : "**FAIL**");
    std::printf("got[0..3] %.5f %.5f %.5f %.5f   ref %.5f %.5f %.5f %.5f\n", got[0], got[1],
                got[2], got[3], want[0], want[1], want[2], want[3]);

    // ---- 性能：按真实尺寸（C=64,O=64, 480p）单独跑一遍，20 次取平均 ----
    {
        const int bh_ = 270, bw_ = 480, bc_ = 64, bo_ = 64;
        Buf big_in = make_buf(dev, pd, (VkDeviceSize)bc_ * bh_ * bw_ * fmt,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
        Buf big_out = make_buf(dev, pd, (VkDeviceSize)bo_ * bh_ * bw_ * fmt,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
        VkDescriptorBufferInfo dbi2[4] = {{big_in.buf, 0, big_in.size}, {bw.buf, 0, bw.size},
                                          {bb.buf, 0, bb.size}, {big_out.buf, 0, big_out.size}};
        VkWriteDescriptorSet w2[4]{};
        for (int i = 0; i < 4; ++i) {
            w2[i] = writes[i];
            w2[i].pBufferInfo = &dbi2[i];
        }
        vkUpdateDescriptorSets(dev, 4, w2, 0, nullptr);

        int32_t pc2[4] = {bc_, bh_, bw_, bo_};
        const uint32_t gx = (uint32_t)((bw_ + 7) / 8), gy = (uint32_t)((bh_ + 7) / 8);
        const int reps = 10;

        // 录一遍完整绑定（绑定必须每次重建命令缓冲时都重来）
        auto record = [&]() {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(cb, &bi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc2), sizeof(pcf), pcf);
            for (int i = 0; i < reps; ++i) vkCmdDispatch(cb, gx, gy, 1u);
            VK_CHECK(vkEndCommandBuffer(cb));
        };
        VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si2.commandBufferCount = 1;
        si2.pCommandBuffers = &cb;

        record();                                   // 热身：第一次含管线/缓存预热
        VK_CHECK(vkQueueSubmit(queue, 1, &si2, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));

        double best = 1e9;
        for (int round = 0; round < 3; ++round) {
            record();
            const auto t0 = std::chrono::high_resolution_clock::now();
            VK_CHECK(vkQueueSubmit(queue, 1, &si2, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            const double ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count() / reps;
            best = ms < best ? ms : best;
        }
        const double ms = best;
        const double gflop = 2.0 * bc_ * bo_ * 9.0 * bh_ * bw_ / 1e9;
        std::printf("单层 3x3  C=%d O=%d %dx%d : %.3f ms  (%.1f GFLOP -> %.2f TFLOPS)\n", bc_,
                    bo_, bw_, bh_, ms, gflop, gflop / (ms * 1e-3) / 1000.0);
        std::printf("  40 层外推 %.1f ms/帧（torch bf16 是 39.5 ms/帧 @270x480）\n", ms * 40);
    }

    // ---- 基准：纯流式读写带宽（排除"GPU 时钟/队列异常"这类怀疑）----
    {
        auto spv2 = read_file("shaders/copy.spv");
        VkShaderModuleCreateInfo smci2{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci2.codeSize = spv2.size();
        smci2.pCode = reinterpret_cast<const uint32_t*>(spv2.data());
        VkShaderModule sm2{};
        VK_CHECK(vkCreateShaderModule(dev, &smci2, nullptr, &sm2));

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm2;
        ci.stage.pName = "main";
        ci.layout = pl;
        VkPipeline pipe2{};
        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe2));

        Buf bsrc = make_buf(dev, pd, (VkDeviceSize)64 * 270 * 480 * fmt,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
        Buf bdst = make_buf(dev, pd, (VkDeviceSize)64 * 270 * 480 * fmt,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
        VkDescriptorBufferInfo d3[4] = {{bsrc.buf, 0, bsrc.size}, {bw.buf, 0, bw.size},
                                        {bb.buf, 0, bb.size}, {bdst.buf, 0, bdst.size}};
        VkWriteDescriptorSet w3[4]{};
        for (int i = 0; i < 4; ++i) {
            w3[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w3[i].dstSet = ds;
            w3[i].dstBinding = (uint32_t)i;
            w3[i].descriptorCount = 1;
            w3[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w3[i].pBufferInfo = &d3[i];
        }
        vkUpdateDescriptorSets(dev, 4, w3, 0, nullptr);

        const int n = 64 * 270 * 480;
        int32_t pcn[5] = {n, 0, 0, 0, 0};
        const uint32_t groups = (uint32_t)((n + 255) / 256);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe2);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(int32_t), pcn);
        const int reps2 = 10;
        for (int i = 0; i < reps2; ++i) vkCmdDispatch(cb, groups, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo s3{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        s3.commandBufferCount = 1;
        s3.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue, 1, &s3, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        const auto tb = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkQueueSubmit(queue, 1, &s3, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        const double ms2 = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - tb).count() /
                           reps2;
        const double gb = 2.0 * n * 4.0 / 1e9;     // 读 + 写
        std::printf("流式读写 %d 个 float: %.3f ms  -> %.1f GB/s\n", n, ms2, gb / (ms2 * 1e-3));
    }

    vkDeviceWaitIdle(dev);
    return 0;
}

