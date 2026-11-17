// model_test.cpp -- 在 Vulkan 上跑完整 NanoVSR，逐级与 PyTorch 参考对拍。
//
// 数据来源（由 pack_model.py 生成）：
//   testrun_s/weights_rt.bin   [O][k*k][C] fp32 权重 + [O] bias（带 json 清单）
//   testrun_s/x.bin            [T][H][W][3]
//   testrun_s/feat.bin 等参考输出
//
// 用法: model_test.exe <testrun_dir>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                             \
    do {                                                                        \
        VkResult r_ = (x);                                                      \
        if (r_ != VK_SUCCESS) {                                                 \
            std::printf("[vk] %s failed: %d (%s:%d)\n", #x, (int)r_, __FILE__,   \
                        __LINE__);                                              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static std::vector<char> read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("open failed: %s\n", path.c_str());
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> b((size_t)n);
    if (std::fread(b.data(), 1, (size_t)n, f) != (size_t)n) std::exit(1);
    std::fclose(f);
    return b;
}

// ---------------------------------------------------------------- Vulkan 基础
struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct Push {
    int32_t C_ = 0, H = 0, W = 0, O = 0;
    float slope = 0.0f;
    int32_t k = 0;
    int32_t pad0 = 0, pad1 = 0;
};

struct Gpu {
    VkInstance inst{};
    VkPhysicalDevice pd{};
    VkDevice dev{};
    VkQueue queue{};
    uint32_t qfam = 0;
    VkCommandPool cmdpool{};
    VkCommandBuffer cb{};
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout pl{};
    VkDescriptorPool dpool{};
    std::map<std::string, VkPipeline> pipes;

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst, &n, devs.data());
        pd = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                pd = d;
                break;
            }
        }
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pd, &p);
        std::printf("device: %s  (%u.%u)  shared/blk %u B\n", p.deviceName,
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    p.limits.maxComputeSharedMemorySize);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qfam = i;
                break;
            }
        }
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        // ---- 启用 fp16 + 张量核（cooperative matrix）----
        VkPhysicalDevice16BitStorageFeatures storage16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceShaderFloat16Int8Features f16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        storage16.pNext = &f16;
        f16.pNext = &cmf;
        f2.pNext = &storage16;
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        std::printf("features: storage16=%d float16=%d coopmat=%d\n",
                    (int)storage16.storageBuffer16BitAccess, (int)f16.shaderFloat16,
                    (int)cmf.cooperativeMatrix);
        const char* exts[] = {"VK_KHR_cooperative_matrix"};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &storage16;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;
        VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = qfam;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cmdpool));
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdpool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

        VkDescriptorSetLayoutBinding binds[4]{};
        for (int i = 0; i < 4; ++i) {
            binds[i].binding = (uint32_t)i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 4;
        dslci.pBindings = binds;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(Push);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32768};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 8192;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
    }

    uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        const VkMemoryPropertyFlags wdl = want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & wdl) == wdl) return i;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        std::printf("no suitable memory type\n");
        std::exit(1);
    }

    // 显存里分配 buffer（优先 DEVICE_LOCAL|HOST_VISIBLE，ReBAR 下可直接映射）
    Buf alloc(VkDeviceSize bytes) {
        Buf b;
        b.size = bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, b.buf, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = mem_type(req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        VK_CHECK(vkMapMemory(dev, b.mem, 0, bytes, 0, &b.mapped));
        return b;
    }

    VkPipeline pipe(const std::string& spv_path) {
        auto it = pipes.find(spv_path);
        if (it != pipes.end()) return it->second;
        auto spv = read_file(spv_path);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size();
        smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
        VkShaderModule sm{};
        VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        ci.layout = pl;
        VkPipeline p{};
        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        vkDestroyShaderModule(dev, sm, nullptr);
        pipes[spv_path] = p;
        return p;
    }

    void begin() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    }
    void submit_wait() {
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
    }

    // 一次 dispatch：4 个 storage buffer（binding 0..3，可带偏移）
    void dispatch(VkPipeline p, const VkDescriptorBufferInfo* dbi4, const Push& pc,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dsl;
        VkDescriptorSet ds{};
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &ds));
        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &dbi4[i];
        }
        vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
        vkCmdDispatch(cb, gx, gy, gz);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);    }

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void fill(Buf& b, VkDeviceSize off, VkDeviceSize size, uint32_t v) {
        vkCmdFillBuffer(cb, b.buf, off, size, v);
        barrier();
    }
    void copy(Buf& src, VkDeviceSize so, Buf& dst, VkDeviceSize d_off, VkDeviceSize size) {
        VkBufferCopy r{};
        r.srcOffset = so;
        r.dstOffset = d_off;
        r.size = size;
        vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &r);
        barrier();
    }

    void reset_pool() {
        VK_CHECK(vkResetDescriptorPool(dev, dpool, 0));
    }
};

// ---------------------------------------------------------------- 数据与比较
static std::vector<float> load_bin(const std::string& path, size_t expect = 0) {
    auto raw = read_file(path);
    size_t n = raw.size() / sizeof(float);
    std::vector<float> v(n);
    std::memcpy(v.data(), raw.data(), n * sizeof(float));
    if (expect && n != expect) {
        std::printf("!! %s 大小不符: %zu vs %zu\n", path.c_str(), n, expect);
    }
    return v;
}

struct WLayer {
    int C = 0, O = 0, k = 0;
    int cm_off = -1;
    size_t w_off = 0, w_n = 0, b_off = 0, b_n = 0;
};

static double compare(const char* tag, const float* got, const float* ref, size_t n,
                      double tol) {
    double md = 0.0, sum = 0.0, amax = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        double a = std::fabs((double)ref[i]);
        if (a > amax) amax = a;
        sum += d;
        if (d > md) {
            md = d;
            worst = i;
        }
    }
    // fp16 张量核 vs fp32 参考：绝对 2e-2 或 相对 5e-3（对幅值）任一满足即通过
    const double rel = md / (amax > 1e-6 ? amax : 1.0);
    const bool ok = (md < tol) || (rel < 5e-3);
    std::printf("  %-14s maxdiff %.3e  meandiff %.3e  rel %.2e  %s\n", tag, md, sum / (double)n,
                rel, ok ? "OK" : "**FAIL**");
    if (!ok) {
        std::printf("      最差位置 %zu: got %.6f ref %.6f\n", worst, got[worst], ref[worst]);
    }
    return md;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string dir = argc > 1 ? argv[1] : "testrun_s";
    const std::string S = dir + "/";

    // ---- 读清单 ----
    auto jraw = read_file(S + "weights_rt.json");
    std::string js(jraw.begin(), jraw.end());
    auto jint = [&](const std::string& key, size_t from = 0) {
        size_t p = js.find("\"" + key + "\"", from);
        p = js.find(':', p);
        return std::atoi(js.c_str() + p + 1);
    };
    int T = 0, Hh = 0, Ww = 0;
    {
        size_t p = js.find("\"shape\"");
        T = jint("T", p);
        Hh = jint("H", p);
        Ww = jint("W", p);
    }
    const int jF = jint("num_feat");          // 64 / 48 / 96
    int jnb = jint("num_blocks");             // 20 / 12 / 30（老清单可能没有）
    std::vector<WLayer> layers;
    {
        size_t p = 0;
        while ((p = js.find("\"name\"", p)) != std::string::npos) {
            size_t pe = js.find('}', p);
            std::string obj = js.substr(p, pe - p);
            auto get = [&](const char* key) {
                size_t q = obj.find(std::string("\"") + key + "\"");
                if (q == std::string::npos) return -1;
                q = obj.find(':', q);
                return std::atoi(obj.c_str() + q + 1);
            };
            size_t q = obj.find("\"name\"");
            q = obj.find(':', q);
            size_t q1 = obj.find('"', q);
            size_t q2 = obj.find('"', q1 + 1);
            std::string name = obj.substr(q1 + 1, q2 - q1 - 1);
            WLayer L;
            L.C = get("C");
            L.O = get("O");
            L.k = get("k");
            L.w_off = (size_t)get("w_off");
            L.w_n = (size_t)get("w_n");
            L.b_off = (size_t)get("b_off");
            L.b_n = (size_t)get("b_n");
            L.cm_off = get("cm_off");
            layers.push_back(L);
            p = pe;
        }
    }
    std::printf("num_feat=%d blocks=%d  T=%d H=%d W=%d  n=%zu\n", jF, jnb, T, Hh, Ww, layers.size());
    if ((int)layers.size() < 2 * jnb + 5) {
        std::printf("清单解析异常（层数偏少）\n");
        return 1;
    }
    std::map<std::string, int> idx;
    {
        size_t p = 0;
        int i = 0;
        while ((p = js.find("\"name\"", p)) != std::string::npos) {
            size_t q = js.find(':', p);
            size_t q1 = js.find('"', q);
            size_t q2 = js.find('"', q1 + 1);
            idx[js.substr(q1 + 1, q2 - q1 - 1)] = i++;
            p = q2;
        }
    }
    if (jnb <= 0) {                       // 老版清单没有 num_blocks，按层名兜底
        for (const auto& kv : idx)
            if (kv.first.rfind("forward_", 0) == 0) ++jnb;
    }
    float prelu1 = 0.0263f, prelu2 = 0.0393f;
    {
        size_t p = js.find("\"prelu\"");
        if (p != std::string::npos) {
            size_t b = js.find('[', p);
            prelu1 = (float)std::atof(js.c_str() + b + 1);
            size_t c = js.find(',', b);
            prelu2 = (float)std::atof(js.c_str() + c + 1);
        }
    }
    const float act = 0.1f;

    // ---- 设备与显存 ----
    Gpu g;
    g.init();

    auto wblob = read_file(S + "weights_rt.bin");
    Buf bw = g.alloc(wblob.size());
    std::memcpy(bw.mapped, wblob.data(), wblob.size());
    auto cmblob = read_file(S + "weights_cm.bin");
    Buf bwcm = g.alloc(cmblob.size());
    std::memcpy(bwcm.mapped, cmblob.data(), cmblob.size());

    const size_t F = (size_t)jF;
    const int up1O = layers[idx["up1"]].O;        // 4F
    const int up2O = layers[idx["up2"]].O;        // 128
    const int clC = layers[idx["conv_last"]].C;   // 32
    const size_t px = (size_t)Hh * Ww;
    Buf bx = g.alloc((VkDeviceSize)T * px * 3 * 4);
    Buf bfeat = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bfwd = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bbwd = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bp0 = g.alloc((VkDeviceSize)px * F * 4);
    Buf bp1 = g.alloc((VkDeviceSize)px * F * 4);
    Buf bcatus = g.alloc((VkDeviceSize)px * F * 4);
    Buf bcat = g.alloc((VkDeviceSize)px * 2 * F * 4);
    Buf bfused = g.alloc((VkDeviceSize)px * F * 4);
    Buf bconv1 = g.alloc((VkDeviceSize)px * 4 * (size_t)up1O * 4);
    Buf bup1 = g.alloc((VkDeviceSize)px * 4 * F * 4);
    Buf bconv2 = g.alloc((VkDeviceSize)px * 16 * (size_t)up2O * 4);
    Buf bup2 = g.alloc((VkDeviceSize)px * 16 * (size_t)clC * 4);
    Buf blast = g.alloc((VkDeviceSize)px * 16 * 3 * 4);
    Buf bout = g.alloc((VkDeviceSize)T * px * 16 * 4);
    Buf btmp = g.alloc((VkDeviceSize)T * px * 16 * 4);   // 8bit 结果暂存（uint 打包）

    {
        auto x = read_file(S + "x.bin");
        std::memcpy(bx.mapped, x.data(), x.size());
    }

    const VkPipeline pConv64 = g.pipe("shaders/conv3x3_c64.spv");
    const VkPipeline pConvGen = g.pipe("shaders/conv_gen.spv");
    const VkPipeline pConvSmall = g.pipe("shaders/conv3x3_smallc.spv");
    auto cm_pipe = [&](int C) {            // 张量核变体按 C 选（conv3x3_cm.spv = C64）
        std::string p = "shaders/conv3x3_cm";
        if (C != 64) p += "_" + std::to_string(C);
        return g.pipe(p + ".spv");
    };
    const VkPipeline pShuffle = g.pipe("shaders/shuffle_prelu.spv");
    const VkPipeline pFinal = g.pipe("shaders/bilinear4_add_u8.spv");
    const VkPipeline pAdd = g.pipe("shaders/add.spv");
    const VkPipeline pCat = g.pipe("shaders/cat2.spv");

    // 便捷封装
    auto layer_bufs = [&](const std::string& name, int k, VkDescriptorBufferInfo* dbi,
                          Buf& in, VkDeviceSize in_off, Buf& out, VkDeviceSize out_off,
                          VkDeviceSize out_size) {
        const WLayer& L = layers[idx[name]];
        dbi[0] = {in.buf, in_off, VK_WHOLE_SIZE};
        dbi[1] = {bw.buf, L.w_off, VK_WHOLE_SIZE};
        dbi[2] = {bw.buf, L.b_off, VK_WHOLE_SIZE};
        dbi[3] = {out.buf, out_off, out_size};
    };

    auto run_conv64 = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                          VkDeviceSize out_off, int h, int w, float slope) {
        const WLayer& L = layers[idx[name]];
        const int O = L.O;          // 输出通道一律来自清单，别在外面写死
        if (L.cm_off >= 0) {
            // 张量核路径：权重 [tap][cc][ocg][k][n] f16，O 补齐到 64 的倍数
            VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                             {bwcm.buf, (VkDeviceSize)L.cm_off, VK_WHOLE_SIZE},
                                             {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                             {out.buf, out_off, VK_WHOLE_SIZE}};
            Push pc;
            pc.C_ = L.C;
            pc.H = h;
            pc.W = w;
            pc.O = O;
            pc.slope = slope;
            pc.k = (O + 63) / 64 * 64;
            g.dispatch(cm_pipe(L.C), dbi, pc, (uint32_t)((w + 31) / 32), (uint32_t)((h + 3) / 4),
                       (uint32_t)((O + 63) / 64));
            return;
        }
        VkDescriptorBufferInfo dbi[4];
        layer_bufs(name, 3, dbi, in, in_off, out, out_off, VK_WHOLE_SIZE);
        Push pc;
        pc.C_ = L.C;
        pc.H = h;
        pc.W = w;
        pc.O = O;
        pc.slope = slope;
        if (L.C == 64) {
            g.dispatch(pConv64, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
        } else {
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
    };
    auto run_convg = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                         VkDeviceSize out_off, int h, int w, float slope) {
        const WLayer& L = layers[idx[name]];
        VkDescriptorBufferInfo dbi[4];
        layer_bufs(name, L.k, dbi, in, in_off, out, out_off, VK_WHOLE_SIZE);
        Push pc;
        pc.C_ = L.C;
        pc.H = h;
        pc.W = w;
        pc.O = L.O;
        pc.slope = slope;
        pc.k = L.k;
        if (L.C <= 32) {
            g.dispatch(pConvSmall, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8));
        } else {
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
    };

    // ---- 前向 ----
    const float slope_act = act;
    g.reset_pool();
    g.begin();

    // feat_extract（逐帧）
    for (int t = 0; t < T; ++t) {
        run_convg("feat_extract", bx, (VkDeviceSize)t * px * 3 * 4, bfeat,
                  (VkDeviceSize)t * px * F * 4, Hh, Ww, slope_act);
    }
    g.submit_wait();

    // 隔离测试：feat[0] 连跑 3 个 block（prop=0），逐级接到 bfwd 的 3 个槽里
    g.reset_pool();
    g.begin();
    {
        Buf* cin = &bp0;
        Buf* cout = &bp1;
        g.copy(bfeat, 0, *cin, 0, (VkDeviceSize)px * F * 4);
        for (int b = 0; b < 3; ++b) {
            run_conv64("forward_" + std::to_string(b), *cin, 0, *cout, 0, Hh, Ww, act);
            g.copy(*cout, 0, bfwd, (VkDeviceSize)b * px * F * 4, (VkDeviceSize)px * F * 4);
            std::swap(cin, cout);
        }
    }
    g.submit_wait();
    for (int b = 0; b < 3; ++b) {
        std::string tag = "chain" + std::to_string(b + 1);
        auto ref = load_bin(S + tag + ".bin", px * F);
        compare(tag.c_str(), (const float*)bfwd.mapped + (size_t)b * px * F, ref.data(),
                px * F, 2e-2);
        if (b == 0) {
            // 诊断：按 (x%32 的左右半块, y%2 的奇偶行) 分组看误差分布，定位 CM 分块映射问题
            const float* got = (const float*)bfwd.mapped;
            double gmax[2][2] = {{0, 0}, {0, 0}};
            long zeros_got = 0, zeros_ref = 0;
            for (int y = 0; y < Hh; ++y) {
                for (int x = 0; x < Ww; ++x) {
                    for (int c = 0; c < (int)F; ++c) {
                        const size_t i = ((size_t)y * Ww + x) * F + c;
                        const double d = std::fabs((double)got[i] - (double)ref[i]);
                        const int gx = (x % 32) / 16, gy = y % 2;
                        if (d > gmax[gy][gx]) gmax[gy][gx] = d;
                        if (got[i] == 0.0f) ++zeros_got;
                        if (ref[i] == 0.0f) ++zeros_ref;
                    }
                }
            }
            std::printf("       诊断: (行奇偶, 左右半块) 最大误差 %.2e %.2e / %.2e %.2e  零值 got %ld ref %ld\n",
                        gmax[0][0], gmax[0][1], gmax[1][0], gmax[1][1], zeros_got, zeros_ref);
        }
    }
    g.reset_pool();
    g.begin();
    // forward / backward 链
    // 模型语义（注意两点）：
    //   1. feat_prop = Net20(feats[:, i] + feat_prop) —— "加 feat" 每帧只做一次；
    //   2. feat_prop 跨帧延续（上一帧的输出作为下一帧的 prop），只在序列开头清零。
    const int nb = jnb;
    Buf* scratch[2] = {&bp0, &bp1};
    auto run_dir = [&](bool forward, Buf& dst_all) {
        Buf* prop = scratch[0];
        g.fill(*prop, 0, (VkDeviceSize)px * F * 4, 0);          // prop_{-1} = 0
        for (int k = 0; k < T; ++k) {
            const int t = forward ? k : (T - 1 - k);
            {
                VkDescriptorBufferInfo dbi[4];
                dbi[0] = {bfeat.buf, (VkDeviceSize)t * px * F * 4, VK_WHOLE_SIZE};
                dbi[1] = {prop->buf, 0, VK_WHOLE_SIZE};
                dbi[2] = {bw.buf, 0, 1};
                dbi[3] = {bcatus.buf, 0, VK_WHOLE_SIZE};
                Push pc;
                pc.H = (int32_t)(px * F);
                g.dispatch(pAdd, dbi, pc, (uint32_t)((px * F + 63) / 64));
            }
            Buf* src = &bcatus;
            int w = 0;
            for (int b = 0; b < nb; ++b) {
                const std::string name =
                    (forward ? "forward_" : "backward_") + std::to_string(b);
                run_conv64(name, *src, 0, *scratch[w], 0, Hh, Ww, slope_act);
                src = scratch[w];
                w ^= 1;
            }
            g.copy(*src, 0, dst_all, (VkDeviceSize)t * px * F * 4, (VkDeviceSize)px * F * 4);
            prop = src;
        }
    };
    run_dir(true, bfwd);
    run_dir(false, bbwd);
    g.submit_wait();

    // 调试：add 与 fwd0
    g.reset_pool();
    g.begin();
    {
        VkDescriptorBufferInfo dbi[4];
        dbi[0] = {bfeat.buf, (VkDeviceSize)1 * px * F * 4, VK_WHOLE_SIZE};
        dbi[1] = {bfwd.buf, 0, VK_WHOLE_SIZE};
        dbi[2] = {bw.buf, 0, 1};
        dbi[3] = {bcatus.buf, 0, VK_WHOLE_SIZE};
        Push pc;
        pc.H = (int32_t)(px * F);
        g.dispatch(pAdd, dbi, pc, (uint32_t)((px * F + 63) / 64));
    }
    g.submit_wait();
    {
        auto ref = load_bin(S + "fwd0.bin", px * F);
        compare("fwd0", (const float*)bfwd.mapped, ref.data(), px * F, 2e-2);
        auto ref2 = load_bin(S + "add1.bin", px * F);
        compare("add1", (const float*)bcatus.mapped, ref2.data(), px * F, 2e-2);
        std::printf("       add1 前 4 个: %.4f %.4f %.4f %.4f | 参考 %.4f %.4f %.4f %.4f\n",
                    ((const float*)bcatus.mapped)[0], ((const float*)bcatus.mapped)[1],
                    ((const float*)bcatus.mapped)[2], ((const float*)bcatus.mapped)[3],
                    ref2[0], ref2[1], ref2[2], ref2[3]);
    }

    // ---- 逐帧 fusion + 上采样 ----
    g.reset_pool();
    g.begin();
    for (int t = 0; t < T; ++t) {
        // cat
        {
            VkDescriptorBufferInfo dbi[4];
            dbi[0] = {bfwd.buf, (VkDeviceSize)t * px * F * 4, VK_WHOLE_SIZE};
            dbi[1] = {bbwd.buf, (VkDeviceSize)t * px * F * 4, VK_WHOLE_SIZE};
            dbi[2] = {bw.buf, 0, 1};
            dbi[3] = {bcat.buf, 0, VK_WHOLE_SIZE};
            Push pc;
            pc.C_ = (int32_t)F;
            pc.H = (int32_t)F;
            pc.W = (int32_t)px;
            g.dispatch(pCat, dbi, pc, (uint32_t)((px + 63) / 64));
        }
        run_convg("fusion", bcat, 0, bfused, 0, Hh, Ww, 0.0f);

        // up1: conv C->4F @ H,W -> shuffle -> F @ 2H,2W
        run_conv64("up1", bfused, 0, bconv1, 0, Hh, Ww, 0.0f);
        {
            VkDescriptorBufferInfo dbi[4];
            dbi[0] = {bconv1.buf, 0, VK_WHOLE_SIZE};
            dbi[1] = {bw.buf, 0, 1};
            dbi[2] = {bw.buf, 0, 1};
            dbi[3] = {bup1.buf, 0, VK_WHOLE_SIZE};
            Push pc;
            pc.C_ = (int32_t)F;
            pc.H = Hh;
            pc.W = Ww;
            pc.slope = prelu1;
            g.dispatch(pShuffle, dbi, pc, (uint32_t)((Ww + 15) / 16), (uint32_t)Hh);
        }
        // up2: conv 4F->128 @ 2H,2W -> shuffle -> 32 @ 4H,4W
        run_conv64("up2", bup1, 0, bconv2, 0, Hh * 2, Ww * 2, 0.0f);
        {
            VkDescriptorBufferInfo dbi[4];
            dbi[0] = {bconv2.buf, 0, VK_WHOLE_SIZE};
            dbi[1] = {bw.buf, 0, 1};
            dbi[2] = {bw.buf, 0, 1};
            dbi[3] = {bup2.buf, 0, VK_WHOLE_SIZE};
            Push pc;
            pc.C_ = clC;
            pc.H = Hh * 2;
            pc.W = Ww * 2;
            pc.slope = prelu2;
            g.dispatch(pShuffle, dbi, pc, (uint32_t)((Ww * 2 + 15) / 16), (uint32_t)(Hh * 2));
        }
        run_convg("conv_last", bup2, 0, blast, 0, Hh * 4, Ww * 4, 0.0f);
        // 最终：bilinear4x + 残差 + u8（直接写进 bout 的第 t 帧）
        {
            VkDescriptorBufferInfo dbi[4];
            dbi[0] = {bx.buf, (VkDeviceSize)t * px * 3 * 4, VK_WHOLE_SIZE};
            dbi[1] = {blast.buf, 0, VK_WHOLE_SIZE};
            dbi[2] = {bw.buf, 0, 1};
            dbi[3] = {bout.buf, (VkDeviceSize)t * px * 16 * 4, VK_WHOLE_SIZE};
            Push pc;
            pc.H = Hh;
            pc.W = Ww;
            pc.O = 4;
            g.dispatch(pFinal, dbi, pc, (uint32_t)((px * 16 + 63) / 64));
        }
    }
    g.submit_wait();

    // ---- 对拍 ----
    std::printf("逐级对拍（fp32 参考，容差 2e-2 或相对 5e-3）:\n");
    auto cmp_buf = [&](const char* tag, const char* file, Buf& b, size_t n, double tol,
                       size_t byte_off = 0) {
        auto ref = load_bin(S + file, n);
        return compare(tag, (const float*)b.mapped + byte_off / sizeof(float), ref.data(), n, tol);
    };
    cmp_buf("feat", "feat.bin", bfeat, (size_t)T * px * F, 2e-2);
    cmp_buf("fwd_last", "fwd_last.bin", bfwd, px * F, 2e-2, (size_t)(T - 1) * px * F * 4);
    cmp_buf("bwd_last", "bwd_last.bin", bbwd, px * F, 2e-2);
    cmp_buf("fused0", "fused0.bin", bfused, px * F, 2e-2);
    cmp_buf("up1_0", "up1_0.bin", bup1, px * 4 * F, 2e-2);
    cmp_buf("up2_0", "up2_0.bin", bup2, px * 16 * 32, 2e-2);
    cmp_buf("conv_last_0", "conv_last_0.bin", blast, px * 16 * 3, 2e-2);

    // 最终 8bit：与参考 u8 比，允许 ±1
    {
        auto ref = read_file(S + "final_u8.bin");
        const int npx = (int)(px * 16);
        const uint32_t* got = (const uint32_t*)bout.mapped;
        int worst = 0, cnt_bad = 0;
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < npx; ++p) {
                uint32_t gv = got[t * npx + p];
                const unsigned char* rv = (const unsigned char*)ref.data() + ((size_t)t * npx + p) * 3;
                for (int c = 0; c < 3; ++c) {
                    int d = std::abs((int)((gv >> (c * 8)) & 255) - (int)rv[c]);
                    if (d > worst) worst = d;
                    if (d > 1) ++cnt_bad;
                }
            }
        }
        std::printf("  %-14s 最大差 %d 级, 超 1 级的通道 %d 个  %s\n", "final_u8", worst,
                    cnt_bad, worst <= 1 ? "OK" : "**FAIL**");
    }

    // ---- 速度 ----
    {
        const int reps = 3;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) {
            g.reset_pool();
            g.begin();
            for (int t = 0; t < T; ++t)
                run_convg("feat_extract", bx, (VkDeviceSize)t * px * 3 * 4, bfeat,
                          (VkDeviceSize)t * px * F * 4, Hh, Ww, slope_act);
            for (int t = 0; t < T; ++t) {
                Buf* cur = &bp0;
                Buf* nxt = &bp1;
                for (int b = 0; b < nb; ++b) {
                    run_conv64("forward_" + std::to_string(b), bfeat, (VkDeviceSize)t * px * F * 4,
                               *nxt, 0, Hh, Ww, slope_act);
                    std::swap(cur, nxt);
                }
            }
            g.submit_wait();
        }
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count() / reps;
        std::printf("速度: %d 帧(仅 feat+forward 链) = %.1f ms  -> %.2f ms/帧前向链\n", T, ms,
                    ms / T);
    }

    vkDeviceWaitIdle(g.dev);
    return 0;
}
