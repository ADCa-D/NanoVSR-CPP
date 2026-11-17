// bench_model.cpp -- 只跑模型、不做对拍，用来量真实 GPU 时间。
//
// 和 model_test 的区别：
//   * 不读参考文件、不逐元素比对（那里会占掉大量 CPU 时间，GPU 反而闲着）；
//   * 按阶段分子提交，用时间戳算每段的 ms/帧；
//   * 同时给出每段的 GFLOP 和实测 TFLOPS，便于判断离峰值还有多远。
//
// 用法: bench_model.exe <testrun_dir> [frames]
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
        std::printf("device: %s\n", p.deviceName);
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qfam = i;
                break;
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

    Buf alloc(VkDeviceSize bytes) {
        Buf b;
        b.size = bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
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
        barrier();
    }
    void reset_pool() { VK_CHECK(vkResetDescriptorPool(dev, dpool, 0)); }
};

struct WLayer {
    int C = 0, O = 0, k = 0;
    int cm_off = -1;
    size_t w_off = 0, b_off = 0;
};

static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string S = std::string(argc > 1 ? argv[1] : "testrun_s") + "/";
    const int frames_arg = argc > 2 ? std::atoi(argv[2]) : 0;

    auto jraw = read_file(S + "weights_rt.json");
    std::string js(jraw.begin(), jraw.end());
    auto jint = [&](const std::string& key, size_t from) {
        size_t p = js.find("\"" + key + "\"", from);
        p = js.find(':', p);
        return std::atoi(js.c_str() + p + 1);
    };
    size_t shape = js.find("\"shape\"");
    int T0 = jint("T", shape), Hh = jint("H", shape), Ww = jint("W", shape);
    int jFb = jint("num_feat", 0);          // 64 / 48 / 96
    int jnbb = jint("num_blocks", 0);       // 20 / 12 / 30（老清单可能没有）

    std::vector<WLayer> layers;
    std::map<std::string, int> idx;
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
            size_t q = obj.find(':', obj.find("\"name\""));
            size_t q1 = obj.find('"', q);
            size_t q2 = obj.find('"', q1 + 1);
            WLayer L;
            L.C = get("C");
            L.O = get("O");
            L.k = get("k");
            L.w_off = (size_t)get("w_off");
            L.b_off = (size_t)get("b_off");
            L.cm_off = get("cm_off");
            idx[obj.substr(q1 + 1, q2 - q1 - 1)] = (int)layers.size();
            layers.push_back(L);
            p = pe;
        }
    }
    if (jnbb <= 0) {                       // 老版清单没有 num_blocks，按层名兜底
        for (const auto& kv : idx)
            if (kv.first.rfind("forward_", 0) == 0) ++jnbb;
    }
    const int T = frames_arg > 0 ? frames_arg : T0;
    std::printf("T=%d H=%d W=%d  层数 %zu\n", T, Hh, Ww, layers.size());

    Gpu g;
    g.init();
    auto wblob = read_file(S + "weights_rt.bin");
    Buf bw = g.alloc(wblob.size());
    std::memcpy(bw.mapped, wblob.data(), wblob.size());
    auto cmblob = read_file(S + "weights_cm.bin");
    Buf bwcm = g.alloc(cmblob.size());
    std::memcpy(bwcm.mapped, cmblob.data(), cmblob.size());

    const size_t F = (size_t)jFb;
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

    // 输入填随机数（bench 不关心内容，但不能是 NaN）
    {
        float* p = (float*)bx.mapped;
        size_t n = (size_t)T * px * 3;
        for (size_t i = 0; i < n; ++i) p[i] = (float)((i * 1103515245u) % 1000) / 1000.0f;
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

    auto run_conv64 = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                          VkDeviceSize out_off, int h, int w, float slope) {
        const WLayer& L = layers[idx[name]];
        const int O = L.O;
        // k==1（fusion）走张量核需要额外的流式内核，本工具没接；退回 SIMT 即可
        if (L.cm_off >= 0 && L.k == 3) {
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
            g.dispatch(cm_pipe(L.C), dbi, pc, (uint32_t)((w + 15) / 16), (uint32_t)((h + 7) / 8), (uint32_t)((O + 63) / 64));
            return;
        }
        VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.w_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                         {out.buf, out_off, VK_WHOLE_SIZE}};
        Push pc;
        pc.C_ = L.C;
        pc.H = h;
        pc.W = w;
        pc.O = O;
        pc.slope = slope;
        if (L.C == 64)
            g.dispatch(pConv64, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
        else
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
    };
    auto run_convg = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                         VkDeviceSize out_off, int h, int w, float slope) {
        const WLayer& L = layers[idx[name]];
        VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.w_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                         {out.buf, out_off, VK_WHOLE_SIZE}};
        Push pc;
        pc.C_ = L.C;
        pc.H = h;
        pc.W = w;
        pc.O = L.O;
        pc.slope = slope;
        pc.k = L.k;
        if (L.C <= 32) {
            // 小通道数走专用 kernel（feat_extract C=3、conv_last C=32）
            g.dispatch(pConvSmall, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8));
        } else {
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
    };

    const float act = 0.1f;
    const int nb = jnbb;
    const int reps = std::max(3, T) + 1;      // 第 1 轮只做预热（驱动首次 dispatch 会 JIT）
    double acc[8] = {0};

    for (int rep = 0; rep < reps; ++rep) {
        g.reset_pool();
        double t[8] = {0};
        double a = now_ms();
        g.begin();
        for (int t_ = 0; t_ < T; ++t_)
            run_convg("feat_extract", bx, (VkDeviceSize)t_ * px * 3 * 4, bfeat,
                      (VkDeviceSize)t_ * px * F * 4, Hh, Ww, act);
        g.submit_wait();
        t[0] = now_ms() - a;

        auto run_dir = [&](bool forward, Buf& dst_all) {
            Buf* scratch[2] = {&bp0, &bp1};
            Buf* prop = scratch[0];
            g.fill(*prop, 0, (VkDeviceSize)px * F * 4, 0);
            for (int k = 0; k < T; ++k) {
                const int tt = forward ? k : (T - 1 - k);
                VkDescriptorBufferInfo dbi[4] = {{bfeat.buf, (VkDeviceSize)tt * px * F * 4, VK_WHOLE_SIZE},
                                                 {prop->buf, 0, VK_WHOLE_SIZE},
                                                 {bw.buf, 0, 1},
                                                 {bcatus.buf, 0, VK_WHOLE_SIZE}};
                Push pc;
                pc.H = (int32_t)(px * F);
                g.dispatch(pAdd, dbi, pc, (uint32_t)((px * F + 63) / 64));
                Buf* src = &bcatus;
                int w = 0;
                for (int b = 0; b < nb; ++b) {
                    const std::string name =
                        (forward ? "forward_" : "backward_") + std::to_string(b);
                    run_conv64(name, *src, 0, *scratch[w], 0, Hh, Ww, act);
                    src = scratch[w];
                    w ^= 1;
                }
                g.copy(*src, 0, dst_all, (VkDeviceSize)tt * px * F * 4,
                       (VkDeviceSize)px * F * 4);
                prop = src;
            }
        };
        a = now_ms();
        g.begin();
        run_dir(true, bfwd);
        g.submit_wait();
        t[1] = now_ms() - a;

        a = now_ms();
        g.begin();
        run_dir(false, bbwd);
        g.submit_wait();
        t[2] = now_ms() - a;

        // 每帧后半段（逐算子计时）
        double t3 = 0;
        for (int tt = 0; tt < T; ++tt) {
            double m[6] = {0};
            g.begin();
            {
                VkDescriptorBufferInfo dbi[4] = {
                    {bfwd.buf, (VkDeviceSize)tt * px * F * 4, VK_WHOLE_SIZE},
                    {bbwd.buf, (VkDeviceSize)tt * px * F * 4, VK_WHOLE_SIZE},
                    {bw.buf, 0, 1},
                    {bcat.buf, 0, VK_WHOLE_SIZE}};
                Push pc;
                pc.C_ = (int32_t)F;
                pc.H = (int32_t)F;
                pc.W = (int32_t)px;
                g.dispatch(pCat, dbi, pc, (uint32_t)((px + 63) / 64));
            }
            run_convg("fusion", bcat, 0, bfused, 0, Hh, Ww, 0.0f);
            g.submit_wait();
            double a3 = now_ms();
            g.begin();
            run_conv64("up1", bfused, 0, bconv1, 0, Hh, Ww, 0.0f);
            g.submit_wait();
            m[0] = now_ms() - a3;
            a3 = now_ms();
            g.begin();
            {
                VkDescriptorBufferInfo dbi[4] = {{bconv1.buf, 0, VK_WHOLE_SIZE},
                                                 {bw.buf, 0, 1},
                                                 {bw.buf, 0, 1},
                                                 {bup1.buf, 0, VK_WHOLE_SIZE}};
                Push pc;
                pc.C_ = (int32_t)F;
                pc.H = Hh;
                pc.W = Ww;
                pc.slope = 0.0263f;
                g.dispatch(pShuffle, dbi, pc, (uint32_t)((Ww + 15) / 16), (uint32_t)Hh);
            }
            g.submit_wait();
            m[1] = now_ms() - a3;
            a3 = now_ms();
            g.begin();
            run_conv64("up2", bup1, 0, bconv2, 0, Hh * 2, Ww * 2, 0.0f);
            g.submit_wait();
            m[2] = now_ms() - a3;
            a3 = now_ms();
            g.begin();
            {
                VkDescriptorBufferInfo dbi[4] = {{bconv2.buf, 0, VK_WHOLE_SIZE},
                                                 {bw.buf, 0, 1},
                                                 {bw.buf, 0, 1},
                                                 {bup2.buf, 0, VK_WHOLE_SIZE}};
                Push pc;
                pc.C_ = clC;
                pc.H = Hh * 2;
                pc.W = Ww * 2;
                pc.slope = 0.0393f;
                g.dispatch(pShuffle, dbi, pc, (uint32_t)((Ww * 2 + 15) / 16), (uint32_t)(Hh * 2));
            }
            g.submit_wait();
            m[3] = now_ms() - a3;
            a3 = now_ms();
            g.begin();
            run_convg("conv_last", bup2, 0, blast, 0, Hh * 4, Ww * 4, 0.0f);
            g.submit_wait();
            m[4] = now_ms() - a3;
            a3 = now_ms();
            g.begin();
            {
                VkDescriptorBufferInfo dbi[4] = {
                    {bx.buf, (VkDeviceSize)tt * px * 3 * 4, VK_WHOLE_SIZE},
                    {blast.buf, 0, VK_WHOLE_SIZE},
                    {bw.buf, 0, 1},
                    {bout.buf, (VkDeviceSize)tt * px * 16 * 4, VK_WHOLE_SIZE}};
                Push pc;
                pc.H = Hh;
                pc.W = Ww;
                pc.O = 4;
                g.dispatch(pFinal, dbi, pc, (uint32_t)((px * 16 + 63) / 64));
            }
            g.submit_wait();
            m[5] = now_ms() - a3;
            if (rep == reps - 1 && tt == 0) {
                std::printf("  细分: up1 %.1f   shuffle1 %.1f   up2 %.1f   shuffle2 %.1f"
                            "   conv_last %.1f   bilinear %.1f ms\n",
                            m[0], m[1], m[2], m[3], m[4], m[5]);
            }
            t3 += m[0] + m[1] + m[2] + m[3] + m[4] + m[5];
        }

        t[3] = t3 / T;
        if (rep > 0)
            for (int i = 0; i < 5; ++i) acc[i] += t[i];
        if (rep == 0) {
            std::printf("预热轮(不计入): feat %.2f  fwd %.2f  bwd %.2f  up(每帧) %.2f ms\n", t[0] / T,
                        t[1] / T, t[2] / T, t[3]);
        }
    }

    const int reps_ok = reps - 1;
    double per_frame = (acc[0] / reps_ok / T) + (acc[1] / reps_ok / T) + (acc[2] / reps_ok / T) +
                       (acc[3] / reps_ok);
    std::printf("\n平均（%d 轮，T=%d，不含预热）:\n", reps_ok, T);
    std::printf("  feat     %7.2f ms/帧\n", acc[0] / reps_ok / T);
    std::printf("  forward  %7.2f ms/帧   (%d 层)\n", acc[1] / reps_ok / T, jnbb);
    std::printf("  backward %7.2f ms/帧   (%d 层)\n", acc[2] / reps_ok / T, jnbb);
    std::printf("  up+out   %7.2f ms/帧   (fusion/up1/up2/conv_last/bilinear)\n", acc[3] / reps_ok);
    std::printf("  合计     %7.2f ms/帧  -> %.1f FPS\n", per_frame, 1000.0 / per_frame);

    // 理论计算量（用于算实测 TFLOPS）
    const double Fv = (double)F, u1 = (double)up1O, u2 = (double)up2O, cl = (double)clC;
    const double conv1_g = 2.0 * Fv * Fv * 9.0 * px / 1e9;
    const double gflop = conv1_g * (2.0 * jnbb)                         // nb 个 forward + nb 个 backward
                         + 2.0 * 3 * Fv * 9.0 * px / 1e9                // feat
                         + 2.0 * (2.0 * Fv) * Fv * px / 1e9             // fusion 1x1
                         + 2.0 * Fv * u1 * 9.0 * px / 1e9               // up1
                         + 2.0 * u1 * u2 * 9.0 * px * 4 / 1e9           // up2 @ 2x2
                         + 2.0 * cl * 3 * 9.0 * px * 16 / 1e9;          // conv_last @ 4x4
    std::printf("  计算量   %7.1f GFLOP/帧 -> 实测 %.2f TFLOPS\n", gflop,
                gflop / (per_frame * 1e-3) / 1000.0);

    // ---- 单算子微基准：连续 40 次 dispatch，量 kernel 本身 ----
    {
        g.reset_pool();
        g.begin();
        for (int i = 0; i < 40; ++i)
            run_convg("feat_extract", bx, 0, bfeat, 0, Hh, Ww, act);
        double a = now_ms();
        g.submit_wait();
        double ms = now_ms() - a;
        double gf = 2.0 * 3 * Fv * 9.0 * px / 1e9;
        std::printf("\n微基准 feat_extract x40: %.2f ms/dispatch (single frame)  %.2f GFLOP -> %.2f TFLOPS\n",
                    ms / 40, gf, gf / (ms / 40 * 1e-3) / 1000.0);

        g.reset_pool();
        g.begin();
        for (int i = 0; i < 20; ++i)
            run_conv64("forward_0", bfeat, 0, bp1, 0, Hh, Ww, act);
        a = now_ms();
        g.submit_wait();
        ms = now_ms() - a;
        gf = 2.0 * Fv * Fv * 9.0 * px / 1e9;
        std::printf("微基准 forward_0 x20: %.2f ms/dispatch  %.2f GFLOP -> %.2f TFLOPS\n", ms / 20,
                    gf, gf / (ms / 20 * 1e-3) / 1000.0);
    }

    vkDeviceWaitIdle(g.dev);
    return 0;
}

