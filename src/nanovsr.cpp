// nanovsr.cpp -- NanoVSR 4x 视频超分（Vulkan compute，不依赖 torch/CUDA）
//
// 用法:
//   nanovsr.exe --in in.mp4 --out out.mp4 [--chunk 10] [--crf 18] [--limit 60]
//               [--ffmpeg D:\ffmpeg-9.0.1\ffmpeg.exe] [--bench]
//
// 流程: ffmpeg 解码 -> 共享内存上传 -> u8->f32 -> feat -> 双向循环 -> fusion/上采样
//       -> 4x 双线性残差 -> 打包 u8 -> ffmpeg 编码
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    bool mem_budget_available = false;

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
        std::printf("GPU: %s\n", p.deviceName);

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
        uint32_t nex = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nex, nullptr);
        std::vector<VkExtensionProperties> exts_avail(nex);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nex, exts_avail.data());
        bool has_budget = false;
        for (auto& e : exts_avail)
            if (std::strcmp(e.extensionName, "VK_EXT_memory_budget") == 0) has_budget = true;
        std::vector<const char*> exts = {"VK_KHR_cooperative_matrix"};
        if (has_budget) exts.push_back("VK_EXT_memory_budget");
        mem_budget_available = has_budget;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &storage16;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)exts.size();
        dci.ppEnabledExtensionNames = exts.data();
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
        dpci.maxSets = 16384;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
    }

    // 驱动建议的显存预算（字节）。没有 VK_EXT_memory_budget 就退回到堆大小的 60%。
    VkDeviceSize device_budget() {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp2.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(pd, &mp2);
        VkDeviceSize best = 0;
        for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
            if (!(mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                continue;
            VkDeviceSize b = mem_budget_available ? bp.heapBudget[i] : 0;
            if (b == 0) b = (VkDeviceSize)(mp2.memoryProperties.memoryHeaps[i].size * 0.6);
            if (b > best) best = b;
        }
        return best;
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

    // 系统内存 staging：ReBAR 显存被 CPU 读取时走 PCIe 且不缓存，非常慢
    Buf alloc_host(VkDeviceSize bytes) {
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
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        uint32_t pick = UINT32_MAX;
        // 优先带 HOST_CACHED 的（CPU 读取才快）；type 2 是 write-combined，读起来像 PCIe 一样慢
        for (int pass = 0; pass < 2 && pick == UINT32_MAX; ++pass) {
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
                const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
                if (!(req.memoryTypeBits & (1u << i))) continue;
                if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
                if (!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
                if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) continue;
                if (pass == 0 && !(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) continue;
                pick = i;
                break;
            }
        }
        if (pick == UINT32_MAX) return alloc(bytes);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = pick;
        VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        VK_CHECK(vkMapMemory(dev, b.mem, 0, bytes, 0, &b.mapped));
        return b;
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
    size_t w_off = 0, b_off = 0;
    int cm_off = -1;
};

// ---------------------------------------------------------------- 进程工具
// _popen 会把命令交给 `cmd /c`，而 cmd 对"以引号开头且引号多于两个"的命令会剥掉
// 首尾引号，导致 "prog.exe" args "file" 被拆坏。外面再包一层引号即可。
static FILE* popen_w(const std::string& cmd) {
    return _popen(("\"" + cmd + "\"").c_str(), "wb");
}
static FILE* popen_r(const std::string& cmd) {
    return _popen(("\"" + cmd + "\"").c_str(), "rb");
}
static int pclose_w(FILE* f) { return _pclose(f); }

static bool run_ok(const std::string& cmd) {
    FILE* f = _popen(("\"" + cmd + "\"").c_str(), "r");
    if (!f) return false;
    char buf[512];
    while (std::fread(buf, 1, sizeof(buf), f) > 0) {
    }
    return _pclose(f) == 0;
}

static std::string run_capture(const std::string& cmd) {
    std::string out;
    FILE* f = _popen(("\"" + cmd + "\"").c_str(), "r");
    if (!f) return out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    _pclose(f);
    return out;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string in_path, out_path, ffmpeg = "D:\\ffmpeg-9.0.1\\ffmpeg.exe";
    std::string ffprobe = "D:\\ffmpeg-9.0.1\\ffprobe.exe";
    std::string model_dir = "models_rt/nanovsr_1.7m";
    int chunk_arg = 0, crf = -1, limit = 0, tile_arg_w = 0, tile_arg_h = 0;
    double budget_arg_gb = 0.0;
    std::string downscale = "lanczos";
    bool supersample = false;   // 默认让模型工作在交付分辨率（保住模型效果）；true=先跑满4x再缩   // libplacebo downscaler: spline36|sinc|mitchell|... / lanczos=CPU
    std::string codec_pref = "auto";
    std::string max_out = "4k";       // 4k(8.9Mp, 稳) | 8k(33.2Mp, 需要能编 8K 的编码器)   // auto|av1|av1cpu|hevc|h264   // lanczos(CPU,快) | sinc(GPU,抗混叠更好)
    std::string preset = "balanced";
    bool bench_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--in") in_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--chunk") chunk_arg = std::atoi(next().c_str());
        else if (a == "--crf") crf = std::atoi(next().c_str());
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--tile") { int tw_ = 0, th_ = 0; std::sscanf(next().c_str(), "%dx%d", &tw_, &th_); tile_arg_w = tw_; tile_arg_h = th_; }
        else if (a == "--ffmpeg") ffmpeg = next();
        else if (a == "--budget") budget_arg_gb = std::atof(next().c_str());
        else if (a == "--preset") preset = next();
        else if (a == "--downscale") downscale = next();
        else if (a == "--supersample") supersample = true;
        else if (a == "--codec") codec_pref = next();
        else if (a == "--max-out") max_out = next();
        else if (a == "--model") model_dir = next();
        else if (a == "--model-size") model_dir = std::string("models_rt/nanovsr_") + next();
        else if (a == "--bench") bench_only = true;
        else {
            std::printf("未知参数 %s\n", a.c_str());
            return 1;
        }
    }
    if (in_path.empty()) {
        std::printf("用法: nanovsr --in <video> --out <video> [--chunk N] [--crf 18] [--limit N]\n");
        return 1;
    }

    // ---- 探测分辨率与帧率 ----
    std::string probe = "\"" + ffprobe + "\" -v error -select_streams v:0 -show_entries "
                        "stream=width,height,r_frame_rate -of csv=p=0 \"" + in_path + "\"";
    std::string info = run_capture(probe);
    int W = 0, H = 0;
    double fps = 30.0;
    {
        int a = 0, b = 0, num = 0, den = 1;
        if (std::sscanf(info.c_str(), "%d,%d,%d/%d", &a, &b, &num, &den) >= 2) {
            W = a;
            H = b;
            if (den > 0) fps = (double)num / (double)den;
        }
    }
    if (W <= 0 || H <= 0) {
        std::printf("无法读取视频信息: %s\n", info.c_str());
        return 1;
    }
    std::printf("nanovsr build %s %s\n", __DATE__, __TIME__);
    std::printf("输入: %s  %dx%d  %.3f fps\n", in_path.c_str(), W, H, fps);

    // 4x 后超过 4K（H.264 上限）就先按整数倍降采样：1080p -> /2 -> 960x540 -> 4x -> 3840x2160
    int pre = 1;
    bool gpu_downscale = false;
    // 输出总像素上限：8K = 7680x4320 = 33.18 Mp。超了就按比例压（不是硬性 /2），
    // 保持宽高比、取偶数边长，4x 之后正好落在上限内。
    // 超采样：模型永远按 4x 跑满（保锐度），超出交付上限时在**编码端**缩，
    // 而不是先把输入缩了（输入缩会永久丢细节）。
    const int src_w = W, src_h = H;
    int out_w = W * 4, out_h = H * 4;
    {
        const double MAX_OUT_PX = (max_out == "8k") ? 7680.0 * 4320.0 : 4096.0 * 2304.0;
        const double out_px = (double)out_w * out_h;
        if (out_px > MAX_OUT_PX) {
            const double k = std::sqrt(MAX_OUT_PX / out_px);
            out_w = std::max(2, ((int)std::floor(out_w * k)) & ~1);
            out_h = std::max(2, ((int)std::floor(out_h * k)) & ~1);
            if (supersample) {
                pre = 2;                 // 模型跑满 4x，编码端再缩（保输入细节，但模型细节被稀释）
            } else {
                // 默认：先把输入缩到"4x 正好等于交付尺寸"，让模型工作在交付分辨率上，
                // 这样它的去噪/锐化效果不会被后续降采样冲掉
                W = std::max(2, out_w / 4);
                H = std::max(2, out_h / 4);
                pre = 2;
            }
        }
    }
    std::string dec_scale;
    std::string enc_scale;
    if (pre > 1) {
        // 输出端缩放：优先 libplacebo（GPU，不占 CPU），退回 swscale lanczos
        const std::string fl2 = run_capture("\"" + ffmpeg + "\" -hide_banner -filters");
        if (downscale != "lanczos" && fl2.find("libplacebo") != std::string::npos) {
            enc_scale = " -vf libplacebo=w=" + std::to_string(out_w) +
                        ":h=" + std::to_string(out_h) + ":downscaler=" + downscale;
            gpu_downscale = true;
            std::printf("输出缩放: libplacebo downscaler=%s (GPU) %dx%d -> %dx%d\n", downscale.c_str(), W * 4,
                        H * 4, out_w, out_h);
        } else {
            enc_scale = " -vf scale=" + std::to_string(out_w) + ":" + std::to_string(out_h) +
                        ":flags=lanczos";
            std::printf("输出缩放: swscale lanczos %dx%d -> %dx%d\n", W * 4, H * 4, out_w, out_h);
    }
    if (pre > 1 && !supersample) {
        // 默认路径：模型工作在交付分辨率（W×H），解码端必须把输入缩到 W×H，
        // 否则帧切片错位（解码输出的是原始尺寸）。输出端不需要再缩。
        dec_scale = " -vf scale=" + std::to_string(W) + ":" + std::to_string(H) + ":flags=lanczos";
        enc_scale.clear();
        std::printf("预处理: 解码缩到 %dx%d -> 模型 4x -> 交付 %dx%d\n", W, H, out_w, out_h);
        }
    }
    if (false) {
        // GPU 降采样：libplacebo + 窗口 sinc（抗锯齿更好，且走 Vulkan 不占 CPU）；
        // 没有这个 filter 就退回 swscale 的 lanczos
        const std::string flist = run_capture("\"" + ffmpeg + "\" -hide_banner -filters");
        if (downscale == "sinc" && flist.find("libplacebo") != std::string::npos) {
            dec_scale = " -vf libplacebo=w=" + std::to_string(W) + ":h=" + std::to_string(H) +
                        ":downscaler=sinc";
            gpu_downscale = true;
            std::printf("降采样: libplacebo downscaler=sinc (GPU)，预留 512 MB 显存\n");
        } else {
            dec_scale = " -vf scale=" + std::to_string(W) + ":" + std::to_string(H) +
                        ":flags=lanczos";
            std::printf("降采样: swscale lanczos (CPU，未找到 libplacebo)\n");
        }
        std::printf("预缩放: 处理分辨率 %dx%d，输出 %dx%d (%.1f Mp)\\n", W, H, W * 4, H * 4, (double)(W*4)*(H*4)/1e6);
    } else {
        std::printf("输出分辨率: %dx%d\n", W * 4, H * 4);
    }

    // ---- 模型清单 ----
    // 权重与 shader 的启动检查：下载发布包的人通常没有权重，要给出可操作的提示
    {
        std::ifstream test(model_dir + "/weights_rt.json");
        if (!test.good()) {
            std::printf("\n找不到模型权重: %s\\weights_rt.{bin,json}\n\n", model_dir.c_str());
            std::printf("本程序不附带权重（版权属于 NanoVSR 作者）。三步准备：\n");
            std::printf("  1) 下载官方 checkpoint（nanovsr_1.7m.pth 等）\n");
            std::printf("  2) python pack_model.py --tag 1.7m --out models_rt\\\\nanovsr_1.7m --weights-only\n");
            std::printf("  3) 确认 exe 旁边有 models_rt\\\\nanovsr_1.7m\\\\ 下三个文件\n");
            std::printf("详见仓库里的 models/README.md\n");
            return 2;
        }
        std::ifstream tsh("shaders/conv3x3_cm_64.spv");
        if (!tsh.good()) {
            std::printf("\n找不到 shader: shaders/*.spv（需要先运行 build.bat 编译）\n");
            return 2;
        }
    }
    auto jraw = read_file(model_dir + "/weights_rt.json");
    std::string js(jraw.begin(), jraw.end());
    auto jint = [&](const std::string& key, size_t from) {
        size_t p = js.find("\"" + key + "\"", from);
        p = js.find(':', p);
        return std::atoi(js.c_str() + p + 1);
    };
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
    const int nf_ = jint("num_feat", 0);           // 模型宽度 C
    int nb_ = jint("num_blocks", 0);                // 每个方向多少层
    if (nf_ <= 0) {
        std::printf("清单里没有 num_feat\n");
        return 1;
    }
    if (nb_ <= 0) nb_ = 20;
    const size_t F = (size_t)nf_;
    const int nb = nb_;
    std::printf("模型: num_feat=%d, blocks=%d\n", (int)F, nb);

    const float act = 0.1f;

    // ---- 设备与显存 ----
    Gpu g;
    g.init();
    auto wblob = read_file(model_dir + "/weights_rt.bin");
    Buf bw = g.alloc(wblob.size());
    std::memcpy(bw.mapped, wblob.data(), wblob.size());
    auto cmblob = read_file(model_dir + "/weights_cm.bin");
    Buf bwcm = g.alloc(cmblob.size());
    std::memcpy(bwcm.mapped, cmblob.data(), cmblob.size());

    // ---- preset：三档，显式参数仍然覆盖 ----
    int preset_chunk = 30, preset_crf = 18;
    double preset_budget = 0.0;           // 0 = 驱动建议 x0.8
    if (preset == "preview") {
        preset_chunk = 8;                 // 时间窗短 → 省下的显存给更大的 tile（重叠浪费更小）
        preset_budget = 0.0;              // 预算不变，让搜索挑更大的 tile
        preset_crf = 22;                  // 编码快、体积小
    } else if (preset == "quality") {
        preset_chunk = 30;
        preset_budget = 0.0;
        preset_crf = 14;                  // 编码更保真
    } else if (preset != "balanced") {
        std::printf("未知 preset '%s'，用 balanced（可选 preview / balanced / quality）\n",
                    preset.c_str());
    }
    if (chunk_arg == 0) chunk_arg = preset_chunk;
    if (budget_arg_gb == 0.0) budget_arg_gb = preset_budget;
    if (crf < 0) crf = preset_crf;
    std::printf("preset: %s  (chunk %d, budget %s, crf %d)\n", preset.c_str(), chunk_arg,
                budget_arg_gb > 0 ? "见下" : "驱动建议 x0.8", crf);

    // ---- 空间分块 + chunk（显存预算）----
    // 时间窗优先：递推每 chunk 重置一次，窗太小会周期性闪烁。默认 30 = 训练 Phase 2 的长序列长度。
    int chunk = chunk_arg > 0 ? chunk_arg : 30;
    chunk = std::min(chunk, 30);
    // ---- 显存账本：先算账，再按账分配 ----
    // 每像素的固定开销（不含时间维）
    const double Cw = (double)nf_;
    const double S_PX = Cw * 4 * 3        // bp0/bp1/bcatus
                        + Cw * 4 * 2      // bcat（2C 通道）
                        + Cw * 4          // bfused
                        + Cw * 4 * 2      // bconv1（4C 通道 f16，1x 分辨率）
                        + Cw * 4 * 4      // bup1（C 通道，2x 分辨率 = 4 倍像素）
                        + 4096            // bconv2（128 通道 f16 x 16 倍像素）
                        + 2048            // bup2（32 通道 x 16 倍像素）
                        + 192;            // blast（3 通道 x 16 倍像素）
    // 每像素每帧（时间维）
    const double S_PX_F = 12      // bx（3 通道 f32）
                          + 4     // bu8
                          + Cw * 4 * 3   // feat/fwd/bwd
                          + 64    // bout
                          + 4     // bstage_in
                          + 64;   // bstage_out
    const double drv = (double)g.device_budget();
    // GPU 降采样（libplacebo）会和我们共用显存堆，额外预留 512 MB
    const double budget = budget_arg_gb > 0
                              ? budget_arg_gb * 1e9
                              : std::max(2e9, drv * 0.80 - (gpu_downscale ? 512e6 : 0.0));
    const double w_bytes = (double)(wblob.size() + cmblob.size());
    const size_t max_px = (size_t)std::max(65536.0, (budget - w_bytes) / (S_PX + S_PX_F * chunk));

    struct TileRect {
        int x, y, w, h;
    };
    std::vector<TileRect> tiles;
    const int ov = 96;                     // 重叠 96：每个方向丢 48，覆盖 40 层卷积的接收野
    std::vector<int> xs, ys;
    // tile 尺寸搜索：在预算内选"总处理像素最少"的 (tw,th)。
    // 因为 tile 与帧尺寸的整除关系影响很大（960 宽用 366 会切 4 列），开方取整并不最优。
    auto grid_for = [&](int tw, int th, std::vector<int>& gx, std::vector<int>& gy,
                        size_t& total) {
        gx.clear();
        gy.clear();
        if (tw >= W && th >= H) {
            gx = {0};
            gy = {0};
            total = (size_t)W * H;
            return true;
        }
        const int sx = std::max(1, tw - ov), sy = std::max(1, th - ov);
        for (int x = 0; x < W; x += sx) gx.push_back(std::min(x, W - tw));
        for (int y = 0; y < H; y += sy) gy.push_back(std::min(y, H - th));
        std::sort(gx.begin(), gx.end());
        gx.erase(std::unique(gx.begin(), gx.end()), gx.end());
        std::sort(gy.begin(), gy.end());
        gy.erase(std::unique(gy.begin(), gy.end()), gy.end());
        total = 0;
        for (int yy : gy)
            for (int xx : gx)
                total += (size_t)std::min(tw, W - xx) * std::min(th, H - yy);
        return true;
    };

    int tw0 = std::min(W, 256), th0 = std::min(H, 256);
    size_t best_total = (size_t)-1;
    {
        const int lo = 160;
        const int hi_w = std::min(W, 768), hi_h = std::min(H, 768);
        std::vector<int> gx, gy;
        for (int tw = lo; tw <= hi_w; tw += 16)
            for (int th = lo; th <= hi_h; th += 16) {
                const double foot = (double)tw * th * (S_PX + S_PX_F * chunk) + w_bytes;
                if (foot > budget) continue;          // 超出预算
                size_t total = 0;
                grid_for(tw, th, gx, gy, total);
                if (total < best_total) {
                    best_total = total;
                    tw0 = tw;
                    th0 = th;
                }
            }
        if (best_total == (size_t)-1) {                // 预算太小，退回最小 tile
            tw0 = std::min(W, lo);
            th0 = std::min(H, lo);
        }
        if (tile_arg_w > 0 && tile_arg_h > 0) {        // 手动指定优先
            tw0 = tile_arg_w;
            th0 = tile_arg_h;
        }
    }
    (void)max_px;
    size_t work_total = 0;
    {
        std::vector<int> gx, gy;
        grid_for(tw0, th0, gx, gy, work_total);
        xs = gx;
        ys = gy;
    }
    for (size_t yi = 0; yi < ys.size(); ++yi)
        for (size_t xi = 0; xi < xs.size(); ++xi) {
            TileRect tr;
            tr.x = xs[xi];
            tr.y = ys[yi];
            tr.w = std::min(tw0, W - tr.x);
            tr.h = std::min(th0, H - tr.y);
            tiles.push_back(tr);
        }
    size_t px = 0;
    for (auto& tr : tiles) px = std::max(px, (size_t)tr.w * tr.h);
    {
        const double s_static = px * S_PX, s_temporal = px * S_PX_F * chunk;
        const double total = w_bytes + s_static + s_temporal;
        std::printf("显存账本：预算 %.2f GB（%s）\n", budget / 1e9,
                    budget_arg_gb > 0 ? "手动指定" : "驱动建议 x0.8");
        std::printf("  权重             %7.1f MB  (固定)\n", w_bytes / 1e6);
        std::printf("  静态 tile 缓冲   %7.1f MB  (%.1f KB/px x %.0f k px)\n",
                    s_static / 1e6, S_PX / 1024.0, px / 1000.0);
        std::printf("  时间维缓冲       %7.1f MB  (%.2f KB/px/帧 x %d 帧)\n",
                    s_temporal / 1e6, S_PX_F / 1024.0, chunk);
        std::printf("  ── 合计          %7.2f GB / %.2f GB  (余 %.0f MB)\n", total / 1e9,
                    budget / 1e9, (budget - total) / 1e6);
        std::printf("→ tile %dx%d（overlap %d），分块 %zu 块，单块最大 %.0f k 像素\n", tw0, th0,
                    ov, tiles.size(), px / 1000.0);
        std::printf("  总处理量 %.2f M 像素/帧（帧本身 %.2f M，放大 %.2fx）\n",
                    work_total / 1e6, (double)W * H / 1e6,
                    (double)work_total / std::max(1.0, (double)W * H));
    }



    
    const int T = chunk;
    Buf bx = g.alloc((VkDeviceSize)T * px * 3 * 4);
    Buf bu8 = g.alloc((VkDeviceSize)T * px * 4);              // 解码后的打包 RGB
    Buf bfeat = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bfwd = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bbwd = g.alloc((VkDeviceSize)T * px * F * 4);
    Buf bp0 = g.alloc((VkDeviceSize)px * F * 4);
    Buf bp1 = g.alloc((VkDeviceSize)px * F * 4);
    Buf bcatus = g.alloc((VkDeviceSize)px * F * 4);
    Buf bcat = g.alloc((VkDeviceSize)px * 2 * F * 4);
    Buf bfused = g.alloc((VkDeviceSize)px * F * 4);
    Buf bconv1 = g.alloc((VkDeviceSize)px * 4 * F * 2);   // f16
    Buf bup1 = g.alloc((VkDeviceSize)px * 4 * F * 4);
    Buf bconv2 = g.alloc((VkDeviceSize)px * 16 * 128 * 2);   // f16
    Buf bup2 = g.alloc((VkDeviceSize)px * 16 * 32 * 4);
    Buf blast = g.alloc((VkDeviceSize)px * 16 * 3 * 4);
    Buf bout = g.alloc((VkDeviceSize)T * px * 16 * 4);
    Buf bstage_in = g.alloc_host((VkDeviceSize)T * px * 4);
    Buf bstage_out = g.alloc_host((VkDeviceSize)T * px * 16 * 4);

    const VkPipeline pConv64 = g.pipe("shaders/conv3x3_c64.spv");
    const VkPipeline pConvGen = g.pipe("shaders/conv_gen.spv");
    const VkPipeline pConvSmall = g.pipe("shaders/conv3x3_smallc.spv");
        // 张量核 shader 按 num_feat 编译了变体（C 是编译期常量）
    std::string cm_path = "shaders/conv3x3_cm_" + std::to_string((int)F) + ".spv";
    std::string cm16_path = "shaders/conv3x3_cm_" + std::to_string((int)F) + "_f16.spv";
    if (!std::ifstream(cm_path).good() || !std::ifstream(cm16_path).good()) {
        std::printf("缺少张量核变体: %s / %s（build.bat 按 num_feat 生成，C=%d）\n",
                    cm_path.c_str(), cm16_path.c_str(), (int)F);
        return 1;   // 宁可启动失败，也不要静默套用 C=64 的错内核
    }
    std::printf("张量核 shader: %s\n", cm_path.c_str());
    const VkPipeline pConvCM = g.pipe(cm_path);
    const VkPipeline pConvCM16 = g.pipe(cm16_path);
    const VkPipeline pShuffle16 = g.pipe("shaders/shuffle_prelu_f16.spv");
    const VkPipeline pShuffle = g.pipe("shaders/shuffle_prelu.spv");
    const VkPipeline pFinal = g.pipe("shaders/bilinear4_add_u8.spv");
    const VkPipeline pAdd = g.pipe("shaders/add.spv");
    const VkPipeline pCat = g.pipe("shaders/cat2.spv");
    const VkPipeline pF32 = g.pipe("shaders/u8rgb_to_f32.spv");

    auto run_conv64 = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                          VkDeviceSize out_off, int h, int w, int O, float slope, bool out_f16 = false) {
        const WLayer& L = layers[idx[name]];
        if (name == "forward_0") std::printf("[diag] %s cm_off=%d C=%d O=%d -> %s\n", name.c_str(), L.cm_off, L.C, O, (L.cm_off >= 0) ? "CM" : "SIMT");
        if (L.cm_off >= 0) {
            VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                             {bwcm.buf, (VkDeviceSize)L.cm_off, VK_WHOLE_SIZE},
                                             {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                             {out.buf, out_off, VK_WHOLE_SIZE}};
            Push pc;
            pc.C_ = (int)F;
            pc.H = h;
            pc.W = w;
            pc.O = O;
            pc.k = (O + 63) / 64 * 64;      // 权重里输出通道补齐后的跨度（张量核按 64 一组）
            pc.slope = slope;
            g.dispatch(out_f16 ? pConvCM16 : pConvCM, dbi, pc, (uint32_t)((w + 31) / 32),
                       (uint32_t)((h + 3) / 4), (uint32_t)((O + 63) / 64));
            return;
        }
        VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.w_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                         {out.buf, out_off, VK_WHOLE_SIZE}};
        Push pc;
        pc.C_ = (int)F;
        pc.H = h;
        pc.W = w;
        pc.O = O;
        pc.slope = slope;
        if ((int)F == 64) {
            g.dispatch(pConv64, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
        } else {
            // 通用 shader 按 p.C_ 循环，C != 64 时唯一正确的兜底
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
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
            g.dispatch(pConvSmall, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8));
        } else {
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
    };

    // ---- 管道 ----
    std::string dec = "\"" + ffmpeg + "\" -v error -i \"" + in_path + "\"" + dec_scale +
                      " -f rawvideo -pix_fmt rgb24 -";
    FILE* fin = popen_r(dec);
    if (!fin) {
        std::printf("无法启动解码\n");
        return 1;
    }
    FILE* fout = nullptr;
    if (!bench_only) {
        // 编码器自动选择：AV1(NVENC) > AV1(SVT) > HEVC > H.264
        // 候选按质量/速度排序，但必须"真跑一帧"验证（nvenc 会被列出却因驱动版本打开失败）
        auto probe_enc = [&](const std::string& v) {
            return run_ok("\"" + ffmpeg + "\" -v error -f lavfi -i color=c=black:s=64x64:d=0.1 "
                          "-frames:v 1 " + v + " -f null -");
        };
        struct Cand {
            const char* name;
            std::string venc;
        };
        // 各编码器的 CRF 标尺不同：AV1 的 23 ≈ 视觉无损（x264 的 23 已经明显有损）。
        // 以 x264 为基准，AV1 +5、HEVC +2、NVENC 同 AV1。
        auto qof = [&](const std::string& name) {
            int off = 0;
            if (name.find("av1") != std::string::npos) off = 5;
            else if (name.find("hevc") != std::string::npos || name.find("265") != std::string::npos) off = 2;
            return std::to_string(crf + off);
        };
        std::vector<Cand> cands = {
            {"libsvtav1", "-c:v libsvtav1 -crf " + qof("av1") + " -preset 4 -svtav1-params lp=4"},
            {"libx265", "-c:v libx265 -crf " + qof("x265") + " -preset medium"},
            {"libx264", "-c:v libx264 -crf " + qof("x264") + " -preset veryfast"},
        };
        std::string venc;
        for (auto& c : cands) {
            if (codec_pref != "auto" && codec_pref != c.name) continue;
            if (probe_enc(c.venc)) {
                venc = c.venc;
                break;
            }
            std::printf("  编码器 %s 不可用，跳过\n", c.name);
        }
        if (venc.empty()) venc = "-c:v libx264 -crf " + qof("x264") + " -preset veryfast";
        std::printf("编码器: %s\n", venc.c_str());
        char cmd[2048];
        std::snprintf(cmd, sizeof(cmd),
                      "\"%s\" -v error -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r %.6f -i - "
                      "%s%s -pix_fmt yuv420p \"%s\"",
                      ffmpeg.c_str(), W * 4, H * 4, fps, venc.c_str(), enc_scale.c_str(),
                      out_path.c_str());
        fout = popen_w(cmd);
        if (!fout) {
            std::printf("无法启动编码\n");
            return 1;
        }
    }

    // ---- 主循环：按 (chunk 帧) x (空间 tile) 处理 ----
    const int FW = W, FH = H;                    // 帧尺寸（W/H 在 tile 循环里会被改成 tile 尺寸）
    const size_t FOFF = 4;                       // 放大倍数
    std::vector<std::vector<unsigned char>> frames;
    const size_t OFRAME_BYTES = (size_t)FW * FOFF * FH * FOFF * 3;
    std::vector<unsigned char> out_frames;   // 整个 chunk 的输出（必须按帧分开，否则会互相覆盖）
    long long total = 0;
    double t_start = (double)std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    while (true) {
        int t = 0;
        frames.assign(chunk, {});
        for (; t < chunk; ++t) {
            if (limit > 0 && total + t >= limit) break;
            auto& fr = frames[t];
            fr.resize((size_t)FW * FH * 3);
            if (std::fread(fr.data(), 1, fr.size(), fin) != fr.size()) break;
        }
        if (t == 0) break;
        frames.resize(t);
        out_frames.assign((size_t)t * OFRAME_BYTES, 0);
        total += t;                              // 注意：这里先计，避免末尾重复

        for (size_t ti = 0; ti < tiles.size(); ++ti) {
            const TileRect& tile = tiles[ti];
            W = tile.w;
            H = tile.h;
            // 1) 把本 tile 的像素打包进 staging（系统内存）
            for (int i = 0; i < t; ++i) {
                uint32_t* dst = (uint32_t*)bstage_in.mapped + (size_t)i * px;
                const unsigned char* src = frames[i].data();
                for (int y = 0; y < tile.h; ++y) {
                    const unsigned char* row = src + ((size_t)(tile.y + y) * FW + tile.x) * 3;
                    uint32_t* drow = dst + (size_t)y * tile.w;
                    for (int x = 0; x < tile.w; ++x)
                        drow[x] = (uint32_t)row[x * 3] | ((uint32_t)row[x * 3 + 1] << 8) |
                                  ((uint32_t)row[x * 3 + 2] << 16);
                }
            }

        const double t0 = (double)std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        g.reset_pool();
        g.begin();
        // staging -> 显存（一次拷贝整块）
        g.copy(bstage_in, 0, bu8, 0, (VkDeviceSize)t * px * 4);
        // 转 f32
        for (int i = 0; i < t; ++i) {
            VkDescriptorBufferInfo dbi[4] = {{bu8.buf, (VkDeviceSize)i * px * 4, VK_WHOLE_SIZE},
                                             {bw.buf, 0, 1},
                                             {bw.buf, 0, 1},
                                             {bx.buf, (VkDeviceSize)i * px * 3 * 4, VK_WHOLE_SIZE}};
            Push pc;
            pc.H = (int32_t)px;
            g.dispatch(pF32, dbi, pc, (uint32_t)((px + 63) / 64));
        }
        // feat
        for (int i = 0; i < t; ++i)
            run_convg("feat_extract", bx, (VkDeviceSize)i * px * 3 * 4, bfeat,
                      (VkDeviceSize)i * px * F * 4, H, W, act);
        // 双向链
        Buf* scratch[2] = {&bp0, &bp1};
        auto run_dir = [&](bool forward, Buf& dst_all) {
            Buf* prop = scratch[0];
            g.fill(*prop, 0, (VkDeviceSize)px * F * 4, 0);
            for (int k = 0; k < t; ++k) {
                const int tt = forward ? k : (t - 1 - k);
                VkDescriptorBufferInfo dbi[4] = {
                    {bfeat.buf, (VkDeviceSize)tt * px * F * 4, VK_WHOLE_SIZE},
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
                    run_conv64(name, *src, 0, *scratch[w], 0, H, W, (int)F, act);
                    src = scratch[w];
                    w ^= 1;
                }
                g.copy(*src, 0, dst_all, (VkDeviceSize)tt * px * F * 4,
                       (VkDeviceSize)px * F * 4);
                prop = src;
            }
        };
        const double t1 = (double)std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        run_dir(true, bfwd);
        run_dir(false, bbwd);
        g.submit_wait();
        const double t2 = (double)std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // 每帧后半段
        g.reset_pool();
        g.begin();
        for (int i = 0; i < t; ++i) {
            VkDescriptorBufferInfo dbi[4] = {
                {bfwd.buf, (VkDeviceSize)i * px * F * 4, VK_WHOLE_SIZE},
                {bbwd.buf, (VkDeviceSize)i * px * F * 4, VK_WHOLE_SIZE},
                {bw.buf, 0, 1},
                {bcat.buf, 0, VK_WHOLE_SIZE}};
            Push pc;
            pc.C_ = (int32_t)F;          // 前向特征通道数（不是固定的 64）
            pc.H = (int32_t)F;           // 反向特征通道数
            pc.W = (int32_t)px;
            g.dispatch(pCat, dbi, pc, (uint32_t)((px + 63) / 64));

            run_convg("fusion", bcat, 0, bfused, 0, H, W, 0.0f);
            run_conv64("up1", bfused, 0, bconv1, 0, H, W, 4 * (int)F, 0.0f, true);
            {
                VkDescriptorBufferInfo d2[4] = {{bconv1.buf, 0, VK_WHOLE_SIZE},
                                                {bw.buf, 0, 1},
                                                {bw.buf, 0, 1},
                                                {bup1.buf, 0, VK_WHOLE_SIZE}};
                Push p2;
                p2.C_ = (int)F;
                p2.H = H;
                p2.W = W;
                p2.slope = prelu1;
                g.dispatch(pShuffle16, d2, p2, (uint32_t)((W + 15) / 16), (uint32_t)H);
            }
            run_conv64("up2", bup1, 0, bconv2, 0, H * 2, W * 2, 128, 0.0f, true);
            {
                VkDescriptorBufferInfo d2[4] = {{bconv2.buf, 0, VK_WHOLE_SIZE},
                                                {bw.buf, 0, 1},
                                                {bw.buf, 0, 1},
                                                {bup2.buf, 0, VK_WHOLE_SIZE}};
                Push p2;
                p2.C_ = 32;
                p2.H = H * 2;
                p2.W = W * 2;
                p2.slope = prelu2;
                g.dispatch(pShuffle16, d2, p2, (uint32_t)((W * 2 + 15) / 16), (uint32_t)(H * 2));
            }
            run_convg("conv_last", bup2, 0, blast, 0, H * 4, W * 4, 0.0f);
            {
                VkDescriptorBufferInfo d2[4] = {
                    {bx.buf, (VkDeviceSize)i * px * 3 * 4, VK_WHOLE_SIZE},
                    {blast.buf, 0, VK_WHOLE_SIZE},
                    {bw.buf, 0, 1},
                    {bout.buf, (VkDeviceSize)i * px * 16 * 4, VK_WHOLE_SIZE}};
                Push p2;
                p2.H = H;
                p2.W = W;
                p2.O = 4;
                g.dispatch(pFinal, d2, p2, (uint32_t)((px * 16 + 63) / 64));
            }
        }
        g.submit_wait();
        const double t3 = (double)std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // 输出：先 GPU 拷到 staging，再让 CPU 读系统内存
        g.begin();
        g.copy(bout, 0, bstage_out, 0, (VkDeviceSize)t * px * 16 * 4);
        g.submit_wait();

        // 2) 只把"中心区"合成到输出帧（边缘 overlap 段丢弃）
        const int cx0 = (tile.x == 0) ? 0 : tile.x + ov / 2;
        const int cx1 = (tile.x + tile.w >= FW) ? FW : tile.x + tile.w - ov / 2;
        const int cy0 = (tile.y == 0) ? 0 : tile.y + ov / 2;
        const int cy1 = (tile.y + tile.h >= FH) ? FH : tile.y + tile.h - ov / 2;
        const int tw4 = tile.w * 4;
        const size_t ow4 = (size_t)FW * 4;
        for (int i = 0; i < t; ++i) {
            const uint32_t* src = (const uint32_t*)bstage_out.mapped + (size_t)i * px * 16;
            unsigned char* dframe = out_frames.data() + (size_t)i * OFRAME_BYTES;
            for (int oy = (cy0 - tile.y) * 4; oy < (cy1 - tile.y) * 4; ++oy) {
                const uint32_t* srow = src + (size_t)oy * tw4;
                unsigned char* dstrow = dframe +
                                        ((size_t)(tile.y * 4 + oy) * ow4 + (size_t)tile.x * 4) * 3;
                for (int ox = (cx0 - tile.x) * 4; ox < (cx1 - tile.x) * 4; ++ox) {
                    const uint32_t v = srow[ox];
                    dstrow[ox * 3 + 0] = (unsigned char)(v & 255);
                    dstrow[ox * 3 + 1] = (unsigned char)((v >> 8) & 255);
                    dstrow[ox * 3 + 2] = (unsigned char)((v >> 16) & 255);
                }
            }
        }
        }  // tile 循环结束

        // 3) 整帧写进编码器
        if (fout) {
            for (int i = 0; i < t; ++i) {
                const unsigned char* fp = out_frames.data() + (size_t)i * OFRAME_BYTES;
                if (std::fwrite(fp, 1, OFRAME_BYTES, fout) != OFRAME_BYTES) {
                    std::printf("写出失败\n");
                    break;
                }
            }
        }
        double now = (double)std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::printf("\r处理 %lld 帧  %.2f fps  [tile %d/%d]",
                    total, total / std::max(1e-6, now - t_start), (int)tiles.size(),
                    (int)tiles.size());
        if (t < chunk) break;
    }
    std::printf("\n完成: %lld 帧\n", total);

    if (fin) _pclose(fin);
    if (fout) pclose_w(fout);
    vkDeviceWaitIdle(g.dev);
    return 0;
}






