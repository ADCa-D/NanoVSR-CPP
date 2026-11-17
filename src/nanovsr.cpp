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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
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

static double now_s() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static std::vector<char> read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("open failed: %s\n", path.c_str());
        std::exit(1);
    }
    // 64 位 seek：MSVC 的 long 只有 32 位，权重文件超过 2GB 时 ftell 返回 -1，
    // 转成 size_t 就是天文数字，vector 立刻抛 bad_alloc。
    _fseeki64(f, 0, SEEK_END);
    const long long n64 = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    if (n64 < 0) {
        std::printf("seek failed: %s\n", path.c_str());
        std::exit(1);
    }
    const size_t n = (size_t)n64;
    std::vector<char> b(n);
    if (n && std::fread(b.data(), 1, n, f) != n) std::exit(1);
    std::fclose(f);
    return b;
}

// 常驻线程池：把 [0, nrows) 的行区间按线程切分执行，主线程做第 0 片。
// 打包/合成每 tile 都要跑一次，原来每次 emplace_back + join 一批 std::thread，
// 24 线程 x 2 次 x 24 tile x 每 chunk，光创建销毁就是毫秒级。
class RowPool {
public:
    explicit RowPool(int n) : n_(std::max(1, n)) {
        for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { worker(i); });
    }
    ~RowPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void run(int nrows, const std::function<void(int, int)>& fn) {
        if (nrows <= 0) return;
        const int jobs = std::min(n_, nrows);
        if (jobs <= 1) {
            fn(0, nrows);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            nrows_ = nrows;
            jobs_ = jobs;
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        fn(0, (int)((long long)nrows / jobs));           // 线程 0 的分片由主线程做
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return done_ >= n_ - 1; });
        fn_ = nullptr;
    }

private:
    void worker(int idx) {
        int seen = 0;
        for (;;) {
            const std::function<void(int, int)>* f = nullptr;
            int jobs = 1, nrows = 0;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                f = fn_;
                jobs = jobs_;
                nrows = nrows_;
            }
            if (f && idx < jobs)
                (*f)((int)((long long)nrows * idx / jobs),
                     (int)((long long)nrows * (idx + 1) / jobs));
            bool last = false;
            {
                std::lock_guard<std::mutex> lk(m_);
                last = (++done_ == n_ - 1);   // 只让最后一个完成的线程唤醒主线程，别广播风暴
            }
            if (last) cv_done_.notify_one();
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, cv_done_;
    const std::function<void(int, int)>* fn_ = nullptr;
    int nrows_ = 0, jobs_ = 1, done_ = 0, gen_ = 0;
    bool stop_ = false;
};

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
    static const int NSLOT = 2;              // 双缓冲流水槽数
    VkCommandBuffer cbs[NSLOT]{};
    VkFence fences[NSLOT]{};
    bool slot_pending[NSLOT] = {false, false};
    int cur = 0;                             // 当前录制用的槽
    VkCommandBuffer cb{};                    // == cbs[cur]，保持原有调用点不变
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout pl{};
    VkDescriptorPool dpool{};                // 全局描述符缓存池（见 set_for）
    std::map<std::string, VkPipeline> pipes;
    bool mem_budget_available = false;
    int gpu_index = -1;   // >=0 时强制用该序号的物理设备（调试/核显回退用）
    // 启动时探测的设备能力：决定走张量核（cooperative matrix）还是通用 SIMT 通路
    bool cm_ok = false;          // 扩展 + 特性 + 16x16x16 f16 组合齐全
    std::string cm_reason;       // 不能走张量核的原因（打印给用户）
    uint32_t max_shared = 0;     // maxComputeSharedMemorySize
    uint32_t subgroup_size = 0;  // 当前设备的 subgroup 宽度（张量核 shader 按 32 写死）

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        // 1.1 就够（coopmat 是扩展，不需要 1.2）；要求 1.2 会把 1.1-only 的老设备直接踢掉
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst, &n, devs.data());
        pd = devs[0];
        if (gpu_index >= 0 && (uint32_t)gpu_index < n) {
            pd = devs[gpu_index];   // 显式指定（--gpu N）
        } else {
            if (gpu_index >= 0)
                std::printf("注意: --gpu %d 越界（本机共 %u 个 Vulkan 设备），改用默认选择\n",
                            gpu_index, n);
            for (auto d : devs) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(d, &p);
                if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    pd = d;
                    break;
                }
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
        bool has_cm_ext = false;
        for (auto& e : exts_avail) {
            if (std::strcmp(e.extensionName, "VK_EXT_memory_budget") == 0) has_budget = true;
            if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) has_cm_ext = true;
        }
        mem_budget_available = has_budget;

        // ---- 能力探测：张量核能否使用取决于三件事，跟显存多大无关 ----
        //   1) 有 VK_KHR_cooperative_matrix 且驱动打开了 cooperativeMatrix 特性
        //   2) 支持 16x16x16 的 f16 组合
        //   3) subgroup 宽度 = 32（shader 按 32 写死了 warp 划分）且 shared 装得下
        {
            VkPhysicalDeviceProperties pp{};
            vkGetPhysicalDeviceProperties(pd, &pp);
            max_shared = pp.limits.maxComputeSharedMemorySize;
            VkPhysicalDeviceSubgroupProperties sgp{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &sgp;
            vkGetPhysicalDeviceProperties2(pd, &p2);
            subgroup_size = sgp.subgroupSize;
        }
        bool cm_props_ok = false;
        if (has_cm_ext && cmf.cooperativeMatrix) {
            auto fncm = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
            if (fncm) {
                uint32_t cnt = 0;
                if (fncm(pd, &cnt, nullptr) == VK_SUCCESS && cnt > 0) {
                    std::vector<VkCooperativeMatrixPropertiesKHR> pr(cnt);
                    for (auto& c : pr)
                        c = VkCooperativeMatrixPropertiesKHR{
                            VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
                    if (fncm(pd, &cnt, pr.data()) == VK_SUCCESS) {
                        for (auto& c : pr)
                            if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                                c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                (c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR ||
                                 c.CType == VK_COMPONENT_TYPE_FLOAT16_KHR))
                                cm_props_ok = true;
                    }
                }
            }
        }
        cm_ok = has_cm_ext && cmf.cooperativeMatrix && cm_props_ok &&
                storage16.storageBuffer16BitAccess && f16.shaderFloat16;
        if (!has_cm_ext) cm_reason = "设备没有 VK_KHR_cooperative_matrix 扩展";
        else if (!cmf.cooperativeMatrix) cm_reason = "驱动没有打开 cooperativeMatrix 特性";
        else if (!cm_props_ok) cm_reason = "不支持 16x16x16 的 f16 cooperative matrix 组合";
        else if (!(storage16.storageBuffer16BitAccess && f16.shaderFloat16))
            cm_reason = "缺少 16bit storage / shaderFloat16 特性";

        std::vector<const char*> exts;
        if (has_cm_ext) exts.push_back("VK_KHR_cooperative_matrix");
        if (has_budget) exts.push_back("VK_EXT_memory_budget");
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
        cbai.commandBufferCount = NSLOT;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, cbs));
        cb = cbs[0];
        for (int i = 0; i < NSLOT; ++i) {
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fences[i]));
        }

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
        // 每个 set 用 4 个 storage buffer，池里的 descriptorCount 必须 >= maxSets * 4，
        // 否则分配 8192 个 set 之后就开始返回 VK_ERROR_OUT_OF_POOL_MEMORY
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384 * 4};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 16384;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        // 描述符 set 全部走缓存（set_for），所以只要一个永久池，永不 reset
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
        // pass 0: 系统内存 + cached（CPU 读写最快）
        // pass 1: 系统内存（write-combined，读起来慢但能用）
        // pass 2: 核显 UMA —— 所有堆都带 DEVICE_LOCAL，只能退一步要 cached
        // 四趟严格按「cached 优先、系统内存优先」排序：核显 UMA 上 local 也会出现，
        // 但 local 且非 cached（write-combined）读起来最慢，必须排在最后。
        for (int pass = 0; pass < 4 && pick == UINT32_MAX; ++pass) {
            const bool want_local = (pass == 1 || pass == 3);
            const bool want_cached = (pass < 2);
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
                const VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
                if (!(req.memoryTypeBits & (1u << i))) continue;
                if (!(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
                if (!(fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) continue;
                const bool cached = (fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
                const bool local = (fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                if (local != want_local || cached != want_cached) continue;
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

    // 纯 GPU 缓冲：DEVICE_LOCAL 且不映射。中间特征在 4K/chunk30 下是几百 MB 到 1 GB，
    // 走 HOST_VISIBLE 会落到系统内存（非 ReBAR 机器超过 256MB BAR 直接分配失败），
    // 每个 dispatch 都要跨 PCIe 读，慢到不可用。只有 staging 才需要 CPU 映射。
    Buf alloc_device(VkDeviceSize bytes) {
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
        ai.memoryTypeIndex = mem_type(req.memoryTypeBits, 0);   // want=0 -> mem_type 只要求 DEVICE_LOCAL
        VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
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

    void begin(int slot = -1) {   // 默认当前槽（提交中途再 begin 时不要切到别的槽）
        if (slot < 0) slot = cur;
        cur = slot;
        cb = cbs[slot];
        wait(slot);        // 只等"本槽"上一次提交（它自己的 staging / 命令缓冲要复用）
        VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        since_submit = 0;
        // 另一槽可能还在跑，而两个槽共用中间缓冲（bu8/bfeat/bfwd/bbwd/bp0/bp1...）。
        // 同一队列的提交顺序已保证执行先后，缺的只是读写可见性，所以在 cmd 开头放一次
        // barrier：它是首条命令，等于给本块与所有前序提交建立执行+内存依赖。
        // 这样 CPU 不必死等另一槽的 fence —— 那会把双缓冲打回串行。
        barrier();
    }
    // 不等 GPU：提交后立刻返回，fence[slot] 用来判断这套资源何时可复用
    void submit(int slot) {
        VK_CHECK(vkEndCommandBuffer(cbs[slot]));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cbs[slot];
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fences[slot]));
        slot_pending[slot] = true;
    }
    double wait_s = 0.0;                     // 累计阻塞在 fence 上的时间（= GPU 忙但 CPU 无事可做）
    // 每个提交里最多容纳多少次 dispatch。0 = 不限（整块 tile 一个 cmd buffer）。
    // 通用 SIMT 通路单次 dispatch 要几百毫秒，整块塞进一个提交会让 GPU 长时间不返回，
    // 驱动判定卡死（实测 DEVICE_LOST / 0xC0000409），所以那条通路要分多次提交。
    // 单次提交的 dispatch 上限，只作 Windows TDR 保险：一个 4K tile 约 2700 次 dispatch、
    // 墙钟 ~0.4 s，远低于 2 s TDR 门槛。设太小会在 tile 中途 submit+wait 强制排空队列，
    // 把双槽流水打回同步（评审指出）。SIMT 通路慢 20 倍，由主程序另行降到 8。
    int max_per_submit = 8192;
    int since_submit = 0;
    void wait(int slot) {
        if (!slot_pending[slot]) return;
        const double t0 = now_s();
        VK_CHECK(vkWaitForFences(dev, 1, &fences[slot], VK_TRUE, UINT64_MAX));
        wait_s += now_s() - t0;
        VK_CHECK(vkResetFences(dev, 1, &fences[slot]));
        slot_pending[slot] = false;
    }
    // 队列顺序保证：后提交的 cmd 一定在前一个之后执行，所以 GPU 缓冲可以只留一份
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
    // 描述符 set 缓存：缓冲整个运行期只分配一次，偏移也只由 px / 帧号决定，所以同一个
    // (buffer, offset, range) 组合会重复几千次。原来每层每次 dispatch 都要
    // vkAllocateDescriptorSets + vkUpdateDescriptorSets（热循环里是纯 CPU 开销），这里缓存。
    struct DsKey {
        VkBuffer b[4];
        VkDeviceSize o[4];
        VkDeviceSize r[4];
        bool operator<(const DsKey& k) const {
            for (int i = 0; i < 4; ++i) {
                if (b[i] != k.b[i]) return b[i] < k.b[i];
                if (o[i] != k.o[i]) return o[i] < k.o[i];
                if (r[i] != k.r[i]) return r[i] < k.r[i];
            }
            return false;
        }
    };
    std::map<DsKey, VkDescriptorSet> dscache;
    VkDescriptorSet set_for(const VkDescriptorBufferInfo* dbi4) {
        DsKey k{};
        for (int i = 0; i < 4; ++i) {
            k.b[i] = dbi4[i].buffer;
            k.o[i] = dbi4[i].offset;
            k.r[i] = dbi4[i].range;
        }
        auto it = dscache.find(k);
        if (it != dscache.end()) return it->second;
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
        dscache.emplace(k, ds);
        return ds;
    }
    // do_barrier=false 用于"连续多次 dispatch 且彼此不重叠"的场合（如逐帧 feat_extract），
    // 由调用方在循环结束后补一次 barrier()
    void dispatch(VkPipeline p, const VkDescriptorBufferInfo* dbi4, const Push& pc,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1, bool do_barrier = true) {
        const VkDescriptorSet ds = set_for(dbi4);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);
        vkCmdDispatch(cb, gx, gy, gz);
        if (do_barrier) barrier();
        if (max_per_submit > 0 && ++since_submit >= max_per_submit) {
            // 切一刀：提交已有命令并等完成，然后重新开一个 cmd buffer（begin 会先 wait+下 barrier，
            // 保证下一段看得见这一段写的数据）。
            since_submit = 0;
            submit(cur);
            wait(cur);
            begin(cur);
        }
    }
};

struct WLayer {
    int C = 0, O = 0, k = 0;
    size_t w_off = 0, b_off = 0;
    long long cm_off = -1;   // 5.4m 的 cm 偏移也按 64 位存，int 会在 2GB 处截断
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

// 数一个视频文件里到底有多少个视频包：分段校验用它判断"这段是不是写完整了"。
static int probe_packets(const std::string& ffprobe, const std::string& path) {
    const std::string o = run_capture("\"" + ffprobe +
        "\" -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets "
        "-of csv=p=0 \"" + path + "\"");
    if (o.empty()) return -1;
    return std::atoi(o.c_str());
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // 不硬编码路径：环境变量 > 本机已知路径（存在才用）> PATH
    auto pick_exe = [](const char* env, const char* local, const char* plain) {
        if (const char* e = std::getenv(env); e && *e) return std::string(e);
        std::ifstream probe(local);
        if (probe.good()) return std::string(local);
        return std::string(plain);
    };
    std::string in_path, out_path;
    std::string ffmpeg = pick_exe("FFMPEG", "D:\\ffmpeg-9.0.1\\ffmpeg.exe", "ffmpeg");
    std::string ffprobe = pick_exe("FFPROBE", "D:\\ffmpeg-9.0.1\\ffprobe.exe", "ffprobe");
    std::string model_dir = "models_rt/nanovsr_1.7m";
    int chunk_arg = 0, crf = -1, limit = 0, tile_arg_w = 0, tile_arg_h = 0, gpu_arg = -1;
    bool serial = false;                      // --serial：关掉流水线，用同一份二进制 A/B
    bool simt_arg = false;
    bool no_seg = false;        // 关掉"分段落盘"，回到单个 ffmpeg 管道（老行为）
    bool keep_parts = false;    // 拼接成功后仍保留分段目录（调试/取证用）
    int seg_chunks_arg = 4;     // 几个 chunk 合成一个分段（编码器每次重启有固定开销）                    // --simt：强制通用 SIMT 通路（不用张量核）
    int ov_arg = 0;                           // --ov：手工指定 tile 重叠像素（0=默认 96）
    double budget_arg_gb = 0.0;
    std::string downscale = "lanczos";
    bool supersample = false;   // 默认让模型工作在交付分辨率（保住模型效果）；true=先跑满4x再缩   // libplacebo downscaler: spline36|sinc|mitchell|... / lanczos=CPU
    std::string codec_pref = "auto";
    std::string max_out = "4k";       // 4k(8.9Mp, 稳) | 8k(33.2Mp, 需要能编 8K 的编码器)   // auto|av1|av1cpu|hevc|h264   // lanczos(CPU,快) | sinc(GPU,抗混叠更好)
    std::string preset = "balanced";
    bool bench_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        // 越界防护：--in 后面没有值时会读到 argv[argc]（NULL）并崩
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--in") in_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--serial") serial = true;
        else if (a == "--simt") simt_arg = true;
        else if (a == "--ov") ov_arg = std::atoi(next().c_str());
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
        else if (a == "--gpu") gpu_arg = std::atoi(next().c_str());
        else if (a == "--no-seg") no_seg = true;
        else if (a == "--seg-chunks") seg_chunks_arg = std::atoi(next().c_str());
        else if (a == "--keep-parts") keep_parts = true;
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
                        "stream=width,height,r_frame_rate,avg_frame_rate -of csv=p=0 \"" +
                        in_path + "\"";
    std::string info = run_capture(probe);
    int W = 0, H = 0;
    double fps = 30.0;
    {
        // csv=p=0 给 "W,H,r_frame_rate,avg_frame_rate"。VFR 素材（手机录屏）的
        // r_frame_rate 常常是容器基准 1000/1，优先用 avg_frame_rate，否则输出时长会错。
        std::vector<std::string> fld;
        for (size_t p0 = 0; p0 <= info.size();) {
            size_t p1 = info.find(',', p0);
            if (p1 == std::string::npos) p1 = info.size();
            fld.push_back(info.substr(p0, p1 - p0));
            if (p1 == info.size()) break;
            p0 = p1 + 1;
        }
        auto parse_fps = [](const std::string& v) -> double {
            int num = 0, den = 1;
            if (std::sscanf(v.c_str(), "%d/%d", &num, &den) == 2 && den > 0)
                return (double)num / den;
            return 0.0;
        };
        if (fld.size() >= 2) {
            W = std::atoi(fld[0].c_str());
            H = std::atoi(fld[1].c_str());
        }
        const double fps_avg = (fld.size() >= 4) ? parse_fps(fld[3]) : 0.0;
        const double fps_raw = (fld.size() >= 3) ? parse_fps(fld[2]) : 0.0;
        if (fps_avg > 0.1) fps = fps_avg;
        else if (fps_raw > 0.1) fps = fps_raw;
    }
    if (W <= 0 || H <= 0) {
        std::printf("无法读取视频信息: %s\n", info.c_str());
        return 1;
    }
    std::printf("nanovsr build %s %s\n", __DATE__, __TIME__);
    auto has_ext = [](const std::string& p, std::initializer_list<const char*> exts) {
        std::string low = p;
        for (auto& ch : low) ch = (char)std::tolower((unsigned char)ch);
        for (auto e : exts)
            if (low.size() > std::strlen(e) && low.compare(low.size() - std::strlen(e), std::strlen(e), e) == 0)
                return true;
        return false;
    };
    const bool in_is_image = has_ext(in_path, {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"});
    // 手机竖拍：ffmpeg 解码时默认按显示矩阵把画面转正，但 ffprobe 报的是旋转前的宽高。
    // 不跟着换，就会把 480x854 的帧塞进 854x480 的缓冲 —— 输出斜切花屏，而且不报错。
    if (!in_is_image) {
        const std::string rot =
            run_capture("\"" + ffprobe +
                        "\" -v error -select_streams v:0 -show_entries stream_side_data=rotation -of csv=p=0 \"" +
                        in_path + "\"");
        if (rot.find("90") != std::string::npos || rot.find("270") != std::string::npos) {
            std::printf("注意: 输入带 %s 度旋转元数据，ffmpeg 解码时已转正，按 %dx%d 处理\n",
                        rot.c_str(), H, W);
            std::swap(W, H);
        }
    }
    if (in_is_image) {
        if (limit == 0) limit = 1;                 // 单帧模式
        std::printf("输入: %s  %dx%d  (单帧图片)\n", in_path.c_str(), W, H);
    } else {
        std::printf("输入: %s  %dx%d  %.3f fps\n", in_path.c_str(), W, H, fps);
    }

    // 4x 后超过 4K（H.264 上限）就先按整数倍降采样：1080p -> /2 -> 960x540 -> 4x -> 3840x2160
    int pre = 1;
    bool gpu_downscale = false;
    // 输出总像素上限：8K = 7680x4320 = 33.18 Mp。超了就按比例压（不是硬性 /2），
    // 保持宽高比、取偶数边长，4x 之后正好落在上限内。
    // 超采样：模型永远按 4x 跑满（保锐度），超出交付上限时在**编码端**缩，
    // 而不是先把输入缩了（输入缩会永久丢细节）。
    int out_w = W * 4, out_h = H * 4;
    {
        const double MAX_OUT_PX = (max_out == "8k") ? 7680.0 * 4320.0 : 4096.0 * 2304.0;
        const double out_px = (double)out_w * out_h;
        if (out_px > MAX_OUT_PX) {
            const double k = std::sqrt(MAX_OUT_PX / out_px);
            // 按 4 对齐：否则 out_w 不是 4 的倍数时，W = out_w/4 再乘 4 会差 2 像素
            out_w = std::max(4, ((int)std::floor(out_w * k)) & ~3);
            out_h = std::max(4, ((int)std::floor(out_h * k)) & ~3);
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
    if (pre > 1 && supersample) {
        // 超采样：模型按满 4x 跑，输出端缩到交付尺寸。优先 libplacebo（GPU，不占 CPU），
        // 退回 swscale lanczos。只有真的要用 GPU 缩放时才去探测 filter 列表——
        // 之前这里无条件跑一次 ffmpeg -filters（每次启动多一个子进程，纯浪费）。
        bool placebo = false;
        if (downscale != "lanczos") {
            const std::string fl = run_capture("\"" + ffmpeg + "\" -hide_banner -filters");
            placebo = fl.find("libplacebo") != std::string::npos;
        }
        if (placebo) {
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
    } else if (pre > 1) {
        // 默认路径：模型工作在交付分辨率（W x H），解码端把输入缩到 W x H，
        // 否则帧切片错位（解码输出的是原始尺寸）。输出端不需要再缩。
        dec_scale = " -vf scale=" + std::to_string(W) + ":" + std::to_string(H) + ":flags=lanczos";
        std::printf("预处理: 解码缩到 %dx%d -> 模型 4x -> 交付 %dx%d\n", W, H, out_w, out_h);
    }
    std::printf("输出分辨率: %dx%d\n", W * 4, H * 4);

    // ---- 模型清单 ----
    // 权重目录回退：命令行默认按仓库布局 models_rt/<size> 找；发布包把权重放在
    // models/models_rt/ 下，而且可能只装了其中一档。找不到就按 644k/1.7m/5.4m
    // 的顺序找第一个可用的，免得用户被逼着敲 --model 绝对路径。
    {
        auto has_w = [](const std::string& d) {
            std::ifstream f(d + "/weights_rt.json");
            return f.good();
        };
        if (!has_w(model_dir)) {
            std::string alt;
            if (model_dir.rfind("models_rt/", 0) == 0) alt = "models/" + model_dir;
            if (!alt.empty() && has_w(alt)) {
                std::printf("权重目录: %s\n", alt.c_str());
                model_dir = alt;
            } else {
                const char* sizes[] = {"644k", "1.7m", "5.4m"};
                const char* roots[] = {"models_rt/", "models/models_rt/"};
                for (const char* rt : roots) {
                    for (const char* sz : sizes) {
                        std::string d = std::string(rt) + "nanovsr_" + sz;
                        if (has_w(d)) { model_dir = d; break; }
                    }
                    if (has_w(model_dir)) break;
                }
                if (has_w(model_dir))
                    std::printf("权重目录: %s（请求的档位没装，自动改用已装的一档）\n", model_dir.c_str());
            }
        }
    }
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
    // 数值解析：key 缺失时返回 def。原实现把 npos 直接当偏移用，等于越界读。
    auto jint = [&](const std::string& key, long long def) {
        size_t p = js.find("\"" + key + "\"");
        if (p == std::string::npos) return (int)def;
        p = js.find(':', p);
        if (p == std::string::npos) return (int)def;
        return (int)std::strtoll(js.c_str() + p + 1, nullptr, 10);
    };
    std::vector<WLayer> layers;
    std::map<std::string, int> idx;
    {
        size_t p = 0;
        while ((p = js.find("\"name\"", p)) != std::string::npos) {
            size_t pe = js.find('}', p);
            if (pe == std::string::npos) break;
            std::string obj = js.substr(p, pe - p);
            // 64 位读：5.4m 的权重偏移早就超过 2GB 的 int 上限
            auto getll = [&](const char* key) -> long long {
                size_t q = obj.find(std::string("\"") + key + "\"");
                if (q == std::string::npos) return -1;
                q = obj.find(':', q);
                if (q == std::string::npos) return -1;
                return std::strtoll(obj.c_str() + q + 1, nullptr, 10);
            };
            auto get = [&](const char* key) { return (int)getll(key); };
            size_t q = obj.find(':', obj.find("\"name\""));
            size_t q1 = obj.find('"', q);
            size_t q2 = obj.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) break;
            WLayer L;
            L.C = get("C");
            L.O = get("O");
            L.k = get("k");
            L.w_off = (size_t)getll("w_off");
            L.b_off = (size_t)getll("b_off");
            L.cm_off = getll("cm_off");
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
            size_t c = (b == std::string::npos) ? b : js.find(',', b);
            if (b != std::string::npos && c != std::string::npos) {
                prelu1 = (float)std::atof(js.c_str() + b + 1);
                prelu2 = (float)std::atof(js.c_str() + c + 1);
            }
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
    g.gpu_index = gpu_arg;
    g.init();
    auto wblob = read_file(model_dir + "/weights_rt.bin");
    auto cmblob = read_file(model_dir + "/weights_cm.bin");
    if (wblob.empty() || cmblob.empty()) {
        std::printf("权重文件是空的: %s\n", model_dir.c_str());
        return 1;
    }
    // 清单缺 key 时 getll 返回 -1，转成 size_t 就是天文数字偏移，跑起来只会 DEVICE_LOST。
    // 这里逐层核对一次，出错直接带层名退出。
    for (const auto& kv : idx) {
        const WLayer& L = layers[(size_t)kv.second];
        const size_t need_rt = (size_t)L.C * L.k * L.k * L.O * 4;
        // 1x1 层（fusion）只存 1 个 tap，3x3 层 9 个；每 tap 是 [C/16][ocg][16][16] 的 f16 块
        const size_t need_cm = (size_t)L.k * (size_t)L.k * ((size_t)L.C / 16) * (((((size_t)L.O + 63) / 64) * 4)) * 512;
        // 必须用减法判越界：L.w_off 为 (size_t)-1 时 w_off + need_rt 会回绕成小值，
        // 加法写法反而漏检（评审指出的真 bug）。
        const size_t need_b = (size_t)L.O * 4;
        const bool bad_w = L.w_off > wblob.size() || need_rt > wblob.size() - L.w_off;
        const bool bad_b = L.b_off > wblob.size() || need_b > wblob.size() - L.b_off;
        const bool bad_cm = L.cm_off >= 0 &&
            ((size_t)L.cm_off > cmblob.size() || need_cm > cmblob.size() - (size_t)L.cm_off);
        if (bad_w || bad_b || bad_cm) {
            std::printf("权重清单越界: 层 %s C=%d O=%d k=%d w_off=%zu b_off=%zu cm_off=%lld"
                        "（rt %zu B, cm %zu B）\n",
                        kv.first.c_str(), L.C, L.O, L.k, L.w_off, L.b_off, L.cm_off,
                        wblob.size(), cmblob.size());
            return 1;
        }
    }
    // 权重是每层每次卷积都要读的最热数据，必须住显存。原来 alloc() 走 HOST_VISIBLE，
    // 非 ReBAR 机器上等于每个 dispatch 都在 PCIe 上读权重。
    auto upload_const = [&](const std::vector<char>& blob) {
        Buf b = g.alloc_device(blob.size());
        Buf st = g.alloc_host(blob.size());
        std::memcpy(st.mapped, blob.data(), blob.size());
        g.begin(0);
        g.copy(st, 0, b, 0, blob.size());
        g.submit(0);
        g.wait(0);
        vkDestroyBuffer(g.dev, st.buf, nullptr);
        vkFreeMemory(g.dev, st.mem, nullptr);
        return b;
    };
    Buf bw = upload_const(wblob);
    Buf bwcm = upload_const(cmblob);

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
    if (chunk > 30) std::printf("注意: chunk 上限 30（训练窗长），已从 %d 截到 30\n", chunk);
    chunk = std::min(chunk, 30);
    // 单帧图片：时间维只有 1 帧有用，按 30 分配等于白扔 29/30 的显存，还会把 tile 挤到很小
    if (in_is_image && limit > 0) chunk = std::min(chunk, limit);
    // ---- 显存账本：先算账，再按账分配 ----
    // 每像素的固定开销（不含时间维）
    // ---- 选计算通路 ----
    // 张量核需要：扩展 + 16x16x16 f16 组合 + subgroup=32 + shared 装得下。
    // 任何一条不满足就走通用 SIMT（conv_gen / conv3x3_c64），结果一致、只是慢。
    const int CPAD_ = ((int)nf_ + 31) & ~31;
    // 必须与 shaders/conv3x3_cm.comp 里的 TW/TH 一致（1x1 内核另有自己的 TW/TH）
    constexpr int CM_TW = 16, CM_TH = 8;
    const int cm_shared_need = 6 * 34 * CPAD_ * 2 + 8 * 256 * 4;   // conv3x3_cm.comp 静态 shared
    bool use_cm = true;
    std::string cm_why;
    if (simt_arg) {
        use_cm = false;
        cm_why = "命令行 --simt";
    } else if (!g.cm_ok) {
        use_cm = false;
        cm_why = g.cm_reason;
    } else if (g.subgroup_size != 32) {
        use_cm = false;
        cm_why = "subgroup 宽度 " + std::to_string(g.subgroup_size) + "（张量核 shader 按 32 写死）";
    } else if ((int)g.max_shared < cm_shared_need) {
        use_cm = false;
        cm_why = "shared memory 上限 " + std::to_string(g.max_shared) + " B < 需要 " +
                 std::to_string(cm_shared_need) + " B";
    }
    std::printf("计算通路: %s%s%s\n",
                use_cm ? "张量核 cooperative matrix" : "通用 SIMT (conv_gen / conv3x3_c64)",
                use_cm ? "" : "  —— ", cm_why.c_str());
    if (!use_cm) {
        // 通用通路一次 dispatch 就要几百毫秒，整块 tile 塞进一个提交会让 GPU 长时间不返回，
        // 驱动直接判定卡死（实测 VK_ERROR_DEVICE_LOST / 0xC0000409）。每 8 次派发切一刀，
        // 单次提交控制在几十毫秒量级；多出的提交-等待开销在这条通路上可以忽略。
        g.max_per_submit = 8;
        serial = true;   // 两个槽交替提交时驱动仍会判定超时（实测），这条通路固定串行
        std::printf("  注意: 通用 SIMT 通路约为张量核的 1/20 速度，建议只处理单张图片或极短片段\n");
    }

    const double Cw = (double)nf_;
    const double S_PX = Cw * 4 * 3        // bp0/bp1/bcatus
                        + Cw * 4 * 2      // bcat（2C 通道）
                        + Cw * 4          // bfused
                        + Cw * 4 * (use_cm ? 2 : 4)   // bconv1（4C 通道，张量核 f16 / SIMT f32）
                        + Cw * 4 * 4      // bup1（C 通道，2x 分辨率 = 4 倍像素）
                        + (use_cm ? 4096 : 8192)       // bconv2（128 通道 x 16 倍像素）
                        + 2048            // bup2（32 通道 x 16 倍像素）
                        + 192;            // blast（3 通道 x 16 倍像素）
    // 每像素每帧（时间维）
    const double S_PX_F = 12      // bx（3 通道 f32）
                          + 4     // bu8
                          + Cw * 4 * 3   // feat/fwd/bwd
                          + 64    // bout
                          + 64;   // 流水线第二套槽的 bout_b
                          // 4 份 staging（bstage_in/_b、bstage_out/_b = 4+64 各两份）走
                          // alloc_host，是系统内存，不占显存堆，不计入账本
    const double drv = (double)g.device_budget();
    // GPU 降采样（libplacebo）会和我们共用显存堆，额外预留 512 MB
    const double budget = budget_arg_gb > 0
                              ? budget_arg_gb * 1e9
                              : std::max(2e9, drv * 0.80 - (gpu_downscale ? 512e6 : 0.0));
    const double w_bytes = (double)(wblob.size() + cmblob.size());

    struct TileRect {
        int x, y, w, h;
        int cx0, cx1, cy0, cy1;   // 本块负责写出的中心区（相邻块重叠区的正中线）
    };
    std::vector<TileRect> tiles;
    // 重叠：每个方向丢 ov/2。空间感受野半径 = 每条链的 3x3 层数 + 1（644k 13、1.7m 21、
    // 5.4m 31），所以按 2*(nb+4) 自适应留 4 px 余量就够了。实测把固定 96 收窄到自适应值：
    // 1.7m 4.76 -> 4.06 s（-15%）、5.4m 9.04 -> 8.45 s（-6.5%），与 96 的输出差 69~73 dB
    // （maxabs 4），接缝不可见。--ov 可手工覆盖。
    int ov = std::max(16, ov_arg > 0 ? ov_arg : 2 * (nb + 4));
    // --ov 给大了（或帧本身很小）会让后面的步长退化成 1 像素、tile 数爆炸
    if (ov > std::min(W, H) / 2) {
        const int ovc = std::max(16, std::min(W, H) / 2);
        std::printf("注意: --ov %d 超过帧短边的一半，已收窄到 %d\n", ov, ovc);
        ov = ovc;
    }
    std::vector<int> xs, ys;
    // tile 尺寸搜索：在预算内选"总处理像素最少"的 (tw,th)。
    // 因为 tile 与帧尺寸的整除关系影响很大（960 宽用 366 会切 4 列），开方取整并不最优。
    auto grid_for = [&](int tw, int th, std::vector<int>& gx, std::vector<int>& gy,
                        size_t& total) {
        gx.clear();
        gy.clear();
        // 手动 --tile 可能比帧还大：不夹住会让 min(x, W - tw) 变成负坐标，后面直接越界
        tw = std::min(tw, W);
        th = std::min(th, H);
        // tile 必须比重叠大，否则 tw-ov <= 0 让步长退化到 1 像素：
        // --tile 32x32 配 ov=32 曾在 98x62 的图上切出 2077 块（慢 300 倍）。
        tw = std::max(tw, std::min(W, ov + 16));
        th = std::max(th, std::min(H, ov + 16));
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
        const int atw0 = tw0, ath0 = th0;              // 自动结果留底（手动 tile 太小时要退回）
        if (tile_arg_w > 0 && tile_arg_h > 0) {        // 手动指定优先
            tw0 = std::min(tile_arg_w, W);
            th0 = std::min(tile_arg_h, H);
            // 手动 tile 也要守预算：否则 --tile 2000x1000 在 1080p 上会直接 vkAllocateMemory 失败
            const double per_px = S_PX + S_PX_F * chunk;
            const double foot = (double)tw0 * th0 * per_px + w_bytes;
            if (foot > budget && budget > w_bytes) {
                const double k2 = std::sqrt((budget - w_bytes) / ((double)tw0 * th0 * per_px));
                const int nw = std::max(64, (int)(tw0 * k2) / 16 * 16);
                const int nh = std::max(64, (int)(th0 * k2) / 16 * 16);
                std::printf("注意: --tile %dx%d 超出显存预算（需 %.2f GB / 有 %.2f GB），"
                            "已按比例缩到 %dx%d\n", tw0, th0, foot / 1e9, budget / 1e9, nw, nh);
                tw0 = std::min(nw, W);
                th0 = std::min(nh, H);
            }
            if (tw0 < std::min(W, ov + 16) || th0 < std::min(H, ov + 16)) {
                // 比感受野重叠还小的 tile 没有足够 halo，硬用会出接缝；直接退回自动分块
                std::printf("注意: --tile %dx%d 小于感受野重叠 %d，已忽略，改用自动分块 %dx%d\n",
                            tw0, th0, ov, atw0, ath0);
                tw0 = atw0;
                th0 = ath0;
            }
        }
    }
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
            // 接缝取实际重叠区的中线：末尾那块被 min(x, W-tw) 往回推，重叠会大于 ov，
            // 用固定的 ov/2 会造成漏写或重复写（表现为一条缝或后写覆盖前写）。
            auto pend_x = [&](size_t k) { return xs[k] + std::min(tw0, W - xs[k]); };
            auto pend_y = [&](size_t k) { return ys[k] + std::min(th0, H - ys[k]); };
            tr.cx0 = (xi == 0) ? 0 : (pend_x(xi - 1) + xs[xi]) / 2;
            tr.cx1 = (xi + 1 == xs.size()) ? W : (pend_x(xi) + xs[xi + 1]) / 2;
            tr.cy0 = (yi == 0) ? 0 : (pend_y(yi - 1) + ys[yi]) / 2;
            tr.cy1 = (yi + 1 == ys.size()) ? H : (pend_y(yi) + ys[yi + 1]) / 2;
            tr.cx0 = std::max(tr.cx0, tr.x);
            tr.cx1 = std::min(tr.cx1, tr.x + tr.w);
            tr.cy0 = std::max(tr.cy0, tr.y);
            tr.cy1 = std::min(tr.cy1, tr.y + tr.h);
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
    // 中间缓冲全部 DEVICE_LOCAL：只有 CPU 直接读写的 staging 才用 alloc_host
    Buf bx = g.alloc_device((VkDeviceSize)T * px * 3 * 4);
    Buf bu8 = g.alloc_device((VkDeviceSize)T * px * 4);              // 解码后的打包 RGB
    Buf bfeat = g.alloc_device((VkDeviceSize)T * px * F * 4);
    Buf bfwd = g.alloc_device((VkDeviceSize)T * px * F * 4);
    Buf bbwd = g.alloc_device((VkDeviceSize)T * px * F * 4);
    Buf bp0 = g.alloc_device((VkDeviceSize)px * F * 4);
    Buf bp1 = g.alloc_device((VkDeviceSize)px * F * 4);
    Buf bcatus = g.alloc_device((VkDeviceSize)px * F * 4);
    Buf bcat = g.alloc_device((VkDeviceSize)px * 2 * F * 4);
    Buf bfused = g.alloc_device((VkDeviceSize)px * F * 4);
    Buf bconv1 = g.alloc_device((VkDeviceSize)px * 4 * F * (use_cm ? 2 : 4));   // 张量核 f16 / SIMT f32
    Buf bup1 = g.alloc_device((VkDeviceSize)px * 4 * F * 4);
    Buf bconv2 = g.alloc_device((VkDeviceSize)px * 16 * 128 * (use_cm ? 2 : 4));
    Buf bup2 = g.alloc_device((VkDeviceSize)px * 16 * 32 * 4);
    Buf blast = g.alloc_device((VkDeviceSize)px * 16 * 3 * 4);
    Buf bout = g.alloc_device((VkDeviceSize)T * px * 16 * 4);
    Buf bstage_in = g.alloc_host((VkDeviceSize)T * px * 4);
    Buf bstage_out = g.alloc_host((VkDeviceSize)T * px * 16 * 4);
    // 流水线第二个槽：CPU 打包/合成与 GPU 推理重叠时需要各一套
    const VkDeviceSize B_OUT = (VkDeviceSize)T * px * 16 * 4;
    Buf bout_b = g.alloc_device(B_OUT);
    Buf bstage_in_b = g.alloc_host((VkDeviceSize)T * px * 4);
    Buf bstage_out_b = g.alloc_host(B_OUT);

    const VkPipeline pConv64 = g.pipe("shaders/conv3x3_c64.spv");
    const VkPipeline pConvGen = g.pipe("shaders/conv_gen.spv");
    const VkPipeline pConvSmall = g.pipe("shaders/conv3x3_smallc.spv");
    // 张量核 shader 按 num_feat 编译了变体（C 是编译期常量）；SIMT 通路完全不需要它们
    VkPipeline pConvCM = VK_NULL_HANDLE, pConvCM16 = VK_NULL_HANDLE;
    VkPipeline pShuffle = VK_NULL_HANDLE;   // 张量核通路读 f16 输入，SIMT 通路读 f32
    if (use_cm) {
        std::string cm_path = "shaders/conv3x3_cm_" + std::to_string((int)F) + ".spv";
        std::string cm16_path = "shaders/conv3x3_cm_" + std::to_string((int)F) + "_f16.spv";
        if (!std::ifstream(cm_path).good() || !std::ifstream(cm16_path).good()) {
            std::printf("缺少张量核变体: %s / %s（build.bat 按 num_feat 生成，C=%d）\n",
                        cm_path.c_str(), cm16_path.c_str(), (int)F);
            return 1;   // 宁可启动失败，也不要静默套用 C=64 的错内核
        }
        std::printf("张量核 shader: %s\n", cm_path.c_str());
        pConvCM = g.pipe(cm_path);
        pConvCM16 = g.pipe(cm16_path);
        pShuffle = g.pipe("shaders/shuffle_prelu_f16.spv");
    } else {
        pShuffle = g.pipe("shaders/shuffle_prelu.spv");   // 非 IN_F16：f32 输入
    }
    // 1x1 张量核（fusion）：3x3 版要把整块 halo tile 放 shared，C=192（5.4m）要 72KB 直接超限；
    // 1x1 版按 16 通道流式载入，shared 恒定 16KB，任何 C 都装得下。
    std::map<int, VkPipeline> pipe1x1;
    auto cm1x1 = [&](int cc) -> VkPipeline {
        auto it = pipe1x1.find(cc);
        if (it != pipe1x1.end()) return it->second;
        std::string p = "shaders/conv1x1_cm_" + std::to_string(cc) + ".spv";
        VkPipeline v = VK_NULL_HANDLE;
        if (!std::ifstream(p).good()) {
            std::printf("缺少 1x1 张量核变体: %s（该层退回通用 SIMT）\n", p.c_str());
        } else {
            v = g.pipe(p);
        }
        pipe1x1[cc] = v;
        return v;
    };
    const VkPipeline pFinal = g.pipe("shaders/bilinear4_add_u8.spv");
    const VkPipeline pAdd = g.pipe("shaders/add.spv");
    const VkPipeline pCat = g.pipe("shaders/cat2.spv");
    const VkPipeline pF32 = g.pipe("shaders/u8rgb_to_f32.spv");

    auto run_conv64 = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                          VkDeviceSize out_off, int h, int w, int O, float slope, bool out_f16 = false) {
        const WLayer& L = layers[idx[name]];
        if (use_cm && L.cm_off >= 0) {
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
            g.dispatch(out_f16 ? pConvCM16 : pConvCM, dbi, pc, (uint32_t)((w + CM_TW - 1) / CM_TW),
                       (uint32_t)((h + CM_TH - 1) / CM_TH), (uint32_t)((O + 63) / 64));
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
        pc.k = L.k;          // conv_gen 用 p.k 当卷积核边长（1 或 3）
        if ((int)F == 64) {
            g.dispatch(pConv64, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
        } else {
            // 通用 shader 按 p.C_ 循环，C != 64 时唯一正确的兜底
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
        }
    };
    auto run_convg = [&](const std::string& name, Buf& in, VkDeviceSize in_off, Buf& out,
                         VkDeviceSize out_off, int h, int w, float slope, bool do_barrier = true) {
        const WLayer& L = layers[idx[name]];
        // 1x1 层（fusion）走张量核：它只占全模型 0.5% 的 MAC，走标量 conv_gen 却吃掉 20% 时间
        if (use_cm && L.cm_off >= 0 && L.k == 1) {
            const VkPipeline pp = cm1x1(L.C);
            if (pp != VK_NULL_HANDLE) {
                VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                                 {bwcm.buf, (VkDeviceSize)L.cm_off, VK_WHOLE_SIZE},
                                                 {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                                 {out.buf, out_off, VK_WHOLE_SIZE}};
                Push pc;
                pc.C_ = L.C;
                pc.H = h;
                pc.W = w;
                pc.O = L.O;
                pc.k = (L.O + 63) / 64 * 64;   // 权重里输出通道补齐后的跨度
                pc.slope = slope;
                g.dispatch(pp, dbi, pc, (uint32_t)((w + 31) / 32), (uint32_t)((h + 3) / 4),
                           (uint32_t)((L.O + 63) / 64), do_barrier);
                return;
            }
        }
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
            g.dispatch(pConvSmall, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1,
                       do_barrier);
        } else {
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64), 1, 1, do_barrier);
        }
    };

    // 双向链每层的"计划"在初始化时定死：层下标、通道数、权重偏移、grid 全部预解析，
    // 热循环里不再拼 std::string + 查 std::map（每帧每层一次，tile 多时是纯 CPU 浪费）。
    struct ConvPlan {
        const WLayer* L = nullptr;
        int O = 0, k = 0, gz = 1;
    };
    auto make_plan = [&](const std::string& name) {
        ConvPlan cp;
        cp.L = &layers[idx[name]];
        cp.O = cp.L->O;
        cp.k = (cp.O + 63) / 64 * 64;
        cp.gz = (cp.O + 63) / 64;
        return cp;
    };
    std::vector<ConvPlan> fwdPlan((size_t)nb), bwdPlan((size_t)nb);
    for (int b = 0; b < nb; ++b) {
        fwdPlan[(size_t)b] = make_plan("forward_" + std::to_string(b));
        bwdPlan[(size_t)b] = make_plan("backward_" + std::to_string(b));
    }
    // 双向链的块卷积（C=O=num_feat，输出 f32）；up1/up2 等仍走 run_conv64
    auto run_plan = [&](const ConvPlan& cp, Buf& in, VkDeviceSize in_off, Buf& out,
                        VkDeviceSize out_off, int h, int w, float slope) {
        const WLayer& L = *cp.L;
        Push pc;
        pc.C_ = (int)F;
        pc.H = h;
        pc.W = w;
        pc.O = cp.O;
        pc.slope = slope;
        if (use_cm && L.cm_off >= 0) {
            pc.k = cp.k;      // 张量核：输出通道补齐后的权重跨度
            VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                             {bwcm.buf, (VkDeviceSize)L.cm_off, VK_WHOLE_SIZE},
                                             {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                             {out.buf, out_off, VK_WHOLE_SIZE}};
            g.dispatch(pConvCM, dbi, pc, (uint32_t)((w + CM_TW - 1) / CM_TW), (uint32_t)((h + CM_TH - 1) / CM_TH),
                       (uint32_t)cp.gz);
            return;
        }
        VkDescriptorBufferInfo dbi[4] = {{in.buf, in_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.w_off, VK_WHOLE_SIZE},
                                         {bw.buf, L.b_off, VK_WHOLE_SIZE},
                                         {out.buf, out_off, VK_WHOLE_SIZE}};
        pc.k = L.k;          // 通用通路：核边长（1 或 3）
        if ((int)F == 64)
            g.dispatch(pConv64, dbi, pc, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
        else
            g.dispatch(pConvGen, dbi, pc, (uint32_t)(((size_t)h * w + 63) / 64));
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
    std::string enc_pre;                 // 编码命令前缀（不含输出文件名）
    bool seg_mode = false;               // 分段落盘：每 chunk 一个独立 mkv，可单独校验/重来
    std::string seg_dir, seg_list;       // <out>.parts / 其中的 list.txt
    long long done_chunks = 0;           // 断点续跑：已有且校验通过的分段数
    int seg_chunks = 4;                  // 每个分段包含几个 chunk
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
        // --codec 接受别名：av1 -> libsvtav1、hevc -> libx265、h264 -> libx264
        auto codec_want = [&](const char* n) {
            if (codec_pref == "auto" || codec_pref == n) return true;
            const std::string s = n;
            if (codec_pref == "av1" && s == "libsvtav1") return true;
            if (codec_pref == "hevc" && s == "libx265") return true;
            if (codec_pref == "h264" && s == "libx264") return true;
            return false;
        };
        for (auto& c : cands) {
            if (!codec_want(c.name)) continue;
            if (probe_enc(c.venc)) {
                venc = c.venc;
                break;
            }
            std::printf("  编码器 %s 不可用，跳过\n", c.name);
        }
        if (venc.empty()) {
            if (codec_pref != "auto")
                std::printf("  未知或不可用的编码器 %s，回退 libx264\n", codec_pref.c_str());
            venc = "-c:v libx264 -crf " + qof("x264") + " -preset veryfast";
        }
        const bool out_is_image = has_ext(out_path, {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"});
        std::string pix_tail = " -pix_fmt yuv420p";
        if (out_is_image) {
            venc = has_ext(out_path, {".png"}) ? "-c:v png" : "-c:v mjpeg -q:v 2";
            pix_tail = " -pix_fmt rgb24 -frames:v 1";
            enc_scale.clear();
        }
        std::printf("编码器: %s\n", venc.c_str());
        // 长路径 / 中文路径下 2048 字节会静默截断，直接用 string 拼
        char nums[96];
        std::snprintf(nums, sizeof(nums), "-s %dx%d -r %.6f -i -", W * 4, H * 4, fps);
        // 编码命令前缀（输出文件名留到最后拼）：单段模式直接拼 out_path，
        // 分段模式每个 chunk 拼各自的段文件。
        enc_pre = "\"" + ffmpeg + "\" -v error -y -f rawvideo -pix_fmt rgb24 " + nums + " " +
                  venc + enc_scale + pix_tail + " ";

        // 分段落盘：一个 chunk 一个独立 mkv 段，段写完就用 ffprobe 数包验证。
        // 好处是"某一段坏了只重来这一段"，而且进程被硬杀/断电时已完成的分段还在，
        // 下次同样的命令重跑会自动跳过它们（断点续跑）。
        // 图片输出只有一帧，没有分段的意义；--no-seg 可回到单个长管道的旧行为。
        seg_mode = !out_is_image && !no_seg;
        if (seg_mode) {
            seg_chunks = std::max(1, std::min(64, seg_chunks_arg));
            seg_dir = out_path + ".parts";
            seg_list = seg_dir + "/list.txt";
            const std::string seg_meta = seg_dir + "/meta.txt";
            std::error_code ec;
            std::filesystem::create_directories(seg_dir, ec);
            char sig[768];
            std::snprintf(sig, sizeof(sig),
                          "in=%s\nW=%d\nH=%d\nfps=%.6f\nvenc=%s\nscale=%s\ncrf=%d\nchunk=%d\nov=%d\nsegc=%d\n",
                          in_path.c_str(), W, H, fps, venc.c_str(), enc_scale.c_str(), crf, chunk, ov,
                          seg_chunks);
            auto slurp = [](const std::string& path) {
                std::ifstream f(path, std::ios::binary);
                return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            };
            const std::string want = sig;
            if (slurp(seg_meta) != want) {
                // 参数或输入变了：旧段一律作废
                for (auto& e : std::filesystem::directory_iterator(seg_dir, ec)) {
                    if (e.path().extension() == ".mkv") std::filesystem::remove(e.path(), ec);
                }
                std::ofstream mf(seg_meta, std::ios::binary | std::ios::trunc);
                mf << want;
                mf.close();
                done_chunks = 0;
            } else {
                // 续跑：从 0 数连续存在的段。上一轮每段都是"写完并校验过"才留在磁盘上的，
                // 唯一可能残缺的是被硬杀时正在写的那一段 —— 它必然是最后一个，所以只校验它。
                auto seg_file = [&](long long i) {
                    char b[64];
                    std::snprintf(b, sizeof(b), "/%05lld.mkv", i);
                    return seg_dir + b;
                };
                while (true) {
                    std::ifstream probe(seg_file(done_chunks), std::ios::binary | std::ios::ate);
                    if (!probe.good() || probe.tellg() < 4096) break;
                    ++done_chunks;
                }
                while (done_chunks > 0) {
                    const std::string last = seg_file(done_chunks - 1);
                    // 只认"整 chunk"的段：截断的文件也可能数出若干包，只有包数是
                    // chunk 的整数倍才说明它停在 chunk 边界上（真·最后一段不足一个
                    // chunk 时会被丢弃重算，代价只是重算这一段）。
                    const int got = probe_packets(ffprobe, last);
                    if (got > 0 && got % chunk == 0) break;
                    std::printf("续跑: 最后一段 %lld 不完整（%d 帧），丢弃重来\n", done_chunks - 1, got);
                    std::remove(last.c_str());
                    --done_chunks;
                }
                if (done_chunks > 0)
                    std::printf("续跑: 已有 %lld 个完整分段，跳过它们\n", done_chunks);
            }
            std::printf("分段落盘: %s\n", seg_dir.c_str());
        } else {
            fout = popen_w(enc_pre + "\"" + out_path + "\"");
            if (!fout) {
                std::printf("无法启动编码\n");
                return 1;
            }
        }
    }

    // ---- 主循环：按 (chunk 帧) x (空间 tile) 处理 ----
    const int FW = W, FH = H;                    // 帧尺寸（W/H 在 tile 循环里会被改成 tile 尺寸）
    const size_t FOFF = 4;                       // 放大倍数
    const size_t FRAME_BYTES = (size_t)FW * FH * 3;
    const size_t OFRAME_BYTES = (size_t)FW * FOFF * FH * FOFF * 3;
    // 解码预读线程：ffmpeg 解一个 chunk 要几十到几百 ms，主线程 fread 等着就是让 GPU 空转。
    // 输入双缓冲：读线程填一份、主线程同时用另一份；每份缓冲有独立状态（0 空闲 / 1 填充中 /
    // 2 已就绪或使用中），读线程只碰状态为 0 的那份，所以最多提前一个 chunk。
    struct ChunkRef {
        std::vector<std::vector<unsigned char>>* f = nullptr;
        int n = 0;
        int slot = 0;
    };
    std::vector<std::vector<std::vector<unsigned char>>> inbuf(
        2, std::vector<std::vector<unsigned char>>((size_t)chunk,
                                                   std::vector<unsigned char>(FRAME_BYTES)));
    std::mutex rq_m;
    std::condition_variable rq_ready, rq_free;
    std::deque<ChunkRef> rq;
    int rstate[2] = {0, 0};
    bool rq_stop = false;
    long long rtotal = 0;
    std::thread reader([&] {
        for (;;) {
            int bi = -1;
            {
                std::unique_lock<std::mutex> lk(rq_m);
                rq_free.wait(lk, [&] {
                    return rq_stop || rstate[0] == 0 || rstate[1] == 0;
                });
                if (rq_stop) return;
                for (int i = 0; i < 2; ++i)
                    if (rstate[i] == 0) {
                        bi = i;
                        break;
                    }
                rstate[bi] = 1;
            }
            auto& buf = inbuf[(size_t)bi];
            int n = 0;
            for (; n < chunk; ++n) {
                if (limit > 0 && rtotal + n >= limit) break;
                if (std::fread(buf[(size_t)n].data(), 1, FRAME_BYTES, fin) != FRAME_BYTES) break;
            }
            rtotal += n;
            {
                std::lock_guard<std::mutex> lk(rq_m);
                if (n > 0) {
                    rstate[bi] = 2;
                    rq.push_back(ChunkRef{&buf, n, bi});
                } else {
                    rstate[bi] = 0;
                    rq_stop = true;          // 没有更多帧
                }
            }
            if (n > 0) {
                rq_ready.notify_one();
            } else {
                rq_ready.notify_all();
                return;
            }
        }
    });
    long long total = 0;
    double t_start = now_s();
    double acc_pack = 0, acc_rec = 0, acc_comp = 0;   // 时间分解（秒）

    // 编码写线程：4K 一帧 25MB，fwrite 阻塞主线程就白等 GPU。
    // 输出双缓冲轮转，最多一个 chunk 在写；复用前用 written 计数确认它已经写完。
    struct WriteItem {
        std::vector<unsigned char>* p = nullptr;
        size_t bytes = 0;
        int nframes = 0;
        long long idx = 0;
    };
    // 分段模式下每段一个独立 mkv 文件：一段写完立刻数包校验，包数不对说明编码器
    // 没写完（磁盘满 / 被杀），该段留在磁盘上，同命令重跑会从它重算（回退重来）。
    // 一段跨 seg_chunks 个 chunk：编码器每次启动有固定开销（1080p AV1 约 0.6 s），
    // 段太短会让编码抢成瓶颈，所以编码器要跨 chunk 常驻。
    auto seg_file = [&](long long i) {
        char b[64];
        std::snprintf(b, sizeof(b), "/%05lld.mkv", i);
        return seg_dir + b;
    };
    FILE* seg_f = nullptr;               // 当前正在写的段（常驻，跨 chunk）
    long long seg_open = -1;             // 该段编号
    long long seg_frames = 0;            // 已写进去的帧数（校验用）
    auto close_seg = [&]() -> bool {
        if (!seg_f) return true;
        const long long idx = seg_open;
        const long long want = seg_frames;
        const std::string seg = seg_file(idx);
        const int st = pclose_w(seg_f);
        seg_f = nullptr;
        seg_frames = 0;
        if (st != 0) {
            std::printf("\n分段 %lld: 编码器退出码 %d（分段已保留，重跑同命令会从该段重来）\n", idx, st);
            return false;
        }
        const int got = probe_packets(ffprobe, seg);
        if (got != (int)want) {
            std::printf("\n分段 %lld: 校验失败（数到 %d 帧，应为 %lld）（分段已保留，重跑同命令会从该段重来）\n",
                        idx, got, want);
            return false;
        }
        return true;
    };
    std::mutex wq_m;
    std::condition_variable wq_cv, wq_free_cv;
    std::deque<WriteItem> wq;
    long long pushed = 0, written = 0;
    bool wq_stop = false, wq_err = false;
    std::vector<std::vector<unsigned char>> obuf(2, std::vector<unsigned char>((size_t)chunk * OFRAME_BYTES));
    int oi = 0;
    std::thread writer([&] {
        bool bad = false;
        for (;;) {
            WriteItem it;
            {
                std::unique_lock<std::mutex> lk(wq_m);
                wq_cv.wait(lk, [&] { return wq_stop || !wq.empty(); });
                if (wq.empty()) break;                       // stop 且已写空
                it = wq.front();
                wq.pop_front();
            }
            if (bad) {                                       // 已出错：排空队列即可退出
                std::lock_guard<std::mutex> lk(wq_m);
                ++written;
                wq_free_cv.notify_all();
                continue;
            }
            if (seg_mode) {
                const long long si = it.idx / seg_chunks;
                if (si != seg_open) {
                    bad = !close_seg();                      // 上一段收尾 + 校验
                    if (!bad) {
                        seg_open = si;
                        const std::string seg = seg_file(si);
                        std::remove(seg.c_str());
                        seg_f = popen_w(enc_pre + "\"" + seg + "\"");
                        if (!seg_f) {
                            std::printf("\n分段 %lld: 启动编码器失败\n", si);
                            bad = true;
                        }
                    }
                }
                if (!bad) {
                    if (std::fwrite(it.p->data(), 1, it.bytes, seg_f) != it.bytes)
                        bad = true;
                    else
                        seg_frames += it.nframes;
                }
                if (!bad && (it.idx + 1) % seg_chunks == 0)  // 本段最后一个 chunk 到了
                    bad = !close_seg();
            } else if (fout && std::fwrite(it.p->data(), 1, it.bytes, fout) != it.bytes) {
                bad = true;
            }
            if (bad) {
                std::lock_guard<std::mutex> lk(wq_m);
                wq_err = true;
            }
            {
                std::lock_guard<std::mutex> lk(wq_m);
                ++written;
            }
            wq_free_cv.notify_all();
        }
        if (seg_mode && !bad && !close_seg()) {              // 最后一段（可能不足 seg_chunks 个 chunk）
            std::lock_guard<std::mutex> lk(wq_m);
            wq_err = true;
        }
    });

    RowPool pool((int)std::max(1u, std::thread::hardware_concurrency()));
    long long chunk_idx = 0;
    while (true) {
        // 复用的那份缓冲要等写线程写完两轮前的内容（双缓冲，最多一份在飞）
        {
            std::unique_lock<std::mutex> lk(wq_m);
            wq_free_cv.wait(lk, [&] { return written >= pushed - 1; });
        }
        ChunkRef cr{};
        {
            std::unique_lock<std::mutex> lk(rq_m);
            rq_ready.wait(lk, [&] { return !rq.empty() || rq_stop; });
            if (rq.empty()) break;               // 读线程已到 EOF / 达到 limit
            cr = rq.front();
            rq.pop_front();
        }
        rq_free.notify_one();                    // 另一份缓冲现在可以预读了
        if (wq_err) break;                       // 磁盘满 / 管道断，别再白算后面的 chunk
        const int t = cr.n;
        std::vector<std::vector<unsigned char>>& frames = *cr.f;
        std::vector<unsigned char>& out_frames = obuf[(size_t)oi];
        // 断点续跑：这些帧对应的分段已经存在且校验过，直接跳过推理（解码仍要消费掉）
        if (seg_mode && chunk_idx / seg_chunks < done_chunks) {
            {
                std::lock_guard<std::mutex> lk(rq_m);
                rstate[cr.slot] = 0;
            }
            rq_free.notify_one();
            total += t;
            ++chunk_idx;
            std::printf("\r跳过已完成分段 %lld/%lld ...", chunk_idx / seg_chunks, done_chunks);
            continue;
        }
        total += t;                              // 注意：这里先计，避免末尾重复

        // 多线程按行切分：打包是纯内存循环，合成同理（常驻线程池，不再每次建线程）
        auto parallel_rows = [&](int nrows, auto&& fn) { pool.run(nrows, fn); };
        auto pack_tile = [&](Buf& stage, const TileRect& tile_) {
            parallel_rows(t * tile_.h, [&](int r0, int r1) {
                for (int r = r0; r < r1; ++r) {
                    const int i = r / tile_.h, y = r - i * tile_.h;
                    uint32_t* drow = (uint32_t*)stage.mapped + (size_t)i * px + (size_t)y * tile_.w;
                    const unsigned char* row =
                        frames[i].data() + ((size_t)(tile_.y + y) * FW + tile_.x) * 3;
                    for (int x = 0; x < tile_.w; ++x)
                        drow[x] = (uint32_t)row[x * 3] | ((uint32_t)row[x * 3 + 1] << 8) |
                                  ((uint32_t)row[x * 3 + 2] << 16);
                }
            });
        };
        auto composite_tile = [&](const TileRect& tile_, Buf& sout) {
            const int cx0 = tile_.cx0, cx1 = tile_.cx1;
            const int cy0 = tile_.cy0, cy1 = tile_.cy1;
            const int rows = (cy1 - cy0) * 4;             // 本块的中心区输出行数
            if (rows <= 0) return;
            const int tw4 = tile_.w * 4;
            const size_t ow4 = (size_t)FW * 4;
            parallel_rows(t * rows, [&](int r0, int r1) {
                for (int r = r0; r < r1; ++r) {
                    const int i = r / rows, oy = (cy0 - tile_.y) * 4 + (r - i * rows);
                    const uint32_t* srow = (const uint32_t*)sout.mapped + (size_t)i * px * 16 +
                                           (size_t)oy * tw4;
                    unsigned char* dstrow = out_frames.data() + (size_t)i * OFRAME_BYTES +
                                            ((size_t)(tile_.y * 4 + oy) * ow4 +
                                             (size_t)tile_.x * 4) * 3;
                    for (int ox = (cx0 - tile_.x) * 4; ox < (cx1 - tile_.x) * 4; ++ox) {
                        const uint32_t v = srow[ox];
                        dstrow[ox * 3 + 0] = (unsigned char)(v & 255);
                        dstrow[ox * 3 + 1] = (unsigned char)((v >> 8) & 255);
                        dstrow[ox * 3 + 2] = (unsigned char)((v >> 16) & 255);
                    }
                }
            });
        };

        // 双缓冲：slot 0/1 各自一套 staging-in / 输出 / staging-out
        int hold[2] = {-1, -1};                       // 该槽里还没合成的 tile 序号
        Buf* sin_s[2] = {&bstage_in, &bstage_in_b};
        Buf* bout_s[2] = {&bout, &bout_b};
        Buf* sout_s[2] = {&bstage_out, &bstage_out_b};

        for (size_t ti = 0; ti < tiles.size(); ++ti) {
            const TileRect& tile = tiles[ti];
            const int slot = (int)(ti & 1);
            // 本槽的上一个 tile 跑完了吗？跑完就顺手把它合成掉（与 GPU 上的另一块重叠）
            if (hold[slot] >= 0) {
                g.wait(slot);
                const double c0 = now_s();
                composite_tile(tiles[hold[slot]], *sout_s[slot]);
                acc_comp += now_s() - c0;
                hold[slot] = -1;
            }
            W = tile.w;
            H = tile.h;
            // 1) 把本 tile 的像素打包进 staging（系统内存，多线程）
            {
                const double p0 = now_s();
                pack_tile(*sin_s[slot], tile);
                acc_pack += now_s() - p0;
            }

        const double r0 = now_s();
        const double w0 = g.wait_s;   // 录制段里 begin() 内部的等待要扣掉
        g.begin(slot);
        // staging -> 显存（一次拷贝整块）
        g.copy(*sin_s[slot], 0, bu8, 0, (VkDeviceSize)t * px * 4);
        // 转 f32：整个 chunk 一次 dispatch（bu8 与 bx 都是按帧连续的，H 直接当元素总数）
        {
            VkDescriptorBufferInfo dbi[4] = {{bu8.buf, 0, VK_WHOLE_SIZE},
                                             {bw.buf, 0, 1},
                                             {bw.buf, 0, 1},
                                             {bx.buf, 0, VK_WHOLE_SIZE}};
            Push pc;
            pc.H = (int32_t)((VkDeviceSize)t * px);
            g.dispatch(pF32, dbi, pc, (uint32_t)(((VkDeviceSize)t * px + 63) / 64));
        }
        // feat_extract：各帧读写区间互不重叠，中间的 barrier 全部省掉，循环后补一次
        for (int i = 0; i < t; ++i)
            run_convg("feat_extract", bx, (VkDeviceSize)i * px * 3 * 4, bfeat,
                      (VkDeviceSize)i * px * F * 4, H, W, act, false);
        g.barrier();
        // 双向链
        Buf* scratch[2] = {&bp0, &bp1};
        auto run_dir = [&](bool forward, Buf& dst_all) {
            Buf* pu = scratch[0];
            VkDeviceSize poff = 0;
            g.fill(*pu, 0, (VkDeviceSize)px * F * 4, 0);
            for (int k = 0; k < t; ++k) {
                const int tt = forward ? k : (t - 1 - k);
                const VkDeviceSize foff = (VkDeviceSize)tt * px * F * 4;
                VkDescriptorBufferInfo dbi[4] = {
                    {bfeat.buf, foff, VK_WHOLE_SIZE},
                    {pu->buf, poff, VK_WHOLE_SIZE},
                    {bw.buf, 0, 1},
                    {bcatus.buf, 0, VK_WHOLE_SIZE}};
                Push pc;
                const VkDeviceSize n_el = (VkDeviceSize)W * H * F;   // 只算本 tile 真实元素
                pc.H = (int32_t)n_el;
                g.dispatch(pAdd, dbi, pc, (uint32_t)((n_el + 63) / 64));
                Buf* src = &bcatus;
                int w = 0;
                const ConvPlan* plan = forward ? fwdPlan.data() : bwdPlan.data();
                for (int b = 0; b < nb; ++b) {
                    const bool last = (b + 1 == nb);
                    // 最后一层直接写进 dst_all 该帧的位置：省掉整帧 scratch->dst_all 的拷贝
                    Buf& dst = last ? dst_all : *scratch[w];
                    const VkDeviceSize doff = last ? foff : 0;
                    run_plan(plan[b], *src, 0, dst, doff, H, W, act);
                    if (!last) { src = scratch[w]; w ^= 1; }
                }
                pu = &dst_all;
                poff = foff;
            }
        };
        run_dir(true, bfwd);
        run_dir(false, bbwd);

        // 每帧后半段（和前半段在同一个 cmd buffer 里，不能重新 begin —— 那会把已录制的命令全丢掉）
        for (int i = 0; i < t; ++i) {
            VkDescriptorBufferInfo dbi[4] = {
                {bfwd.buf, (VkDeviceSize)i * px * F * 4, VK_WHOLE_SIZE},
                {bbwd.buf, (VkDeviceSize)i * px * F * 4, VK_WHOLE_SIZE},
                {bw.buf, 0, 1},
                {bcat.buf, 0, VK_WHOLE_SIZE}};
            Push pc;
            pc.C_ = (int32_t)F;          // 前向特征通道数（不是固定的 64）
            pc.H = (int32_t)F;           // 反向特征通道数
            pc.W = (int32_t)((VkDeviceSize)W * H);   // tile 真实像素数（不是缓冲上限 px）
            g.dispatch(pCat, dbi, pc, (uint32_t)(((VkDeviceSize)W * H + 63) / 64));

            run_convg("fusion", bcat, 0, bfused, 0, H, W, 0.0f);
            run_conv64("up1", bfused, 0, bconv1, 0, H, W, 4 * (int)F, 0.0f, use_cm);
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
                g.dispatch(pShuffle, d2, p2, (uint32_t)((W + 15) / 16), (uint32_t)H);
            }
            run_conv64("up2", bup1, 0, bconv2, 0, H * 2, W * 2, 128, 0.0f, use_cm);
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
                g.dispatch(pShuffle, d2, p2, (uint32_t)((W * 2 + 15) / 16), (uint32_t)(H * 2));
            }
            run_convg("conv_last", bup2, 0, blast, 0, H * 4, W * 4, 0.0f);
            {
                VkDescriptorBufferInfo d2[4] = {
                    {bx.buf, (VkDeviceSize)i * px * 3 * 4, VK_WHOLE_SIZE},
                    {blast.buf, 0, VK_WHOLE_SIZE},
                    {bw.buf, 0, 1},
                    {bout_s[slot]->buf, (VkDeviceSize)i * px * 16 * 4, VK_WHOLE_SIZE}};
                Push p2;
                p2.H = H;
                p2.W = W;
                p2.O = 4;
                g.dispatch(pFinal, d2, p2,
                           (uint32_t)(((VkDeviceSize)H * 4 * W * 4 + 63) / 64));
            }
        }
        // output copy goes into the same cmd: fence means infer+copy both done
        g.copy(*bout_s[slot], 0, *sout_s[slot], 0, (VkDeviceSize)t * px * 16 * 4);
        g.submit(slot);
        hold[slot] = (int)ti;
        acc_rec += (now_s() - r0) - (g.wait_s - w0);   // 只算纯 CPU 录制时间
        if (serial) {                      // 对照：提交后立刻等 + 合成（旧行为）
            g.wait(slot);
            const double c0 = now_s();
            composite_tile(tile, *sout_s[slot]);
            acc_comp += now_s() - c0;
            hold[slot] = -1;
        }
        }  // tile 循环结束
        // flush: wait for the two in-flight tiles and composite them
        // 按提交先后（hold 里存的 tile 序号）等待：奇数块时最后一块在 slot 0、前一块在
        // slot 1，固定 s=0..1 会先去等后提交的那块，白等一个 tile。
        int order[2] = {0, 1};
        if (hold[0] >= 0 && hold[1] >= 0 && hold[1] < hold[0]) { order[0] = 1; order[1] = 0; }
        for (int k = 0; k < 2; ++k) {
            const int s = order[k];
            if (hold[s] < 0) continue;
            g.wait(s);
            const double c0 = now_s();
            composite_tile(tiles[hold[s]], *sout_s[s]);
            acc_comp += now_s() - c0;
            hold[s] = -1;
        }

        // 输入缓冲用完（pack_tile 是最后一次读它），还给读线程
        {
            std::lock_guard<std::mutex> lk(rq_m);
            rstate[cr.slot] = 0;
        }
        rq_free.notify_one();

        // 3) 整帧交给写线程（与下一个 chunk 的 GPU 推理重叠），换另一份缓冲继续
        {
            std::lock_guard<std::mutex> lk(wq_m);
            wq.push_back(WriteItem{&out_frames, (size_t)t * OFRAME_BYTES, t, chunk_idx});
            ++pushed;
        }
        wq_cv.notify_one();
        oi ^= 1;
        ++chunk_idx;
        double now = (double)std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::printf("\r处理 %lld 帧  %.2f fps  [%zu 块/帧]", total,
                    total / std::max(1e-6, now - t_start), tiles.size());
    }
    {
        const double tot = std::max(1e-9, now_s() - t_start);
        const double busy = acc_pack + acc_rec + acc_comp;
        const double other = std::max(0.0, tot - busy - g.wait_s);
        std::printf("\n时间分解（总墙钟 %.2f s，%lld 帧，%.1f fps）:\n", tot, total,
                    (double)total / tot);
        std::printf("  打包 u8->u32  %6.1f%%  %7.2f s\n", 100.0 * acc_pack / tot, acc_pack);
        std::printf("  录制命令      %6.1f%%  %7.2f s\n", 100.0 * acc_rec / tot, acc_rec);
        std::printf("  合成 u32->RGB %6.1f%%  %7.2f s\n", 100.0 * acc_comp / tot, acc_comp);
        std::printf("  等 GPU(fence) %6.1f%%  %7.2f s\n", 100.0 * g.wait_s / tot, g.wait_s);
        std::printf("  其余(读写编)  %6.1f%%  %7.2f s\n", 100.0 * other / tot, other);
        std::printf("  → CPU 侧占比 %.1f%%（GPU 空转 = 等 GPU 占比），上限 %.1f%%\n",
                    100.0 * busy / tot, 100.0 * g.wait_s / tot);
    }
    std::printf("完成: %lld 帧\n", total);

    {
        std::lock_guard<std::mutex> lk(wq_m);
        wq_stop = true;
    }
    wq_cv.notify_all();
    reader.join();
    writer.join();
    int ret = 0;
    // 分段模式：所有段都已落盘并通过校验，按顺序 -c copy 拼成最终文件
    if (seg_mode && !wq_err) {
        std::string list;
        long long nseg = 0;
        for (long long i = 0;; ++i) {
            // concat 里的相对路径按 list 文件所在目录解析，会变成 <parts>/<parts>/xxx，
            // 所以这里一律写绝对路径。
            const std::string seg = std::filesystem::absolute(seg_file(i)).string();
            std::ifstream probe(seg, std::ios::binary);
            if (!probe.good()) break;
            probe.close();
            std::string esc;
            for (char c : seg) {
                if (c == '\'') esc += "'\\''";
                else esc += c;
            }
            list += "file '" + esc + "'\n";
            ++nseg;
        }
        {
            std::ofstream lf(seg_list, std::ios::binary | std::ios::trunc);
            lf << list;
        }
        long long sum_pk = 0;                     // 总包数必须等于已处理帧数，否则有帧没落盘
        bool sum_ok = true;
        for (long long i = 0; i < nseg; ++i) {
            const int got = probe_packets(ffprobe, seg_file(i));
            if (got <= 0) {
                sum_ok = false;
                break;
            }
            sum_pk += got;
        }
        if (sum_ok && sum_pk != total) {
            sum_ok = false;
            std::printf("分段总帧数不符: 磁盘 %lld 帧，应为 %lld 帧\n", sum_pk, total);
        }
        if (!sum_ok) {
            std::printf("分段校验失败，已保留分段目录（同命令重跑会从坏段重来）: %s\n", seg_dir.c_str());
            ret = 1;
        } else {
        std::printf("\n拼接 %lld 个分段（共 %lld 帧）-> %s\n", nseg, sum_pk, out_path.c_str());
        const std::string cc = "\"" + ffmpeg + "\" -v error -y -f concat -safe 0 -i \"" + seg_list +
                               "\" -c copy \"" + out_path + "\"";
        if (!run_ok(cc)) {
            std::printf("拼接失败（目标容器可能不吃 -c copy？）分段已保留: %s\n", seg_dir.c_str());
            ret = 1;
        } else if (keep_parts) {
            std::printf("分段目录保留（--keep-parts）: %s\n", seg_dir.c_str());
        } else {
            std::error_code ec;
            std::filesystem::remove_all(seg_dir, ec);
        }
        }
    } else if (seg_mode) {
        std::printf("有分段失败，分段目录已保留（同命令重跑可从断点继续）: %s\n", seg_dir.c_str());
    }
    if (wq_err) {
        std::printf("写出失败: 编码器管道写不进去（磁盘满 / ffmpeg 中途退出？）\n");
        ret = 1;
    }
    if (total <= 0) {
        std::printf("失败: 没有解码到任何帧（输入损坏 / ffmpeg 不可用？）\n");
        ret = 1;
    }
    if (fin) _pclose(fin);
    if (fout) {
        // ffmpeg 自己失败（输出路径不可写、编码器拒绝参数）时这里非 0：
        // 以前无条件 return 0，用户脚本会以为成功
        const int st = pclose_w(fout);
        if (st != 0) {
            std::printf("编码器 ffmpeg 异常退出（返回值 %d），输出文件可能不完整\n", st);
            ret = 1;
        }
    }
    vkDeviceWaitIdle(g.dev);
    return ret;
}






