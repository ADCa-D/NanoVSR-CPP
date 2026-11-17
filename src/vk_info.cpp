// vk_info.cpp -- 打印设备能力，决定下一步走 fp16 还是 cooperative matrix（张量核）。
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{};
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    for (auto pd : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pd, &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        std::printf("=== %s  (API %u.%u.%u) ===\n", p.deviceName,
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion));
        std::printf("driver: %u.%u.%u   vendor 0x%04x  device 0x%04x\n",
                    VK_API_VERSION_MAJOR(p.driverVersion), VK_API_VERSION_MINOR(p.driverVersion),
                    VK_API_VERSION_PATCH(p.driverVersion), p.vendorID, p.deviceID);
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(pd, &p2);
        std::printf("subgroup %u  maxSharedMem/blk %u B  maxWG %u,%u,%u  maxWGcount %u\n",
                    sg.subgroupSize, p.limits.maxComputeSharedMemorySize,
                    p.limits.maxComputeWorkGroupSize[0], p.limits.maxComputeWorkGroupSize[1],
                    p.limits.maxComputeWorkGroupSize[2],
                    p.limits.maxComputeWorkGroupInvocations);

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            std::printf("heap %u: %.1f MiB  %s\n", i, mp.memoryHeaps[i].size / 1048576.0,
                        (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL" : "host");
        }
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            std::printf("  type %u: flags 0x%03x  heap %u\n", i, mp.memoryTypes[i].propertyFlags,
                        mp.memoryTypes[i].heapIndex);
        }

        uint32_t nc = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nc, nullptr);
        std::vector<VkExtensionProperties> ex(nc);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &nc, ex.data());
        const char* want[] = {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_matrix",
                              "VK_KHR_cooperative_matrix2",
                              "VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage",
                              "VK_KHR_8bit_storage", "VK_KHR_shader_float_controls",
                              "VK_KHR_shader_subgroup_uniform_control_flow",
                              "VK_KHR_maintenance4", "VK_KHR_maintenance5", "VK_KHR_maintenance6",
                              "VK_EXT_shader_atomic_float", "VK_KHR_buffer_device_address"};
        std::printf("扩展:\n");
        for (const char* w : want) {
            bool found = false;
            for (auto& e : ex) {
                if (std::strcmp(e.extensionName, w) == 0) { found = true; break; }
            }
            std::printf("  %-46s %s\n", w, found ? "YES" : "-");
        }
        std::printf("(共 %u 个扩展)\n", nc);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
