#include "rmm/mr/device/per_device_resource.hpp"
rmm::mr::device_memory_resource* rmm::mr::detail::initial_resource()
{
  static cuda_memory_resource mr{};
  return &mr;
}
std::mutex& rmm::mr::detail::map_lock()
{
  static std::mutex map_lock;
  return map_lock;
}
std::mutex& rmm::mr::detail::ref_map_lock()
{
  static std::mutex ref_map_lock;
  return ref_map_lock;
}
auto rmm::mr::detail::get_ref_map() -> std::map<cuda_device_id::value_type, device_async_resource_ref>&
{
  static std::map<cuda_device_id::value_type, device_async_resource_ref> device_id_to_resource_ref;
  return device_id_to_resource_ref;
}
auto rmm::mr::detail::get_map() -> std::map<cuda_device_id::value_type, device_memory_resource*>&
{
  static std::map<cuda_device_id::value_type, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

rmm::mr::device_memory_resource* rmm::mr::get_per_device_resource(cuda_device_id device_id)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};
  auto& map = detail::get_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id.value());
  return (found == map.end()) ? (map[device_id.value()] = detail::initial_resource())
                              : found->second;
}
rmm::mr::device_memory_resource* rmm::mr::set_per_device_resource(cuda_device_id device_id,
                                                                  device_memory_resource* new_mr)
{
  std::lock_guard<std::mutex> lock{detail::map_lock()};

  // Note: even though set_per_device_resource() and set_per_device_resource_ref() are not
  // interchangeable, we call the latter from the former to maintain resource_ref
  // state consistent with the resource pointer state. This is necessary because the
  // Python API still uses the raw pointer API. Once the Python API is updated to use
  // resource_ref, this call can be removed.
  detail::set_per_device_resource_ref_unsafe(device_id, new_mr);

  auto& map          = detail::get_map();
  auto const old_itr = map.find(device_id.value());
  // If a resource didn't previously exist for `id`, return pointer to initial_resource
  auto* old_mr           = (old_itr == map.end()) ? detail::initial_resource() : old_itr->second;
  map[device_id.value()] = (new_mr == nullptr) ? detail::initial_resource() : new_mr;
  return old_mr;
}
rmm::mr::device_memory_resource* rmm::mr::get_current_device_resource()
{
  return get_per_device_resource(rmm::get_current_cuda_device());
}
rmm::mr::device_memory_resource* rmm::mr::set_current_device_resource(
  device_memory_resource* new_mr)
{
  return set_per_device_resource(rmm::get_current_cuda_device(), new_mr);
}
rmm::device_async_resource_ref rmm::mr::get_per_device_resource_ref(cuda_device_id device_id)
{
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  auto& map = detail::get_ref_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id.value());
  if (found == map.end()) {
    auto item = map.insert({device_id.value(), detail::initial_resource()});
    return item.first->second;
  }
  return found->second;
}
rmm::device_async_resource_ref rmm::mr::set_per_device_resource_ref(
  cuda_device_id device_id, device_async_resource_ref new_resource_ref)
{
  std::lock_guard<std::mutex> lock{detail::ref_map_lock()};
  return detail::set_per_device_resource_ref_unsafe(device_id, new_resource_ref);
}
rmm::device_async_resource_ref rmm::mr::get_current_device_resource_ref()
{
  return get_per_device_resource_ref(rmm::get_current_cuda_device());
}
rmm::device_async_resource_ref rmm::mr::set_current_device_resource_ref(
  device_async_resource_ref new_resource_ref)
{
  return set_per_device_resource_ref(rmm::get_current_cuda_device(), new_resource_ref);
}
rmm::device_async_resource_ref rmm::mr::reset_per_device_resource_ref(cuda_device_id device_id)
{
  return set_per_device_resource_ref(device_id, detail::initial_resource());
}
rmm::device_async_resource_ref rmm::mr::reset_current_device_resource_ref()
{
  return reset_per_device_resource_ref(rmm::get_current_cuda_device());
}
rmm::device_async_resource_ref rmm::mr::detail::set_per_device_resource_ref_unsafe(
  cuda_device_id device_id, device_async_resource_ref new_resource_ref)
{
  auto& map          = detail::get_ref_map();
  auto const old_itr = map.find(device_id.value());
  // If a resource didn't previously exist for `device_id`, return pointer to initial_resource
  // Note: because resource_ref is not default-constructible, we can't use std::map::operator[]
  if (old_itr == map.end()) {
    map.insert({device_id.value(), new_resource_ref});
    return device_async_resource_ref{detail::initial_resource()};
  }

  auto old_resource_ref = old_itr->second;
  old_itr->second       = new_resource_ref;  // update map directly via iterator
  return old_resource_ref;
}