#include "rmm/detail/format.hpp"

#include <array>
#include <sstream>
#include <string>

std::string rmm::detail::format_bytes(std::size_t value)
{
  static std::array units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

  int index = 0;
  auto size = static_cast<double>(value);
  while (size > 1024) {
    size /= 1024;
    index++;
  }

  return std::to_string(size) + ' ' + units.at(index);
}
std::string rmm::detail::format_stream(rmm::cuda_stream_view stream)
{
  std::stringstream sstr{};
  sstr << std::hex << stream.value();
  return sstr.str();
}