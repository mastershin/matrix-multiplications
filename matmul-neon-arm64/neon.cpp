#include "neon.h"

bool detect_neon()
{
#if defined(__APPLE__) && defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

void print_neon_features()
{
#if defined(__APPLE__) && defined(__aarch64__)
  std::cout << "NEON is supported.\n";
#else
  std::cout << "NEON capability check is not supported on this platform.\n";
#endif
}
