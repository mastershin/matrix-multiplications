#pragma once

#include <iostream>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

bool detect_neon();
void print_neon_features();