# Compiler and flags
CXX = g++
CXXFLAGS = --std=c++17

# Flags for debug and release builds
CXXFLAGS_DEBUG = -g -O0
CXXFLAGS_RELEASE = -O2

# Default build type (can be overridden by command line)
BUILD_TYPE ?= debug

# Set CXXFLAGS based on the build type
ifeq ($(BUILD_TYPE), release)
    CXXFLAGS += $(CXXFLAGS_RELEASE)
    BUILD_MODE = "Release"
else
    CXXFLAGS += $(CXXFLAGS_DEBUG)
    BUILD_MODE = "Debug"
endif

# Target and source files
TARGET = matmul_cpu.ex
SOURCES = main.cpp matmul_cpu.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
.PHONY: all
all: build

# Build target
.PHONY: build
build: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Building $(TARGET) in $(BUILD_MODE) mode..."
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
.PHONY: clean
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run target
.PHONY: run
run: build
	./$(TARGET) $(ARGS)

# Add a release target
.PHONY: release
release:
	$(MAKE) BUILD_TYPE=release

# Add a debug target
.PHONY: debug
debug:
	$(MAKE) BUILD_TYPE=debug

# Phony targets to ensure they are always executed
.PHONY: all build clean run release debug
