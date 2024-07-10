# Define variables
CXX = g++
CXXFLAGS = --std=c++17
TARGET = matmul_cpu_neon.ex
SOURCES = main.cpp neon.cpp matmul_cpu_neon.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
.PHONY: all
all: build

# Build target
.PHONY: build
build: $(TARGET)

$(TARGET): $(OBJECTS)
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

# Phony targets to ensure they are always executed
.PHONY: build clean run

