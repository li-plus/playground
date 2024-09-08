TARGETS := finite init move optional avx float asm

CXXFLAGS := -std=c++17 -mavx -O2

TARGET_DIR := ./build

TARGETS := $(TARGETS:%=$(TARGET_DIR)/%)

all: $(TARGETS)

$(TARGET_DIR):
	mkdir -p $@

$(TARGET_DIR)/%: %.cpp $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(TARGET_DIR)/%.s: %.cpp $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) -S $< -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cpp

clean:
	rm -rf $(TARGET_DIR)
