TARGETS := finite init move optional avx float asm

CXXFLAGS := -std=c++17 -mavx -O2

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

%.s: %.cpp
	$(CXX) $(CXXFLAGS) -S $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cpp

clean:
	rm -rf $(TARGETS)
