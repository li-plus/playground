TARGETS := finite init move optional avx

CXXFLAGS := -std=c++17 -mavx -O2

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cpp

clean:
	rm -rf $(TARGETS)
