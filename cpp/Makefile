TARGETS := finite init move optional

CXXFLAGS := -std=c++17

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cpp

clean:
	rm -rf $(TARGETS)
