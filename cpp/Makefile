TARGETS := finite init move

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cpp

clean:
	rm -rf $(TARGETS)
