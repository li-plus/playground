TARGETS := finite init

all: $(TARGETS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -rf $(TARGETS)
