#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

template <typename T>
class Queue {
  public:
    Queue() = default;

    void push(const T &value) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(value);
        condition_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        auto result = std::move(queue_.front());
        queue_.pop();
        return result;
    }

  private:
    std::mutex mtx_;
    std::condition_variable condition_;
    std::queue<T> queue_;
};

class ThreadPool {
  public:
    ThreadPool(size_t size) {
        workers_.reserve(size);
        for (size_t i = 0; i < size; i++) {
            workers_.emplace_back([this, i] { worker(i); });
        }
    }

    virtual ~ThreadPool() {
        for (size_t i = 0; i < workers_.size(); i++) {
            tasks_.push(std::nullopt);
        }

        for (auto &thread : workers_) {
            thread.join();
        }
    }

    template <typename F, typename... Args>
    std::future<std::invoke_result_t<F, Args...>> submit(F f, Args... args) {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();
        tasks_.push([task] { (*task)(); });
        return result;
    }

  private:
    void worker(size_t i) {
        while (true) {
            auto maybe_task = tasks_.pop();
            if (!maybe_task) {
                break;
            }
            auto task = *maybe_task;
            task();
        }
    }

  private:
    Queue<std::optional<std::function<void(void)>>> tasks_;
    std::vector<std::thread> workers_;
};

inline std::string get_time() {
    std::time_t t = std::time(nullptr);
    std::tm *tm = std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(tm, "[%H:%M:%S]");
    return oss.str();
}

int main() {
    const int num_tasks = 100;

    std::vector<std::future<std::string>> results;
    results.reserve(num_tasks);

    ThreadPool pool(8);
    for (int i = 0; i < num_tasks; i++) {
        auto result = pool.submit([i] {
            const int ms = rand() % 1000;
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));

            std::ostringstream oss;
            if (rand() % 2 == 0) {
                oss << get_time() << " task " << i << " failed within " << ms << " ms";
                throw std::runtime_error(oss.str());
            } else {
                oss << get_time() << " task " << i << " finished within " << ms << " ms";
            }
            return oss.str();
        });
        results.emplace_back(std::move(result));
    }

    for (auto &result : results) {
        // print results in submission order
        try {
            const std::string msg = result.get();
            std::cout << msg << std::endl;
        } catch (const std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }
    std::cout << get_time() << " all tasks completed" << std::endl;

    return 0;
}
