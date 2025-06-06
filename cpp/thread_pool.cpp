#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
  public:
    ThreadPool(size_t size) : shutdown_(false) {
        workers_.reserve(size);
        for (size_t i = 0; i < size; i++) {
            workers_.emplace_back([this, i] { worker(i); });
        }
    }

    virtual ~ThreadPool() {
        shutdown_ = true;
        cv_.notify_all();

        for (auto &thread : workers_) {
            thread.join();
        }
    }

    template <typename F, typename... Args>
    std::future<typename std::invoke_result<F, Args...>::type> submit(F f, Args... args) {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();
        {
            std::lock_guard<std::mutex> lk(m_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

  private:
    void worker(size_t i) {
        while (true) {
            std::function<void(void)> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return !tasks_.empty() || shutdown_; });
                if (tasks_.empty()) {
                    break;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

  private:
    bool shutdown_;
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<std::function<void(void)>> tasks_;
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

    std::vector<std::future<void>> results;
    results.reserve(num_tasks);

    ThreadPool pool(8);
    for (int i = 0; i < num_tasks; i++) {
        auto result = pool.submit([i] {
            int ms = rand() % 1000;
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));

            std::cout << get_time() << " task " << i << " finished within " << ms << " ms" << std::endl;
        });
        results.emplace_back(std::move(result));
    }

    for (auto &result : results) {
        result.get();
    }
    std::cout << get_time() << " all tasks completed" << std::endl;

    return 0;
}
