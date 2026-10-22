// Persistent thread pool for the -threads builds (PLAN.md §6). Created once in a
// workload's init export, so thread startup is never timed. Under Emscripten the
// pthreads come from the pre-spawned PTHREAD_POOL_SIZE Web Workers; `threads` must
// not exceed that, or the constructor waits on a worker that needs the event loop.
//
// run(fn) calls fn(t) for every t in [0, size()): t = 0 on the calling thread,
// the rest on the pool's threads. It returns when all of them have finished.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  /** `threads` participants in total, including the caller of run(). */
  explicit ThreadPool(int threads) {
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this, t] { loop(t); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  template <class F>
  void run(F& fn) {
    run_task([](void* ctx, int t) { (*static_cast<F*>(ctx))(t); }, &fn);
  }

 private:
  using Task = void (*)(void* ctx, int t);

  void run_task(Task task, void* ctx) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = task;
      ctx_ = ctx;
      pending_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    start_.notify_all();
    task(ctx, 0);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
  }

  void loop(int t) {
    std::uint64_t seen = 0;
    for (;;) {
      Task task;
      void* ctx;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        task = task_;
        ctx = ctx_;
      }
      task(ctx, t);
      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  Task task_ = nullptr;
  void* ctx_ = nullptr;
  int pending_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
};
