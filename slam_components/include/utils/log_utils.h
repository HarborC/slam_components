#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <memory>
#include <vector>

#include <glog/logging.h>

#define GLOG_TEMPLATE(level, ...) LOG(level) << std::fixed << __VA_ARGS__;

#define GINFO(...) GLOG_TEMPLATE(INFO, __VA_ARGS__)
#define GWARN(...) GLOG_TEMPLATE(WARNING, __VA_ARGS__)
#define GERROR(...) GLOG_TEMPLATE(ERROR, __VA_ARGS__)
#define GFATAL(...) GLOG_TEMPLATE(FATAL, __VA_ARGS__)

class TimeTicToc {
public:
  TimeTicToc() { tic(); }

  void tic() { start = std::chrono::system_clock::now(); }

  double toc() {
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() * 1000;
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};

class TimeStatistics {
public:
  using Ptr = std::shared_ptr<TimeStatistics>;

public:
  TimeStatistics() {
    timer.tic();
    timer_all.tic();

    time_cost.reserve(10);
    time_descri.reserve(10);
  }

  void tic() { timer.tic(); }

  void reStart() {
    timer.tic();
    timer_all.tic();

    time_cost.clear();
    time_descri.clear();
  }

  void tocAndTic(std::string descri) {
    GINFO(descri);

    double t = timer.toc();
    timer.tic();

    time_cost.push_back(t);
    time_descri.push_back(descri);
  }

  double logTimeStatistics(double time_now = 0) {
    double t_all = timer_all.toc();

    time_cost.push_back(t_all);
    time_descri.push_back("all");

    auto &&log = COMPACT_GOOGLE_LOG_INFO;
    log.stream() << "TimeSummary:" << std::fixed << std::setprecision(6)
                 << time_now << ";";
    for (size_t i = 0; i < time_cost.size(); i++) {
      log.stream() << time_descri.at(i) << ":" << time_cost.at(i) << " ms;";
    }
    log.stream() << "\n";

    return t_all;
  }

public:
  std::vector<double> time_cost;
  std::vector<std::string> time_descri;

  TimeTicToc timer;
  TimeTicToc timer_all;
};
