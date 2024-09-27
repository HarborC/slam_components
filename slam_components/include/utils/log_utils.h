#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

class TimeTicToc {
public:
  TimeTicToc();

  void tic();

  double toc();

private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};

class TimeStatistics {
public:
  using Ptr = std::shared_ptr<TimeStatistics>;

public:
  TimeStatistics(std::string node_name = "Empty");

  void tic();

  void reStart();

  void tocAndTic(std::string descri);

  double logTimeStatistics(double time_now = 0);

public:
  std::vector<double> time_cost;
  std::vector<std::string> time_descri;

  TimeTicToc timer;
  TimeTicToc timer_all;

  std::string node_description;
};

std::string getNowDateTime();

void initSpdlog(std::string node_name, std::string &log_path,
                bool alsologtostderr = false);