#include "utils/log_utils.h"
#include "utils/io_utils.h"

TimeTicToc::TimeTicToc() { tic(); }

void TimeTicToc::tic() { start = std::chrono::system_clock::now(); }

double TimeTicToc::toc() {
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  return elapsed_seconds.count() * 1000;
}

TimeStatistics::TimeStatistics() {
  timer.tic();
  timer_all.tic();

  time_cost.reserve(10);
  time_descri.reserve(10);
}

void TimeStatistics::tic() { timer.tic(); }

void TimeStatistics::reStart() {
  timer.tic();
  timer_all.tic();

  time_cost.clear();
  time_descri.clear();
}

void TimeStatistics::tocAndTic(std::string descri) {
  SPDLOG_INFO("{}", descri);

  double t = timer.toc();
  timer.tic();

  time_cost.push_back(t);
  time_descri.push_back(descri);
}

double TimeStatistics::logTimeStatistics(double time_now) {
  double t_all = timer_all.toc();

  time_cost.push_back(t_all);
  time_descri.push_back("all");

  std::string log_msg = fmt::format("TimeSummary:{:.6f};", time_now);

  for (size_t i = 0; i < time_cost.size(); i++) {
    log_msg += fmt::format("{}:{:.6f} ms;", time_descri.at(i), time_cost.at(i));
  }

  SPDLOG_INFO("{}", log_msg);

  return t_all;
}

std::string getNowDateTime() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::tm *local_tm = std::localtime(&in_time_t);
  local_tm->tm_hour += 8;
  std::mktime(local_tm);

  return std::to_string(local_tm->tm_year + 1900) + "-" +
         std::to_string(local_tm->tm_mon + 1) + "-" +
         std::to_string(local_tm->tm_mday) + "-" +
         std::to_string(local_tm->tm_hour) + "-" +
         std::to_string(local_tm->tm_min) + "-" +
         std::to_string(local_tm->tm_sec);
}

void initSpdlog(std::string node_name, std::string &log_path,
                bool alsologtostderr) {
  // 创建日志目录路径
  log_path = log_path + "/" + getNowDateTime();
  Utils::CreateRecursiveDirIfNotExists(log_path);

  // 创建文件日志 sink
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      log_path + "/log_info.log", true);

  // 设置控制台输出 sink（带有颜色的输出）
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // 如果需要将日志同时输出到 stderr
  if (alsologtostderr) {
    console_sink->set_level(
        spdlog::level::info); // 输出 INFO 及以上级别日志到控制台
  } else {
    console_sink->set_level(spdlog::level::off); // 禁止控制台输出
  }

  // 设置文件输出的日志级别
  file_sink->set_level(spdlog::level::info); // 记录 INFO 及以上级别日志到文件

  // 创建一个 logger，将多个 sink 组合在一起
  spdlog::sinks_init_list sink_list = {file_sink, console_sink};
  auto logger = std::make_shared<spdlog::logger>(node_name, sink_list.begin(),
                                                 sink_list.end());

  // 设置日志格式
  logger->set_pattern("[%H:%M:%S.%e][%l][%g:%# %!] %v");

  // 注册 logger
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);

  // 设置日志级别
  spdlog::set_level(spdlog::level::info); // 默认级别为 INFO
  spdlog::flush_on(spdlog::level::info); // 每条日志输出后立即刷新到文件
}
