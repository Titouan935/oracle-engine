#pragma once
#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

enum class LogLevel { DEBUG, INFO, WARNING, LERROR, FATAL };

class Logger {
public:
    using ErrCb = std::function<void(LogLevel, const std::string&, const std::string&)>;

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    void set_error_callback(ErrCb cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        err_cb_ = std::move(cb);
    }

    void log(LogLevel level, const std::string& module, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string line = format(level, module, msg);
        std::cout << line << "\n";
        std::cout.flush();
        if (file_.is_open()) { file_ << line << "\n"; file_.flush(); }
        if (level == LogLevel::FATAL)
            std::cerr << "[FATAL] " << msg << "\n";
        // Notifie le BugAgent pour les erreurs et fatales
        if (err_cb_ && (level == LogLevel::LERROR || level == LogLevel::FATAL))
            err_cb_(level, module, msg);
    }

    void open(const std::string& path) {
        file_.open(path, std::ios::app);
    }

private:
    Logger() = default;
    std::ofstream file_;
    std::mutex    mutex_;
    ErrCb         err_cb_;

    std::string format(LogLevel level, const std::string& module, const std::string& msg) {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream ss;
        ss << std::put_time(std::localtime(&t), "%H:%M:%S");
        ss << " [" << levelStr(level) << "] [" << module << "] " << msg;
        return ss.str();
    }

    const char* levelStr(LogLevel l) {
        switch(l) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO ";
            case LogLevel::WARNING: return "WARN ";
            case LogLevel::LERROR:  return "ERROR";
            case LogLevel::FATAL:   return "FATAL";
        }
        return "?";
    }
};

#define LOG_INFO(mod, msg)  Logger::get().log(LogLevel::INFO,    mod, msg)
#define LOG_WARN(mod, msg)  Logger::get().log(LogLevel::WARNING, mod, msg)
#define LOG_ERR(mod, msg)   Logger::get().log(LogLevel::LERROR,  mod, msg)
#define LOG_DEBUG(mod, msg) Logger::get().log(LogLevel::DEBUG,   mod, msg)
#define LOG_FATAL(mod, msg) Logger::get().log(LogLevel::FATAL,   mod, msg)
