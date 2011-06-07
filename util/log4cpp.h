#ifndef LOG4CPP_H
#define LOG4CPP_H

#include <iostream>
#include <cstdarg>
#include <typeinfo>
#include <string>
#include <sstream>

namespace log4cpp 
{
    using std::ostream;
    using std::string;
    using std::cerr;
    using std::endl;

    inline string file_path_basename(const char *fp) 
    {
        string fps(fp);
        size_t pos = fps.rfind('/');
        if (pos == string::npos) {
            return fp;
        } else {
            return fps.substr(pos + 1);
        }
    }
    
    
    class logger 
    {
    public:	
        ostream* os;
        bool enabled;
        bool tracing_enabled;

        logger(ostream* os=(&cerr))
                : os(os), enabled(true), tracing_enabled(true) {};

        void log_header(const string& error_level, const char *file,
                        const char *func, int line) const {
            if (enabled) {
                (*os) << "[" << error_level << ": " << func << " @ "
                      << file_path_basename(file) << "#" << line << "] ";
            }
        }

        void log_error(const char *file, const char *func, int line,
                       const string &s) const {
            if (enabled) {
                log_header("ERROR", file, func, line);
                (*os) << s << endl;
            }
        }

        void log_warning(const char *file, const char *func, int line,
                         const string &s) const {
            if (enabled) {
                log_header("WARNING", file, func, line);
                (*os) << s << endl;
            }
        }

        void log_info(const char *file, const char *func, int line,
                      const string &s) const {
            if (enabled) {
                log_header("INFO", file, func, line);
                (*os) << s << endl;
            }
        }

        void log_debug(const char *file, const char *func, int line,
                       const string &s) const {
            if (enabled) {
                log_header("DEBUG", file, func, line);
                (*os) << s << endl;
            }
        }

        void func_call(const char *file, const char *func,
                       int line) const {
            if (tracing_enabled) {
                log_header("CALL", file, func, line);
                (*os) << endl;
            }
        }

        void func_return(const char *file, const char *func,
                         int line) const {
            if (tracing_enabled) {
                log_header("RETURN", file, func, line);
                (*os) << endl;
            }
        }
    };

    extern const logger cerr_logger;
    //extern const logger cout_logger;    
}



#define LOG_LEVEL_FATAL 0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_INFO 3
#define LOG_LEVEL_DEBUG 4
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_DEBUG
#endif

#define TRACE 1


#if LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(arg)                                                  \
	do {                                                                \
        ostringstream ss;                                               \
        ss << arg;                                                      \
        log4cpp::cerr_logger.log_error(__FILE__, __FUNCTION__, __LINE__ , ss.str()); \
    } while (0);
#else
#define LOG_ERROR(arg) do {} while(0) /* do nothing */
#endif /* LOG_LEVEL >= LOG_LEVEL_ERROR */

#if LOG_LEVEL >= LOG_LEVEL_WARNING
#define LOG_WARNING(arg)                                                \
	do {                                                                \
        ostringstream ss;                                               \
        ss << arg;                                                      \
        log4cpp::cerr_logger.log_warning(__FILE__, __FUNCTION__, __LINE__ , ss.str()); \
    } while (0);
#else
#define LOG_WARNING(arg) do {} while(0) /* do nothing */
#endif /* LOG_LEVEL >= LOG_LEVEL_WARNING */

#if LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(arg)                                                  \
	do {                                                                \
        ostringstream ss;                                               \
        ss << arg;                                                      \
        log4cpp::cerr_logger.log_info(__FILE__, __FUNCTION__, __LINE__ , ss.str()); \
    } while (0);
#else
#define LOG_INFO(arg) do {} while(0) /* do nothing */
#endif /* LOG_LEVEL >= LOG_LEVEL_INFO */

#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(arg)                                                  \
	do {                                                                \
        ostringstream ss;                                               \
        ss << arg;                                                      \
        log4cpp::cerr_logger.log_debug(__FILE__, __FUNCTION__, __LINE__ , ss.str()); \
    } while (0);
#else
#define LOG_DEBUG(...) do {} while(0) /* do nothing */
#endif /* LOG_LEVEL >= LOG_LEVEL_DEBUG */
    
#ifdef TRACE
#define IS_TRACE_ON 1
#define TRACE_START \
	log4cpp::cerr_logger.func_call(__FILE__,  __FUNCTION__, __LINE__ )
#define TRACE_END \
	log4cpp::cerr_logger.func_return(__FILE__,  __FUNCTION__, __LINE__ )
#else /* TRACE */
#define IS_TRACE_ON 0
#define TRACE_START do {} while(0)
#define TRACE_END do {} while(0)
#endif /* TRACE */

#define RETURN_TRACING(x) do {TRACE_END; return x;} while (0)
#define RETURN_TRACING0 do {TRACE_END; return;} while (0)
	
#endif /* LOG4CPP_H */
