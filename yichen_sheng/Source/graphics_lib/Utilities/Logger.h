#pragma once
#include <iostream>
#include <sstream>

//#include "global_variable.h"

class logger 
{
public:
	~logger() {};

	static logger* instance() {
		if (!m_instance)
			m_instance = new logger();

		return m_instance;
	}

	template<typename T, typename TT, typename TTT>
	void log(const std::string info, T file, TT line, TTT func) {
		std::stringstream oss;
		oss << info << std::endl;

		//if (global_variable::instance()->verbose)
			std::cerr << oss.str();

		m_log_str += oss.str();
	}

	template<typename INFO_T, typename FUNC>
	void info(const INFO_T s, FUNC func) {
		std::stringstream oss;
		oss << s << std::endl;

		//if (global_variable::instance()->verbose)
			std::cout << oss.str();

		m_log_str += oss.str();
	}

	template<typename INFO_T, typename FUNC>
	void warn(const INFO_T s, FUNC func) {
		std::stringstream oss;
		oss << s << std::endl;

		//if (global_variable::instance()->verbose)
			std::cerr << oss.str();

		m_log_str += oss.str();
	}

	std::string get_log() { return m_log_str; }

private:
	std::string m_log_str;
	static logger *m_instance;
};

#ifndef LOG
#define LOG(s) logger::instance()->log(s, __FILE__, __LINE__, __FUNCTION__)
#endif

#ifndef INFO
#define INFO(s) logger::instance()->info(s, __FUNCTION__)
#endif // !INFO


#ifndef WARN
#define WARN(s) logger::instance()->warn(s, __FUNCTION__)
#endif // !INFO
