#pragma once

#define logText(a,b) Logging::logTextFileLine(a,b,__FILE__,__FUNCTION__,__LINE__)
#define logTextParams(a,b,...) Logging::logTextFileLine(a,TiXmlString(b,__VA_ARGS__),__FILE__,__FUNCTION__,__LINE__)

// JRTODO
#define logAssert(a) if((int)(a) == 0) logText(Logging::LT_ERROR,Str("Assert failed: "+Str(#a)).c_str());

class Logging
{
public:
	enum LoggingType
	{
		LT_INFORMATION,
		LT_WARNING,
		LT_ERROR,
		LT_DEBUG,
		LT_MEMORY
	};

	static void makeSureLoggingFileExists();

	static void logTextFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber);

	//static void logTextParamsFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber,...);

private:
	static FILE *m_pLoggingFile;
};
