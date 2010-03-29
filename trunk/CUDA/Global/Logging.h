#pragma once

#define logText(a,b)																					\
	if( (g_uiAllowedTypesConsole & (unsigned int)a)														\
		|| (g_uiAllowedTypesFile & (unsigned int)a) )													\
	{																									\
		Logging::logTextFileLine(a,b,__FILE__,__FUNCTION__,__LINE__);									\
	}																									


#define logTextParams(a,b,...)																			\
	if( (g_uiAllowedTypesConsole & (unsigned int)a)														\
		|| (g_uiAllowedTypesFile & (unsigned int)a) )													\
	{																									\
		Logging::logTextFileLine(a,TiXmlString(b,__VA_ARGS__).c_str(),__FILE__,__FUNCTION__,__LINE__);	\
	}
		
		
// JRTODO
#define logAssert(a) if((int)(a) == 0) logText(Logging::LT_ERROR,Str("Assert failed: "+Str(#a)).c_str());

class Logging
{
public:
	enum LoggingType
	{	// all values must have different bits
		LT_INFORMATION	= 1 << 0,
		LT_WARNING		= 1 << 1,
		LT_ERROR		= 1 << 2,
		LT_DEBUG		= 1 << 3,
		LT_MEMORY		= 1 << 4
	};

	static void makeSureLoggingFileExists();

	static void logTextFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber);

	static void setAllowedLoggingTypes(unsigned int p_uiNewAllowedTypesConsole, unsigned int p_uiNewAllowedTypesFile);

	//static void logTextParamsFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber,...);

	
private:
	static FILE *m_pLoggingFile;
};

// these variables should be inside Logging class, but they are used by kernels in TrainNetwork.cu, so they have to be global
extern unsigned int g_uiAllowedTypesConsole;
extern unsigned int g_uiAllowedTypesFile;