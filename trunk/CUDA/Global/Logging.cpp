#include "stdafx.h"

FILE *Logging::m_pLoggingFile;
unsigned int g_uiAllowedTypesConsole = 0xFFFFFFFF;	// all logging types allowed
unsigned int g_uiAllowedTypesFile = 0xFFFFFFFF;		// all logging types allowed

void Logging::makeSureLoggingFileExists()
{
	if(m_pLoggingFile == NULL)
	{
		Str sLogFileName("LoggingCNL.txt");
		m_pLoggingFile = TiXmlFOpen(sLogFileName.c_str(),"a");
		if(m_pLoggingFile == NULL)
		{
			printf("Could not open log file %s, exiting\n", sLogFileName.c_str());
			exit(1);
		}

		// We put some text at the beginning of session
		fputs("\n",m_pLoggingFile);
		logTextFileLine(LT_INFORMATION,"Started session","============","============================",0);
	}
}

void Logging::logTextFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber)
{
	// we check if this logging type is logged either by file or console
	if(!((g_uiAllowedTypesConsole & (unsigned int)p_eLoggingType) 
		|| (g_uiAllowedTypesFile & (unsigned int)p_eLoggingType)))
		return;

	makeSureLoggingFileExists();

	static Str sLogging;
	Str sLoggingType;
	switch(p_eLoggingType)
	{
		case LT_INFORMATION:	sLoggingType = "INFORMATION"; break;
		case LT_WARNING:		sLoggingType = "WARNING    "; break;
		case LT_ERROR:			sLoggingType = "ERROR      "; break;
		case LT_DEBUG:			sLoggingType = "DEBUG      "; break;
		case LT_MEMORY:			sLoggingType = "MEMORY     "; break;
	}

	Str sFileName(p_sFileName);
	size_t iFound = sFileName.rfind('\\');
	if(iFound != -1)
		sFileName = sFileName.substring(iFound+1);
	else if((iFound = sFileName.rfind('/')) != -1)
		sFileName = sFileName.substring(iFound+1);

#ifdef _MSC_VER		// Identifies Microsoft compilers
	SYSTEMTIME now;
	GetLocalTime(&now);

	sLogging.format("%d.%02d.%02d %02d:%02d:%02d:%03d    %s%25s%50s%5d    %s\n",
		now.wYear,now.wMonth,now.wDay,
		now.wHour,now.wMinute,now.wSecond,now.wMilliseconds,
		sLoggingType.c_str(),sFileName.c_str(),p_sFunctionName,
		p_lLineNumber,p_sLoggingText);
#else
	timeval now1;
	gettimeofday(&now1, NULL);
	time_t now = now1.tv_sec;
	localtime(&now);

	sLogging.format("%d.%02d.%02d %02d:%02d:%02d:%03d    %s%25s%50s%5d    %s\n",
		now.tm_year+1900,now.tm_mon+1,now.tm_mday,
		now.tm_hour,now.tm_min,now.tm_sec,now1.tv_usec,
		sLoggingType.c_str(),sFileName.c_str(),p_sFunctionName,
		p_lLineNumber,p_sLoggingText);
#endif

	if(g_uiAllowedTypesFile & (unsigned int)p_eLoggingType)
	{
		fputs(sLogging.c_str(),m_pLoggingFile);
		fflush(m_pLoggingFile);
	}

	if(g_uiAllowedTypesConsole & (unsigned int)p_eLoggingType)
		printf("%s",sLogging.c_str());
}

void Logging::setAllowedLoggingTypes(unsigned int p_uiNewAllowedTypesConsole, unsigned int p_uiNewAllowedTypesFile)
{
	g_uiAllowedTypesConsole = p_uiNewAllowedTypesConsole;
	g_uiAllowedTypesFile = p_uiNewAllowedTypesFile;
}
