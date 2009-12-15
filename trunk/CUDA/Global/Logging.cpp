#include "stdafx.h"

FILE *Logging::m_pLoggingFile;

void Logging::makeSureLoggingFileExists()
{
	if(m_pLoggingFile == NULL)
	{
		Str sLogFileName("LogXXX.txt");
		m_pLoggingFile = TiXmlFOpen(sLogFileName.c_str(),"a");
		if(m_pLoggingFile == NULL)
			throw exception(Str("Cannot open log file: %s",sLogFileName.c_str()).c_str());

		// We put some text at the beginning of session
		fputs("\n",m_pLoggingFile);
		logTextFileLine(LT_INFORMATION,"Started session","============","============================",0);
	}
}

void Logging::logTextFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber)
{
	makeSureLoggingFileExists();

#ifndef PRINT_MEMORY
	if(p_eLoggingType == LT_MEMORY)
		return;
#endif

#ifndef PRINT_DEBUG
	if(p_eLoggingType == LT_DEBUG)
		return;
#endif

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
	int iFound = sFileName.rfind('\\');
	if(iFound != -1)
		sFileName = sFileName.substring(iFound+1);
	else if((iFound = sFileName.rfind('/')) != -1)
		sFileName = sFileName.substring(iFound+1);

	time_t rawTime;
	time ( &rawTime );
	tm *pTimeStruct = localtime(&rawTime);

	sLogging.format("%d.%02d.%02d %02d:%02d:%02d    %s%20s%40s%5d    %s\n",
		pTimeStruct->tm_year+1900,pTimeStruct->tm_mon+1,pTimeStruct->tm_mday,
		pTimeStruct->tm_hour,pTimeStruct->tm_min,pTimeStruct->tm_sec,
		sLoggingType.c_str(),
		sFileName.c_str(),p_sFunctionName,p_lLineNumber,
		p_sLoggingText);

	fputs(sLogging.c_str(),m_pLoggingFile);	
	printf("%s",sLogging.c_str());
	fflush(m_pLoggingFile);
}
