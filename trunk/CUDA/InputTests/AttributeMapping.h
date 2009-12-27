#pragma once

class AttributeMapping
{
	Str m_sColumnName;

	vector<Str> m_vecAttributeValues; // Only used with literal attribute
	double m_dMin; // Only used with non-literal attribute
	double m_dMax; // Only used with non-literal attribute
	
	bool m_bLiteralAttribute;
	bool m_bOutputAttribute;

	int m_iColumnInInputFile;
	int m_iFirstAttributeInStructure;


};
