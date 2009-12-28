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

public:
	Str getColumnName() const;
	bool isOutputAttribute() const;

	AttributeMapping(Str p_sColumnName = "",bool p_bIsOutputVector = false
		,int p_iColumnIndexInVecElements = -1,int p_iElementInStructure = -1);

	void saveToXML(TiXmlElement &p_XML) const;
	void loadFromXML(const TiXmlElement &p_XML);
	void setLiteralPossibleValues(const vector<Str> &p_vecPosibleValues);
	void setPossibleRange(double p_dMin,double p_dMax);
};
