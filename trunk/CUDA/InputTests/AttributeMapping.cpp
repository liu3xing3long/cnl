#include "stdafx.h"

const Str m_XMLColumnName("ColumnName");
const Str m_XMLMinValue("MinValue");
const Str m_XMLMaxValue("MaxValue");
const Str m_XMLIsLiteralAttribute("IsLiteralAttribute");
const Str m_XMLTrue("True");
const Str m_XMLFalse("False");
const Str m_XMLIsOutputAttribute("IsOutputAttribute");
const Str m_XMLColumnIndexInInputFile("ColumnIndexInInputFile");
const Str m_XMLColumnIndexInStructure("ColumnIndexInStructure");
const Str m_XMLColumnElementName("ColumnElementName");

Str AttributeMapping::getColumnName() const
{
	return m_sColumnName;
}

bool AttributeMapping::isOutputAttribute() const
{
	return m_bOutputAttribute;
}

AttributeMapping::AttributeMapping(Str p_sColumnName,bool p_bIsOutputVector, int p_iColumnIndexInVecElements,int p_iElementInStructure)
{
	m_sColumnName = p_sColumnName;
	m_bOutputAttribute = p_bIsOutputVector;
	m_iColumnInInputFile = p_iColumnIndexInVecElements;
	m_iFirstAttributeInStructure = p_iElementInStructure;
	m_dMin = m_dMax = numeric_limits<double>::quiet_NaN(); // will be filled later (in case of non-literal attribute)
	m_bLiteralAttribute = false;
}

void AttributeMapping::saveToXML(TiXmlElement &p_XML) const
{
	// We save neuron type
	p_XML.SetAttribute(m_XMLColumnName.c_str(),m_sColumnName.c_str());
	Str sTemp = (m_bOutputAttribute ? m_XMLTrue : m_XMLFalse);
	p_XML.SetAttribute(m_XMLIsOutputAttribute.c_str(),sTemp.c_str());
	p_XML.SetAttribute(m_XMLColumnIndexInInputFile.c_str(),Str("%d",m_iColumnInInputFile).c_str());
	p_XML.SetAttribute(m_XMLColumnIndexInStructure.c_str(),Str("%d",m_iFirstAttributeInStructure).c_str());
	sTemp = (m_bLiteralAttribute ? m_XMLTrue : m_XMLFalse);
	p_XML.SetAttribute(m_XMLIsLiteralAttribute.c_str(),sTemp.c_str());
	if(m_bLiteralAttribute == true)
	{
		for(unsigned uAttributeValueIndex = 0;uAttributeValueIndex < m_vecAttributeValues.size();++uAttributeValueIndex)
		{
			TiXmlElement elementToXML(m_XMLColumnElementName.c_str());
			TiXmlText valueToSave(m_vecAttributeValues[uAttributeValueIndex].c_str());
			elementToXML.InsertEndChild(valueToSave);
			p_XML.InsertEndChild(elementToXML);
		}
	}
	else
	{
		p_XML.SetAttribute(m_XMLMinValue.c_str(),Str("%lf",m_dMin).c_str());
		p_XML.SetAttribute(m_XMLMaxValue.c_str(),Str("%lf",m_dMax).c_str());
	}
}

void AttributeMapping::loadFromXML(const TiXmlElement &p_XML)
{
	m_sColumnName = p_XML.Attribute(m_XMLColumnName.c_str());
	m_bOutputAttribute = p_XML.Attribute(m_XMLIsOutputAttribute.c_str()) == m_XMLTrue;
	m_iColumnInInputFile = atoi(p_XML.Attribute(m_XMLColumnIndexInInputFile.c_str()));
	m_iFirstAttributeInStructure = atoi(p_XML.Attribute(m_XMLColumnIndexInStructure.c_str()));

	Str sIsLiteralAttribute = p_XML.Attribute(m_XMLIsLiteralAttribute.c_str());
	m_bLiteralAttribute = (sIsLiteralAttribute == m_XMLTrue);
	if(m_bLiteralAttribute == true)
	{
		const TiXmlElement *pXMLColumnElementName = p_XML.FirstChildElement(m_XMLColumnElementName.c_str());
		while(pXMLColumnElementName)
		{
			m_vecAttributeValues.push_back(pXMLColumnElementName->GetText());
			pXMLColumnElementName = pXMLColumnElementName->NextSiblingElement(m_XMLColumnElementName.c_str());
		}
	}
	else
	{
		m_dMin = atof(p_XML.Attribute(m_XMLMinValue.c_str()));
		m_dMax = atof(p_XML.Attribute(m_XMLMaxValue.c_str()));
	}
}

void AttributeMapping::setLiteralPossibleValues(const vector<Str> &p_vecPosibleValues)
{
	m_vecAttributeValues = p_vecPosibleValues;
	m_bLiteralAttribute = true;
}

void AttributeMapping::setPossibleRange(double p_dMin,double p_dMax)
{
	if((!_isnan(m_dMin) && p_dMin != m_dMin) || (!_isnan(m_dMax) && p_dMax != m_dMax))
		logTextParams(Logging::LT_ERROR,"Different min/max values. Before %lf/%lf , after %lf/%lf",m_dMin,m_dMax,p_dMin,p_dMax);

	m_dMin = p_dMin;
	m_dMax = p_dMax;
	m_bLiteralAttribute = false;
}

int AttributeMapping::getFirstAttributeInStructure() const
{
	return m_iFirstAttributeInStructure;
}

bool AttributeMapping::isLiteralAttribute() const
{
	return m_bLiteralAttribute;
}

unsigned AttributeMapping::getAttributeValuesCount() const
{
	return m_vecAttributeValues.size();
}

Str AttributeMapping::getAttributeValue(unsigned p_uIndex) const
{
	return m_vecAttributeValues[p_uIndex];
}

double AttributeMapping::getMinValue() const
{
	return m_dMin;
}

double AttributeMapping::getMaxValue() const
{
	return m_dMax;
}
