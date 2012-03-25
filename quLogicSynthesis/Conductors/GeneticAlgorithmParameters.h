#pragma once
#pragma managed

namespace Conductor
{
  class GeneticAlgorithmParameters
  {
  public:
    int m_nGen;
    int m_nRun;
    double m_Pm;
    double m_Pc;
    int m_nCrossOver;  
    gcroot<StreamReader^> m_sr;

    GeneticAlgorithmParameters(void)
    {
      //String^ s = gcnew String(Config::HomeFolder.c_str()) + "GeneticAlgorithmParameterss.csv";
      //m_sr = gcnew StreamReader(s);
      //m_sr->ReadLine();  // Skip Header;
    }

    ~GeneticAlgorithmParameters(void)
    {
      //m_sr->Close();
      //delete m_sr;
    }

    bool NextGeneticAlgorithmParameters()
    {
      //String ^s;

      //if(m_sr->Peek() >= 0)
      //  s = m_sr->ReadLine();
      //else
      //  return false;

      //Console::WriteLine("Configuration: {0}", s);
      //array<String^>^ list = s->Split(',');

      //if (list->Length == 5) {
      //  // m_nGen,m_nRun,m_Pm,m_Pc
      //  m_nGen = Convert::ToUInt32(list[0]);
      //  m_nRun = Convert::ToUInt32(list[1]);
      //  m_Pm = Convert::ToDouble(list[2]);
      //  m_Pc = Convert::ToDouble(list[3]);
      //  m_nCrossOver = list[4] == "Single" ? 0 : 1;

      m_nCrossOver = 0;
      m_Pc = 0.7;
      m_Pm = 0.1;

        return true;
      //}
      //return false;
    }
  };
}