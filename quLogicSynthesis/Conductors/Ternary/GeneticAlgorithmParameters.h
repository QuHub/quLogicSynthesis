#pragma once
#pragma managed

namespace Conductor
{
  class GeneticAlgorithmParameters
  {
  public:
    int m_nGenerations;
    int m_nRuns;
    double m_Pm;
    double m_Pc;
    int m_nCrossOver;  
    gcroot<StreamReader^> m_sr;

    GeneticAlgorithmParameters(void)
    {
      m_sr = gcnew StreamReader("GAParams.csv");
      m_sr->ReadLine();  // Skip Header;
    }

    ~GeneticAlgorithmParameters(void)
    {
      m_sr->Close();
      delete m_sr;
    }

    String^ GeneticParametersForDisplay()
    {
      return String::Format("#Generations: {0}, #Runs: {1}, Pm: {2}, Pc: {3}, Crossover: {4}", m_nGenerations, m_nRuns, m_Pm, m_Pc, m_nCrossOver);
    }

    bool NextGeneticAlgorithmParameters()
    {
      String ^s;

      if(m_sr->Peek() >= 0)
        s = m_sr->ReadLine();
      else
        return false;

      Console::WriteLine("Configuration: {0}", s);
      array<String^>^ list = s->Split(',');

      if (list->Length == 5) {
        // m_nGen,m_nRun,m_Pm,m_Pc
        m_nGenerations = Convert::ToUInt32(list[0]);
        m_nRuns = Convert::ToUInt32(list[1]);
        m_Pm = Convert::ToDouble(list[2]);
        m_Pc = Convert::ToDouble(list[3]);
        m_nCrossOver = list[4] == "Single" ? 0 : 1;
        return true;
      }
      Console::WriteLine("Something is amiss in GAParams.csv");
      return false;
    }
  };
}