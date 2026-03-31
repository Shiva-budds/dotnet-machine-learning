using System;
using Microsoft.ML.Data;

namespace DOTNET_ML_MARKS_ANALYZER;

public class StudentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Passed;
    public float Probability{get; set;}
    public float Score{get; set;}

}
