using System;
using Microsoft.ML.Data;

namespace DOTNET_ML_MARKS_ANALYZER;

public class StudentData
{
    [LoadColumn(0)]
    public float StudyHours ;
    [LoadColumn(1)]
    public float Attendance ;
    [LoadColumn(2) , ColumnName("Label")]
    public bool Passed;
}
