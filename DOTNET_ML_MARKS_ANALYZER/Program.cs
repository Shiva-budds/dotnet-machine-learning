
using Microsoft.ML;

namespace DOTNET_ML_MARKS_ANALYZER
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromTextFile<StudentData>(
                path: @"c:\Users\shiva\Downloads\student_data.csv",
                hasHeader: true,
                separatorChar:','
            );//load the the data 
            var pipeLine = context.Transforms
            .Concatenate("Features",nameof(StudentData.StudyHours), nameof(StudentData.Attendance))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(
                new Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer.Options
                {
                    MaximumNumberOfIterations = 10 
                }
            ));
            Console.WriteLine("Training the model....");
            var model = pipeLine.Fit(data);            /*
            var newStudent = new StudentData
            {
                StudyHours = 8,
                Attendance = 80
            };
            */

            Console.WriteLine("Enter Study Hours:");
            float inputStudyHours = float.Parse(Console.ReadLine() ?? "0");

            Console.WriteLine("Enter Attendance Percentage:");
            float inputAttendance = float.Parse(Console.ReadLine() ?? "0");

            var newStudent = new StudentData
            {
                StudyHours = inputStudyHours,
                Attendance = inputAttendance
            };

            Console.WriteLine("Model training complete.");
            var predictor = context.Model.CreatePredictionEngine<StudentData , StudentPrediction>(model);
            // var newStudent = new StudentData
            // {
            //     StudyHours = 8,
            //     Attendance = 80
            // };
            var result = predictor.Predict(newStudent);
            Console.WriteLine($"Study Hours: {newStudent.StudyHours}, Attendance: {newStudent.Attendance} \nWillPass: {result.Passed}, Probability: {result.Probability:P2}");
        }
    }
}
