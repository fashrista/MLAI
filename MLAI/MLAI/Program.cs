using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLAI
{
    class Program
    {
        //paths to training and testing data , and also path to trained model 
        static readonly string trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\Asus\AppData\LocalLow\S2P\Muscle board\TrainingSignals.csv");
        static readonly string testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\Asus\AppData\LocalLow\S2P\Muscle board\TrainingSignals1.csv");
        static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\Asus\AppData\LocalLow\S2P\Muscle board\Model.zip");
        static readonly string serverPath = Path.Combine(Environment.CurrentDirectory, "Data", @"C:\Users\Asus\AppData\LocalLow\S2P\Muscle board\server.csv");
        static void Main(string[] args)
        {
            Mlai mlai = new Mlai();
            Console.WriteLine(mlai.SayHello());
            var mlContext = mlai.CreateAMLContext();
            var textLoader = mlai.CreateATextLoader(mlContext);
            var model = mlai.Train(mlContext, trainDataPath, textLoader, modelPath);
            var result = mlai.Evaluate(mlContext, model, testDataPath, textLoader);
            Console.WriteLine("RSquared:  " + result.RSquared + "  RMS: " + result.Rms);
            List<double> signal = new List<double>() {31, 14,  133,  22 , 11 , 11,  16 , 18 , 12 ,12 ,14 ,11 ,13 ,14 ,15 ,143 ,484523,  123 ,   123 ,   957};
            float rfd = mlai.TestSinglePrediction(signal, mlContext, modelPath);

            Console.WriteLine("prediction:  " + rfd );
            Console.ReadKey();
        }
    }
}
