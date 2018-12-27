using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;

namespace MLAI
{
    public class Mlai
    {
       
        //public TextLoader textLoader;
        //public MLContext mlContext;

        public MLContext CreateAMLContext()
        {
            MLContext mlContext = null;
            try
            {
                mlContext = new MLContext(seed:0);
            }
            catch (Exception ex)
            {
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + ex + "   " + ex.StackTrace);
            }
            return mlContext;
        }

        public TextLoader CreateATextLoader(MLContext mlContext)
        {
            TextLoader textLoader = null;
            try
            {
               
                textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = ";",
                    HasHeader = true,
                    Column = new[]
                    {
                    new TextLoader.Column("Sig1", DataKind.R4, 0),
                    new TextLoader.Column("Sig2", DataKind.R4, 1),
                    new TextLoader.Column("Sig3", DataKind.R4, 2),
                    new TextLoader.Column("Sig4", DataKind.R4, 3),
                    new TextLoader.Column("Sig5", DataKind.R4, 4),
                    new TextLoader.Column("Sig6", DataKind.R4, 5),
                    new TextLoader.Column("Sig7", DataKind.R4, 6),
                    new TextLoader.Column("Sig8", DataKind.R4, 7),
                    new TextLoader.Column("Sig9", DataKind.R4, 8),
                    new TextLoader.Column("Sig10", DataKind.R4, 9),
                    new TextLoader.Column("Sig11", DataKind.R4, 10),
                    new TextLoader.Column("Sig12", DataKind.R4, 11),
                    new TextLoader.Column("Sig13", DataKind.R4, 12),
                    new TextLoader.Column("Sig14", DataKind.R4, 13),
                    new TextLoader.Column("Sig15", DataKind.R4, 14),
                    new TextLoader.Column("Sig16", DataKind.R4, 15),
                    new TextLoader.Column("Sig17", DataKind.R4, 16),
                    new TextLoader.Column("Sig18", DataKind.R4, 17),
                    new TextLoader.Column("Sig19", DataKind.R4, 18),
                    new TextLoader.Column("Sig20", DataKind.R4, 19),
                    new TextLoader.Column("RfdX", DataKind.R4, 20),
                }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + ex + "   " + ex.StackTrace);
            }
            return textLoader;
        }

        public ITransformer Train(MLContext mlContext, string dataPath, TextLoader textLoader, string finishedModelPath)
        {
            IDataView dataView = textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("RfdX", "Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Sig1", "Sig2", "Sig3", "Sig4", "Sig5", "Sig6", "Sig7", "Sig8", "Sig9", "Sig10", "Sig11", "Sig12", "Sig13", "Sig14", "Sig15", "Sig16", "Sig17", "Sig18", "Sig19", "Sig20"))
                .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            SaveModelAsFile(mlContext, model, finishedModelPath);
            return model;
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model, string modelPath)
        {
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
        }

        public RegressionEvaluator.Result Evaluate(MLContext mlContext, ITransformer model, string testDataPath, TextLoader textLoader)
        {
            IDataView dataView = textLoader.Read(testDataPath);
            var predictions = model.Transform(dataView);
            RegressionEvaluator.Result metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            return metrics;
        }

        public float TestSinglePrediction(List<double> signal, MLContext mlContext, string modelPath)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            var predictionFunction = loadedModel.MakePredictionFunction<AbstractSignal, PredictedRfd>(mlContext);
            var abstractSignal = new AbstractSignal()
            {
                Sig1 = (float)signal[0],
                Sig2 = (float)signal[1],
                Sig3 = (float)signal[2],
                Sig4 = (float)signal[3],
                Sig5 = (float)signal[4],
                Sig6 = (float)signal[5],
                Sig7 = (float)signal[6],
                Sig8 = (float)signal[7],
                Sig9 = (float)signal[8],
                Sig10 = (float)signal[9],
                Sig11 = (float)signal[10],
                Sig12 = (float)signal[11],
                Sig13 = (float)signal[12],
                Sig14 = (float)signal[13],
                Sig15 = (float)signal[14],
                Sig16 = (float)signal[15],
                Sig17 = (float)signal[16],
                Sig18 = (float)signal[17],
                Sig19 = (float)signal[18],
                Sig20 = (float)signal[19],
                RfdX = 0

            };
            var prediction = predictionFunction.Predict(abstractSignal);
            return prediction.RfdX;
        }

        public string SayHello()
        {
            return "Hello from framework ML api";
        }
    }
}
