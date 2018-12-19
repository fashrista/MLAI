using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLAI
{
    class Program
    {
        static void Main(string[] args)
        {
            Mlai mlai = new Mlai();
            Console.WriteLine(mlai.SayHello());
            RegressionEvaluator.Result result = mlai.CreateAModel();
            Console.WriteLine("RSquared:  " + result.RSquared + "  RMS: " + result.Rms);
           Console.ReadKey();
        }
    }
}
