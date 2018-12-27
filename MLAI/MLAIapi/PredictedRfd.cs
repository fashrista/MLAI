using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;


namespace MLAI
{
    public class PredictedRfd
    {
        [ColumnName("Score")]
        public float RfdX;
    }
}
