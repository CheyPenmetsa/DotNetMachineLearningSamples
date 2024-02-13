using Microsoft.ML.OnnxRuntime.Tensors;

namespace SklearnOnnxApi
{
    public class HappyClassificationData
    {
        public Int64 Infoavail { get; set; }

        public Int64 Housecost { get; set; }

        public Int64 Schoolquality { get; set; }

        public Int64 Policetrust { get; set; }

        public Int64 Streetquality { get; set; }

        public Int64 Events { get; set; }

        public Tensor<Int64> AsTensor()
        {
            Int64[] data = new Int64[]
            {
            Infoavail, Housecost, Schoolquality, Policetrust, Streetquality, Events
            };
            int[] dimensions = new int[] { 1, 6 };
            return new DenseTensor<Int64>(data, dimensions);
        }
    }
}
