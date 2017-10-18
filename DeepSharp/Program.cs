using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using nn = DeepSharp.NN;
using CNTK;
using C = CNTK.CNTKLib;
namespace DeepSharp
{
    class Program
    {
        private static void GenerateRawDataSamples(int sampleSize, int inputDim,

            out float[] data)

        {

            Random random = new Random(0);



            data = new float[sampleSize * inputDim];





            for (int sample = 0; sample < sampleSize; sample++)

            {





                for (int i = 0; i < inputDim; i++)

                {

                    data[sample * inputDim + i] = (float)GenerateGaussianNoise(3, 1, random) * 3;

                }

            }

        }



        static double GenerateGaussianNoise(double mean, double stdDev, Random random)

        {

            double u1 = 1.0 - random.NextDouble();

            double u2 = 1.0 - random.NextDouble();

            double stdNormalRandomValue = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            return mean + stdDev * stdNormalRandomValue;

        }
        static void Main(string[] args)
        {

            void TestConv2D()
            {
                var x = Variable.InputVariable(new int[] { 50, 100, 16}, DataType.Float);

                var m = nn.Conv2D(33, 3, stride: 2);

                var y = m(x);

                foreach (var d in y.Output.Shape.Dimensions)
                {
                    Console.Write(d + " ");
                }

                Console.WriteLine();

                m = nn.Conv2D(33, (3,5), stride:(2, 1), padding:(true, true), dilation:(1, 1));
                y = m(x);

                foreach (var d in y.Output.Shape.Dimensions)
                {
                    Console.Write(d + " ");
                }

                Console.WriteLine();

            }


            TestConv2D();

            //var m = nn.Conv1D(128, 3);
            //var inputSize = 6;
            //var outputSize = 3;
            //var steps = 12;
            //var bsz = 6;
            //NDShape shape = NDShape.CreateNDShape(new int[] { steps, inputSize });
            //var inputVariable = Variable.InputVariable(shape, DataType.Float);
            //var y = m(inputVariable);


            ////float[] input;

            //GenerateRawDataSamples(bsz, inputSize * steps, out float [] input);
            //var inputValue = Value.CreateBatch<float>(shape, input, DeviceDescriptor.UseDefaultDevice());
            
            



            //var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, inputValue } };

            //var outputDataMap = new Dictionary<Variable, Value>() { { y.Output, null } };

            //y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.UseDefaultDevice());
            //var outputValue = outputDataMap[y.Output];
            //IList<IList<float>> output = outputValue.GetDenseData<float>(y.Output);
        }
    }
}
