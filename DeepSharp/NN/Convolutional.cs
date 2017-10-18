using CNTK;
using C = CNTK.CNTKLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepSharp
{
    public static partial class NN
    {


        private static OneArgumentModule Convolution(
            int nbFilters,
            int[] filterShape,
            int stride = 1,
            bool padding=false,
            int dilation = 1,
            bool bias = true,
            bool sharing=true,
            uint reductionRank=1,
            string name=""
            )
        {

            var rank = filterShape.Length;

            bool[] autoPadding = new bool[rank + 1];
            autoPadding[rank] = false;
            for (int i = 0; i < rank; i++)
            {
                autoPadding[i] = padding;
            }

            int[] strides = new int[rank + 1];
            strides[rank] = NDShape.InferredDimension;
            for(int i = 0; i < rank; i++)
            {
                strides[i] = stride;
            }


            if(dilation > 1)
            {
                if(stride != 1)
                {
                    throw new Exception("Invalid strides for dilated convolution");
                }
            }

            int[] dilations = new int[rank];
            for(int i = 0; i < rank; i++)
            {
                dilations[i] = dilation;
            }


            return Convolution(nbFilters, filterShape, strides, autoPadding, dilations, bias, new bool[] { sharing }, reductionRank, name);

        }

        private static OneArgumentModule Convolution(
            int nbFilters,
            int[] filterShape,
            int[] strides,
            bool[] paddings,
            int[] dilations,
            bool bias,
            bool[] sharing,
            uint reductionRank = 1,
            string name = ""
            )
        {
            var kernelInitializer = C.GlorotUniformInitializer(
                        C.DefaultParamInitScale,
                        C.SentinelValueForInferParamInitRank,
                        C.SentinelValueForInferParamInitRank);

            var rank = filterShape.Length;
            int[] kernelShape = new int[rank + 2];

            for (int i = 0; i < filterShape.Length; i++)
            {
                kernelShape[i] = filterShape[i];
            }

            kernelShape[rank] = NDShape.InferredDimension;
            kernelShape[rank + 1] = nbFilters;


            int[] biasShape = new int[rank + 1];
            for (int i = 0; i < rank; i++)
            {
                biasShape[i] = 1;
            }
            biasShape[rank] = nbFilters;

            Parameter w = new Parameter(kernelShape, DataType.Float, kernelInitializer);

            Parameter b = bias ? new Parameter(biasShape, DataType.Float, 0) : null;

            Function _Convolve(Variable x)
            {
                var r = C.Convolution(w, x, strides, new BoolVector(sharing), new BoolVector(paddings), dilations, reductionRank, 0, name);
                if (bias)
                {
                    r += b;
                }

                return r;
            }

            return _Convolve;
        }


        public static OneArgumentModule Conv1D(
            int nbFilters,
            int kernelSize,
            int stride = 1,
            bool padding = false,
            int dilation = 1,
            bool bias = true,
            string name = "")
        {

            return Convolution(nbFilters, new int[] { kernelSize }, stride, padding, dilation, bias, true, 1, name);
        }

        public static OneArgumentModule Conv2D(
            int nbOutChannels,
            (int h, int w) kernleSize,
            (int h, int w) stride,
            (bool h, bool w) padding,
            (int h, int w) dilation,
            bool bias=true,
            string name="")
        {
            return Convolution(nbOutChannels,
                new int[] { kernleSize.h, kernleSize.w },
                new int[] { stride.h, stride.w },
                new bool[] { padding.h, padding.w }, 
                new int[] { dilation.h, dilation.w }, 
                bias, new bool[] { true }, 1, name);
        }

        public static OneArgumentModule Conv2D(
            int nbOutChannels,
            int kernelSize,
            int stride = 1,
            bool padding = false,
            int dilation = 1,
            bool bias = true,
            string name = "")
        {

            return Conv2D(nbOutChannels, (kernelSize, kernelSize), (stride, stride), (padding, padding), (dilation, dilation), bias, name);
        }

        public static OneArgumentModule Conv3D(
            int nbOutChannels,
            (int d, int h, int w) kernleSize,
            (int d, int h, int w) stride,
            (bool d, bool h, bool w) padding,
            (int d, int h, int w) dilation,
            bool bias = true,
            string name = "")
        {
            return Convolution(nbOutChannels,
                new int[] { kernleSize.d, kernleSize.h, kernleSize.w },
                new int[] {stride.d, stride.h, stride.w },
                new bool[] {padding.d, padding.h, padding.w },
                new int[] { dilation.d, dilation.h, dilation.w },
                bias, new bool[] { true }, 1, name);
        }

        public static OneArgumentModule Conv3D(
            int nbOutChannels,
            int kernelSize,
            int stride = 1,
            bool padding = false,
            int dilation = 1,
            bool bias = true,
            string name = "")
        {

            return Conv3D(nbOutChannels, (kernelSize, kernelSize, kernelSize), (stride, stride, stride), (padding, padding, padding), (dilation, dilation, dilation), bias, name);
        }

    }
}
