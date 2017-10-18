using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using C = CNTK.CNTKLib;

namespace DeepSharp
{
    public static partial class NN
    {
        private static OneArgumentModule Pooling(PoolingType poolingType, int kernelSize, int rank = 1, int stride = 1, bool padding = false, bool ceilMode = false, string name = "")
        {
            int[] kernelShape = new int[rank];
            int[] strides = new int[rank];
            bool[] autoPadding = new bool[rank];

            for(int i = 0; i < rank; i++)
            {
                kernelShape[i] = kernelSize;
                strides[i] = stride;
                autoPadding[i] = padding;
            }

            return x => C.Pooling(x, poolingType, kernelShape, strides, new BoolVector(autoPadding), ceilMode, padding, name);
        }

        private static OneArgumentModule Pooling(PoolingType poolingType, int [] kernelShape, int [] strides, bool [] paddings, bool ceilMode = false, string name = "")
        {

            return x => C.Pooling(x, poolingType, kernelShape, strides, new BoolVector(paddings), ceilMode, true, name);
        }

        public static OneArgumentModule MaxPool1D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, kernelSize, rank: 1, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool1D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, kernelSize, rank: 1, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool2D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, kernelSize, rank: 2, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool2D((int h, int w) kernelShape, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride, stride }, new bool [] { padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool2D((int h, int w) kernelShape, (int h, int w) stride, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride.h, stride.w }, new bool[] { padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool2D((int h, int w) kernelShape, (int h, int w) stride, (bool h, bool w) padding, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride.h, stride.w }, new bool[] { padding.h, padding.w }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool2D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, kernelSize, rank: 2, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool2D((int h, int w) kernelShape, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride, stride }, new bool[] { padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool2D((int h, int w) kernelShape, (int h, int w) stride, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride.h, stride.w }, new bool[] { padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool2D((int h, int w) kernelShape, (int h, int w) stride, (bool h, bool w) padding, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, new int[] { kernelShape.h, kernelShape.w }, new int[] { stride.h, stride.w }, new bool[] { padding.h, padding.w }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool3D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, kernelSize, rank: 3, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool3D((int d, int h, int w) kernelShape, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.d, kernelShape.h, kernelShape.w }, new int[] { stride, stride, stride }, new bool[] { padding, padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool3D((int d, int h, int w) kernelShape, (int d, int h, int w) stride, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.d, kernelShape.h, kernelShape.w }, new int[] { stride.d, stride.h, stride.w }, new bool[] { padding, padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule MaxPool3D((int d, int h, int w) kernelShape, (int d, int h, int w) stride, (bool d, bool h, bool w) padding, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Max, new int[] { kernelShape.d, kernelShape.h, kernelShape.w }, new int[] {stride.d, stride.h, stride.w }, new bool[] {padding.d, padding.h, padding.w }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool3D(int kernelSize, int stride = 1, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, kernelSize, rank: 3, stride: stride, padding: padding, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool3D((int d, int h, int w) kernelShape, (int d, int h, int w) stride, bool padding = false, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, new int[] { kernelShape.d, kernelShape.h, kernelShape.w }, new int[] { stride.d, stride.h, stride.w }, new bool[] { padding, padding, padding }, ceilMode: cielMode, name: name);
        }

        public static OneArgumentModule AvgPool3D((int d, int h, int w) kernelShape, (int d, int h, int w) stride, (bool d, bool h, bool w) padding, bool cielMode = false, string name = "")
        {
            return Pooling(PoolingType.Average, new int[] { kernelShape.d, kernelShape.h, kernelShape.w }, new int[] { stride.d, stride.h, stride.w }, new bool[] { padding.d, padding.h, padding.w }, ceilMode: cielMode, name: name);
        }
    }
}
