using CNTK;
using C = CNTK.CNTKLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepSharp
{

    public delegate Function OneArgumentModule(Variable input);
    public delegate Function MultiArgumentsModule(params Variable[] inputs);

    public static partial class NN
    {


        public static OneArgumentModule Linear(int inputSize, int outputSize, bool biase = true, string name = "")
        {
            var initializer = C.GlorotUniformInitializer(
                                C.DefaultParamInitScale,
                                C.SentinelValueForInferParamInitRank,
                                C.SentinelValueForInferParamInitRank);



            Parameter w = new Parameter(new int[] { outputSize, inputSize }, DataType.Float, initializer);
            Parameter b = null;
            if (biase)
            {
                b = new Parameter(new int[] { outputSize }, DataType.Float, 0);
            }


            return x => biase ? C.Times(w, x) + b : C.Times(w, x);
        }

        public static OneArgumentModule Linear(int outputSize, bool biase = true, string name = "")
        {
            var initializer = C.GlorotUniformInitializer(
                    C.DefaultParamInitScale,
                    C.SentinelValueForInferParamInitRank,
                    C.SentinelValueForInferParamInitRank);

            Parameter w = new Parameter(new int[] { outputSize, NDShape.InferredDimension }, DataType.Float, initializer);
            Parameter b = null;
            if (biase)
            {
                b = new Parameter(new int[] { outputSize }, DataType.Float, 0);
            }


            return x => biase ? C.Times(w, x) + b : C.Times(w, x);
        }



        /// <summary>
        /// A simple lookup table that stores embeddings of a fixed dictionary and size.
        /// This module is often used to store word embeddings and retrieve them using indices.
        /// The input to the module is a list of indices, and the output is the corresponding word embeddings.
        /// </summary>
        /// <param name="numEmbeddings">Size of the dictionary of embeddings</param>
        /// <param name="embeddingDim">The size of each embedding vector</param>
        /// <param name="initializer">Learnable embedding only) initial value of weights `E`</param>
        /// <param name="weights"></param>
        /// <param name="name">The name of the function instance in the network</param>
        /// <returns></returns>
        //TODO: replace the initializer 
        public static OneArgumentModule Embedding(int numEmbeddings, int embeddingDim, CNTKDictionary initializer, float[][] weights = null, bool trainable = true, string name = null)
        {
            Variable w;
            if (weights == null && initializer == null)
            {
                throw new Exception();
            }

            if(trainable)
            {
                w = new Parameter(new int[] { embeddingDim, numEmbeddings }, DataType.Float, initializer);
                if (weights != null)
                {
                    var data = Value.Create<float>(w.Shape, weights, null, DeviceDescriptor.UseDefaultDevice()).Data;
                    ((Parameter)w).SetValue(data);
                }
            }
            else
            {
                if (weights != null)
                {
                    var data = Value.Create<float>(new int[] { embeddingDim, numEmbeddings },
                        weights, null, DeviceDescriptor.UseDefaultDevice()).Data;

                    w = new Constant(data);

                }
                else
                {
                    throw new Exception();
                }
                
            }
            
            return x => C.Times(w, x);

        }

        
    }
}
