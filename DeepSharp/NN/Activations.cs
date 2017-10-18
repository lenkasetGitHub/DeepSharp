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
        public static Func<Variable, Function> Sigmoid(string name=null)
        {
            return x => C.Sigmoid(x, name);
        }

        public static Func<Variable, Function> Tanh(string name = null)
        {
            return x => C.Tanh(x, name);
        }

        public static Func<Variable, Function> Softmax(string name = null)
        {
            return x => C.Softmax(x, name);
        }

        public static Func<Variable, Function> Softplus(string name = null)
        {
            return x => C.Softplus(x, name);
        }

        public static Func<Variable, Function> ReLU(string name = null)
        {
            return x => C.ReLU(x, name);
        }

    }

}
