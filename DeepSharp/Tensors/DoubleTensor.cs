using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepSharp
{
    public class DoubleTensor : Tensor<double>
    {
        public DoubleTensor(int[] dimensions) : base(dimensions)
        {
        }
    }
}
