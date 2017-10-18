using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepSharp
{
    public class FloatTensor : Tensor<float>
    {
        public FloatTensor(int[] dimensions) : base(dimensions)
        {
        }
    }
}
