using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Extensions;
namespace DeepSharp
{
    public interface ITensor
    {
        
    }

    public abstract class Tensor<T>: 
        ITensor,
        IDisposable
        where T:
        struct,
        IComparable,
        IComparable<T>,
        IConvertible,
        IEquatable<T>,
        IFormattable
    { 
        protected T[] data;
        protected int numDimensions;
        protected long numElements;
        protected int[] dimensions;
        protected int[] strides;

        public Tensor(int[] dimensions)
        {
            this.numElements = dimensions.Product();
            this.numDimensions = dimensions.Length;
            this.data = new T[this.numElements];
            this.dimensions = dimensions;

            int stride = 1;
            int n = dimensions.Length - 1;
            for (; n >= 0; n--)
            {

                this.strides[n] = stride;
                stride *= dimensions[n];
            }
        }
        public void Dispose()
        {
            
        }

        public override string ToString()
        {

            var precision = 4;
            var threshold = 1000;
            var edgeItems = 3;
            var lineWidth = 80;
            var minSize = 5;

            
            //(string Formatter, int Size) FormatNumber()
            //{
            //    var expMin = data.Min().ToDouble(null);

            //}

            StringBuilder stringBuilder = new StringBuilder();
           
            if (this.numDimensions == 1)
            {
                
                int i = 0;
                for (; i < data.Length -1; i++)
                {
                    stringBuilder.Append(data[i]);
                    stringBuilder.Append(" ");
                }
                stringBuilder.Append(data[i]);

            }
            else if(this.numDimensions == 2)
            {
                
                for (int i = 0; i < dimensions[0]; i++)
                {
                    int j = 0;

                    
                    for (; j < dimensions[1] - 1; j++)
                    {
                        stringBuilder.Append(data[i * dimensions[1] + j]);
                        stringBuilder.Append(" ");
                    }
                    stringBuilder.Append(data[i * dimensions[1] + j]);
                    
                    if(i < dimensions[0] - 1)
                        stringBuilder.AppendLine();
                }
            }
            else
            {
                int n = 0;
                int r = this.dimensions[this.numDimensions - 2];
                int c = this.dimensions[this.numDimensions - 1];
                int[] dims = new int[this.numDimensions];

                while(n < this.numElements)
                {
                    stringBuilder.AppendLine();
                    stringBuilder.Append("(");
                    for (int i = 0; i < this.numDimensions - 2; i++)
                    {
                        stringBuilder.Append(dims[i]);
                        stringBuilder.Append(",");
                    }
                    stringBuilder.Append(".,.)=");
                    stringBuilder.AppendLine();

                    for (int i = 0; i < r; i++)
                    {
                        for (int j = 0; j < c; j++)
                        {
                            stringBuilder.Append(this.data[n++]);
                            stringBuilder.Append(" ");
                           
                        }

                        stringBuilder.AppendLine();
                    }

                    int curid = this.numDimensions - 3;
                    while (curid >= 0)
                    {
                        dims[curid]++;
                        if (dims[curid] == this.dimensions[curid])
                        {
                            dims[curid] = 0;
                            curid--;
                        }
                        else
                            break;
                    }
                }
            }

            
            stringBuilder.AppendLine();
            stringBuilder.Append("[");
            stringBuilder.Append(this.GetType().FullName);
            stringBuilder.Append(" of size ");
            int k = 0;
            for(; k < this.numDimensions - 1; k++)
            {
                stringBuilder.Append(dimensions[k]);
                stringBuilder.Append("x");
            }
            stringBuilder.Append(dimensions[k]);
            stringBuilder.Append(" ]");
            return stringBuilder.ToString();
        }

        //protected abstract void PosInfMask(out T[] mask);
        //protected abstract void NegInfMask(out T[] mask);
    }

    
}
