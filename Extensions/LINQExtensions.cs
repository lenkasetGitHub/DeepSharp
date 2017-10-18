using System;
using System.Collections.Generic;
using System.Text;

namespace Extensions
{
    public static class LINQExtensions
    {
        public static long Product(this ICollection<int> data)
        {
            long p = 1;
            foreach (var x in data)
            {
                p *= x;
            }
            return p;
        }

        public static long Product(this ICollection<long> data)
        {
            long p = 1;
            foreach (var x in data)
            {
                p *= x;
            }
            return p;
        }
    }
}
