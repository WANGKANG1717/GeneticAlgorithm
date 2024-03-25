namespace RandomDataGenerate
{
    public class MyProgram
    {
        public void Run()
        {
            randomData(100);
        }

        public double f(int a, int b, int c, int d, int x, int y, int z)
        {
            return a * Math.Sin(x) + b * Math.Cos(y) + c / z + d;
        }

        public double f2(int x, int a)
        {
            return x * Math.Sin(a * Math.PI * x) + 2;
        }

        public double f3(int x, int a, int b)
        {
            return b * x * x + 2 * a * x + 1;
        }

        public void randomData(int N)
        {
            int a = 1;
            int b = 2;
            int c = 6;
            int d = 10;
            Random random = new Random();
            string resX = "[";
            string resY = "[";
            for (int i = 1; i <= N; i++)
            {
                int x = i;
                int y = random.Next(1, 101);
                int z = random.Next(1, 101);

                resX += "[" + x + "," + y + "," + z + "],";
                resY += f(a, b, c, d, x, y, z).ToString("0.00") + ",";
            }
            resX += "]";
            resY += "]";
            Console.WriteLine(resX);
            Console.WriteLine(resY);
        }

        public void randomData2(int N)
        {
            int a = 10;
            string resX = "[";
            string resY = "[";
            for (int x = 1; x <= N; x++)
            {
                resX += "[" + x + "],";
                resY += f2(x, a) + ",";
            }
            resX += "]";
            resY += "]";
            Console.WriteLine(resX);
            Console.WriteLine(resY);
        }

        public void randomData3(int N)
        {
            int a = 10;
            int b = 100;
            string resX = "[";
            string resY = "[";
            for (int x = 1; x <= N; x++)
            {
                resX += "[" + x + "],";
                resY += f3(x, a, b) + ",";
            }
            resX += "]";
            resY += "]";
            Console.WriteLine(resX);
            Console.WriteLine(resY);
        }
    }
}
