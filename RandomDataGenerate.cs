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
    }
}
