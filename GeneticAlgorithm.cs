/**
 * @date: 2024-03-04 21:11:41
 * @author: WangKang
 * @blog: 
 * @filepath: GeneticAlgorithm.cs
 * @description: 遗传算法
 * Copyright 2024 WANGKANG, All Rights Reserved.
 */

/*
算法目标：优化函数参数，使得输出误差与实际值差距最小
y = f(x)
输入x，y，优化参数a
x：为矩阵[n, m] 行为输入的参数 列为同时输入的数据数
y: 为一维向量[n, 1] 每一行是x根据f(x)对应的输出值
a: 为一维向量[1, k] f(x)的需要优化的参数值

demo: y = a * sinx + b *cosy + c/x + d
X:[[1, 2], [2, 3], [3, 4]]
Y:[10, 15, 20]
根据这组数据，找出最贴切的
 */

// 编码方式： 使用浮点数编码
// todo 设置多种交叉方式 1. 单点交叉 2. 两点交叉 3. 均匀交叉    // 搞定
// todo 设置多种变异方式 1. 单点变异 2. 均匀变异
// end 多种选择方式 1. 轮盘赌选择 2. 随机选择 3. 锦标赛选择 4. 多目标选择
// end 多种适应度计算方式 1. 均方误差 2. 绝对值误差 3. 相对值误差 4. 自定义适应度计算方式

using System.Text;

namespace GA_Template
{
    public class GeneticAlgorithm
    {
        private string encodeType = "Binary"; // 编码方式 Binary / Double
        private double[][] X;
        private double[] Y;
        private double[][] Y_hat; // 行为chromosomeNum，列为N
        private int N; // =X的行数 / =Y的行数
        private int M; // =X的列数
        private int K; // 需要优化的参数个数
        private double minValue; // 参数取值范围
        private double maxValue;
        private int iteratorNum; // 迭代次数
        private int chromosomeNum; // 染色体数量
        private double mutationRate; // 变异概率
        private double crossoverRate; // 交叉概率

        private string[][]? chromosomeMatrix; // 染色体矩阵 二进制编码

        private double[][]? chromosomeMatrixDouble; // 染色体矩阵 浮点数编码

        double accuracy = 0.001; // 精度
        string crossType; // 交叉方式 single/twoPoint/uniform
        string mutationType; // 变异方式 single/uniform
        int numberOfBits = 6; // 二进制编码的位数 需要根据需要的精度进行动态计算

        Func<double[], double[], double> function; // 计算函数 使用Lambda表达式

        private List<double[]> resultAdaptability = []; // 适应度结果集
        private List<int[][]> resultChromosomeMatrix = []; // 染色体矩阵结果集

        string[] bestChromosome; // 最大适应度 二进制编码
        double[] bestChromosomeDouble; // 最大适应度 浮点数编码


        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="X">输入参数矩阵</param>
        /// <param name="Y">输出结果向量</param>
        /// <param name="K">需要优化的参数个数</param>
        /// <param name="minValue">参数取值范围下限</param> 
        /// <param name="maxValue">参数取值范围上限</param>
        /// <param name="iteratorNum">迭代次数</param>
        /// <param name="chromosomeNum">染色体数量</param>
        /// <param name="crossoverRate">交叉概率</param>
        /// <param name="mutationRate">变异概率</param>
        /// <param name="function">计算函数</param>
        /// <param name="accuracy">精度</param>
        /// <param name="crossType">交叉方式 single/twoPoint/uniform</param>
        /// <param name="mutationType">变异方式 single/uniform</param>
        public GeneticAlgorithm(
            double[][] X,
            double[] Y,
            int K,
            double minValue,
            double maxValue,
            int iteratorNum,
            int chromosomeNum,
            double crossoverRate,
            double mutationRate,
            Func<double[], double[], double> function,
            double accuracy = 0.001, // 二进制编码有用
            string crossType = "single",
            string mutationType = "single",
            string encodeType = "Binary")
        {
            Verify(X, Y);
            if (chromosomeNum % 2 != 0)
            {
                throw new Exception("染色体数量必须为偶数");
            }
            this.X = X;
            this.Y = Y;
            this.N = X.Length;
            this.M = X[0].Length;
            this.K = K;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.iteratorNum = iteratorNum;
            this.chromosomeNum = chromosomeNum;
            this.crossoverRate = crossoverRate;
            this.mutationRate = mutationRate;
            this.accuracy = accuracy;
            this.crossType = crossType;
            this.mutationType = mutationType;

            InitY_hat();
            this.numberOfBits = calculateNumberOfBits();

            this.chromosomeMatrix = null;
            this.function = function;

            if (encodeType == "Binary" || encodeType == "Double")
            {
                this.encodeType = encodeType;
            }
            else
            {
                throw new Exception("不支持的类型(仅支持Binary和Double类型)");
            }
        }

        public void Test()
        {
            // Console.WriteLine("numberOfBits:" + this.numberOfBits);
            // Console.WriteLine(Decode("11111111111"));
            // Console.WriteLine(Decode("00000000000"));

            // Console.WriteLine(StringToNumber("111"));
            // Console.WriteLine(ReverseString("1,2,3,34,4,5,6,7,8,9"));

            // for (int i = 0; i < 10; i++)
            // {
            // 	 int number = RandomNumber(4);
            // 	string bianryString = RandomBinaryString(4);
            // 	 Console.WriteLine(number + "," + bianryString);
            // }

            Run();
            // for (int i = 0; i < chromosomeNum; i++)
            // {
            // 	for (int j = 0; j < K; j++)
            // 	{
            // 		Console.Write(chromosomeMatrix[i][j] + ", ");
            // 	}
            // 	Console.WriteLine();
            // }
            // Console.WriteLine("============================");
            // for (int i = 0; i < chromosomeNum; i++)
            // {
            //     for (int j = 0; j < K; j++)
            //     {
            //         Console.Write(Decode(chromosomeMatrix[i][j]) + ", ");
            //     }
            //     Console.WriteLine();
            // }
            double[] adaptability = CalculateAdaptability();
            double[] naturalSelectionRate = CalculateNaturalSelectionRate(adaptability);
            int index = -1;
            double maxAdaptability = double.MinValue;

            for (int i = 0; i < chromosomeNum; i++)
            {
                if (adaptability[i] > maxAdaptability)
                {
                    maxAdaptability = adaptability[i];
                    index = i;
                }
            }
            Console.WriteLine("============================");
            for (int i = 0; i < K; i++)
            {
                if (encodeType == "Binary")
                {
                    Console.Write(Decode(chromosomeMatrix[index][i]) + ", ");
                }
                else if (encodeType == "Double")
                {
                    Console.Write(chromosomeMatrixDouble[index][i] + ", ");
                }
            }
            Console.WriteLine();
            Console.WriteLine("=====================");
            for (int i = 0; i < K; i++)
            {
                if (encodeType == "Binary")
                {
                    Console.Write(bestChromosome[i] + ", ");
                }
                else if (encodeType == "Double")
                {
                    Console.Write(bestChromosomeDouble[i] + ", ");
                }
            }
        }


        /// <summary>
        /// 计算符合精度的二进制编码的长度
        /// </summary>
        /// <returns>二进制编码的应有长度</returns>
        private int calculateNumberOfBits()
        {
            int length = (int)Math.Ceiling(Math.Log2((maxValue - minValue) / accuracy + 1));
            if (length > 32)
            {
                length = 32;
                Console.WriteLine("Warning: 二进制编码位数超过32位，将使用32位");
            }
            return length;
        }


        /// <summary>
        /// 解码二进制编码
        /// </summary>
        /// <param name="binaryString">二进制编码</param>
        /// <returns>解码后的实际值</returns>
        private double Decode(string binaryString)
        {
            if (binaryString.Length != numberOfBits)
            {
                throw new Exception("二进制编码长度不正确");
            }
            return minValue + (maxValue - minValue) / (Math.Pow(2, numberOfBits) - 1) * StringToNumber(binaryString);
        }

        /// <summary>
        /// 解码二进制编码
        /// </summary>
        /// <param name="binaryString">染色体</param>
        /// <returns>解码后的实际值</returns>
        private double[] Decode(string[] chromosome)
        {
            double[] res = new double[K];
            for (int i = 0; i < K; i++)
            {
                res[i] = Decode(chromosome[i]);
            }
            return res;
        }


        /// <summary>
        /// 将二进制字符串转换为十进制数字
        /// </summary>
        /// <param name="binaryString">二进制字符串</param>
        /// <returns>十进制数字</returns>
        private int StringToNumber(string binaryString)
        {
            int res = 0;
            for (int i = 0; i < binaryString.Length; i++)
            {
                res = res * 2 + binaryString[i] - '0';
            }
            return res;
        }

        private string RandomBinaryString(int length)
        {
            int randomNumber = RandomNumber(length);
            // Console.Write("/" + randomNumber + "/ ");
            string binaryString = "";
            for (int i = 0; i < length; i++)
            {
                binaryString += (randomNumber % 2).ToString();
                randomNumber /= 2;
            }
            // Console.WriteLine("#" + binaryString + "#");
            binaryString = ReverseString(binaryString);
            // Console.WriteLine("#" + binaryString + "#");
            return binaryString;
        }

        private string ReverseString(string str)
        {
            return new string(str.ToCharArray().Reverse().ToArray());
        }

        private int RandomNumber(int length)
        {
            Random random = new Random();
            return random.Next(0, (int)Math.Pow(2, length));
        }

        public double[] Run()
        {
            GenerateFirstGeneration();
            // 迭代繁衍
            for (int itIndex = 0; itIndex < iteratorNum; itIndex++)
            {
                // 计算各条染色体的适应度
                double[] adaptability = CalculateAdaptability();
                double[] naturalSelectionRate = CalculateNaturalSelectionRate(adaptability);
                if (encodeType == "Binary")
                {
                    // 按照自然选择率大小先选择{chromosomeNum}条染色体
                    // 相当于np.random.choice
                    string[][] newChromosomeMatrix;
                    newChromosomeMatrix = Select(chromosomeMatrix, naturalSelectionRate);
                    // 选择{crossoverNum}条染色体参与交叉
                    newChromosomeMatrix = Cross(newChromosomeMatrix);
                    // 选择{mutationNum}条染色体参与变异
                    newChromosomeMatrix = Mutation(newChromosomeMatrix, itIndex);

                    // 更新chromosomeMatrix
                    chromosomeMatrix = newChromosomeMatrix;
                    if (chromosomeMatrix.Length != chromosomeNum)
                    {
                        throw new Exception("染色体数量不正确");
                    }
                }
                else if (encodeType == "Double")
                {
                    // 按照自然选择率大小先选择{chromosomeNum}条染色体
                    // 相当于np.random.choice
                    double[][] newChromosomeMatrix;
                    // 选择{crossoverNum}条染色体参与交叉
                    newChromosomeMatrix = SelectDouble(chromosomeMatrix, naturalSelectionRate);
                    newChromosomeMatrix = CrossDouble(newChromosomeMatrix);
                    // 选择{mutationNum}条染色体参与变异
                    newChromosomeMatrix = MutationDouble(newChromosomeMatrix, itIndex);

                    // 更新chromosomeMatrix
                    chromosomeMatrixDouble = newChromosomeMatrix;
                    if (chromosomeMatrixDouble.Length != chromosomeNum)
                    {
                        throw new Exception("染色体数量不正确");
                    }
                }
            }

            // 返回值
            if (encodeType == "Binary")
            {
                return Decode(bestChromosome);
            }
            else
            {
                return bestChromosomeDouble;
            }
        }

        private double[][] MutationDouble(double[][] newChromosomeMatrix, int iterations)
        {
            Random random = new Random();
            for (int i = 0; i < chromosomeNum; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    double rate = random.NextDouble();
                    if (rate <= mutationRate * Math.Pow(1 - iterations / (iteratorNum - 1), 2)) // 变异率应该逐步减少，，以达到稳定状态
                    {
                        newChromosomeMatrix[i][j] = RandomMutationDouble(newChromosomeMatrix[i][j]); // 单点变异
                    }
                }
            }
            return newChromosomeMatrix;
        }

        private double RandomMutationDouble(double val)
        {
            Random random = new Random();
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }

        private double[][] CrossDouble(double[][] newChromosomeMatrix)
        {
            if (newChromosomeMatrix == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            if (newChromosomeMatrix.Length != chromosomeNum)
            {
                throw new Exception("染色体数量不正确！");
            }
            Random random = new Random();
            double[][] tmpChromosomeMatrix = new double[chromosomeNum][];
            for (int i = 0; i < newChromosomeMatrix.Length; i++)
            {
                double rate = random.NextDouble();
                // 判断该染色体是否需要交叉
                if (rate <= crossoverRate)
                {
                    // 当前染色体作为父本
                    int j = random.Next(0, chromosomeNum); // 选择需要交叉的母本
                    tmpChromosomeMatrix[i] = RandomCrossDouble(newChromosomeMatrix[i], newChromosomeMatrix[j]);// 随机浮点数交叉
                }
                else // 不交叉，直接保留
                {
                    tmpChromosomeMatrix[i] = newChromosomeMatrix[i];
                }
            }
            return tmpChromosomeMatrix;
        }

        private double[] RandomCrossDouble(double[] chromosome1, double[] chromosome2)
        {
            Random random = new Random();
            double[] chromosome = new double[K];
            for (int k = 0; k < K; k++)
            {
                double rate = random.NextDouble();
                chromosome[k] = chromosome1[k] * rate + chromosome2[k] * (1.0 - rate);
            }
            return chromosome;
        }

        private double[][] SelectDouble(string[][]? chromosomeMatrix, double[] naturalSelectionRate)
        {
            if (chromosomeMatrixDouble == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            if (naturalSelectionRate == null)
            {
                throw new Exception("自然选择率未计算！");
            }
            double[][] newChromosomeMatrix = new double[chromosomeNum][];
            int[] selectedIndex = SelectChromosomeIndex(naturalSelectionRate, chromosomeNum);
            for (int i = 0; i < chromosomeNum; i++)
            {
                newChromosomeMatrix[i] = chromosomeMatrixDouble[selectedIndex[i]];
            }
            return newChromosomeMatrix;
        }

        private double[] CalculateNaturalSelectionRate(double[] adaptability)
        {
            if (adaptability == null)
            {
                throw new Exception("适应度未计算！");
            }
            double sumAdaptability = 0.0;
            double[] naturalSelectionRate = new double[chromosomeNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                sumAdaptability += adaptability[i];
            }
            for (int i = 0; i < chromosomeNum; i++)
            {
                naturalSelectionRate[i] = adaptability[i] / sumAdaptability;
            }
            return naturalSelectionRate;
        }

        /// <summary>
        /// 将选择最优染色体后的染色体矩阵进行交叉
        /// </summary>
        /// <param name="chromosomeMatrix">染色体矩阵</param>
        /// <param name="naturalSelectionRate">自然选择率</param>
        /// <returns>自然选择后的染色体矩阵</returns>
        private string[][] Select(string[][]? chromosomeMatrix, double[] naturalSelectionRate)
        {
            if (chromosomeMatrix == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            if (naturalSelectionRate == null)
            {
                throw new Exception("自然选择率未计算！");
            }
            string[][] newChromosomeMatrix = new string[chromosomeNum][];
            int[] selectedIndex = SelectChromosomeIndex(naturalSelectionRate, chromosomeNum);
            for (int i = 0; i < chromosomeNum; i++)
            {
                newChromosomeMatrix[i] = chromosomeMatrix[selectedIndex[i]];
            }
            return newChromosomeMatrix;
            /* // 使用键值对，将染色体适应度与染色体索引进行关联
            // 按照适应度从大到小排序
            KeyValuePair<int, double>[] index_selectRate = new KeyValuePair<int, double>[chromosomeNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                index_selectRate[i] = new KeyValuePair<int, double>(i, naturalSelectionRate[i]);
            }
            Array.Sort(index_selectRate, (x, y) => y.Value.CompareTo(x.Value));

            // 选择适应度最高的reserveNum条染色体保留下来
            for (int i = 0; i < reserveNum; i++)
            {
                newChromosomeList.Add(chromosomeMatrix[index_selectRate[i].Key]);
            }

            string[][] chromosomeMatrixAfterSelect = new string[chromosomeNum - reserveNum][];
            for (int i = reserveNum; i < chromosomeNum; i++)
            {
                chromosomeMatrixAfterSelect[i - reserveNum] = chromosomeMatrix[index_selectRate[i].Key];
            }
            return chromosomeMatrixAfterSelect; */
        }

        private int[] SelectChromosomeIndex(double[] naturalSelectionRate, int num)
        {
            int[] index = new int[num];
            for (int i = 0; i < num; i++)
            {
                index[i] = RWS(naturalSelectionRate);
            }
            return index;
        }

        /**
        * 轮盘赌算法
        * @param selectionProbability 概率数组(下标：元素编号、值：该元素对应的概率)
        * @returns {number} 返回概率数组中某一元素的下标
*/
        private int RWS(double[] selectionProbability)
        {
            Random random = new Random();
            double sum = 0;
            double rand = random.NextDouble();

            for (int i = 0; i < selectionProbability.Length; i++)
            {
                sum += selectionProbability[i];
                if (sum >= rand)
                {
                    return i;
                }
            }

            return selectionProbability.Length - 1;
        }

        private string[][] Cross(string[][]? newChromosomeMatrix)
        {
            if (newChromosomeMatrix == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            if (newChromosomeMatrix.Length != chromosomeNum)
            {
                throw new Exception("染色体数量不正确！");
            }
            Random random = new Random();
            string[][] tmpChromosomeMatrix = new string[chromosomeNum][];
            for (int i = 0; i < newChromosomeMatrix.Length; i++)
            {
                double rate = random.NextDouble();
                // 判断该染色体是否需要交叉
                if (rate <= crossoverRate)
                {
                    // 当前染色体作为父本
                    int j = random.Next(0, chromosomeNum); // 选择需要交叉的母本
                    string[] crossChromosome;
                    switch (crossType)
                    {
                        case "single":
                            crossChromosome = OnePointCross(newChromosomeMatrix[i], newChromosomeMatrix[j]);// 单点交叉
                            break;
                        case "twoPoint":
                            crossChromosome = TwoPointCross(newChromosomeMatrix[i], newChromosomeMatrix[j]);// 单点交叉
                            break;
                        case "uniform":
                            crossChromosome = UniformPointCross(newChromosomeMatrix[i], newChromosomeMatrix[j]);// 均匀交叉
                            break;
                        default:
                            crossChromosome = OnePointCross(newChromosomeMatrix[i], newChromosomeMatrix[j]);// 单点交叉
                            break;
                    }

                    tmpChromosomeMatrix[i] = crossChromosome;
                }
                else // 不交叉，直接保留
                {
                    tmpChromosomeMatrix[i] = newChromosomeMatrix[i];
                }
            }
            return tmpChromosomeMatrix;
        }

        /// <summary>
        /// 单点交叉
        /// </summary>
        private string[] OnePointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                int point = random.Next(1, numberOfBits - 1); // 随机选择交叉点 1-numberOfBits-1
                chromosome[k] = chromosome1[k].Substring(0, point) + chromosome2[k].Substring(point);
            }
            return chromosome;
        }

        /// <summary>
        /// 两点交叉
        /// </summary>
        private string[] TwoPointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                int mid = numberOfBits / 2;
                int point1 = random.Next(0, mid);
                int point2 = random.Next(mid, numberOfBits);
                chromosome[k] = chromosome1[k].Substring(0, point1) + chromosome2[k].Substring(point1, point2 - point1) + chromosome1[k].Substring(point2);
            }
            return chromosome;
        }

        /// <summary>
        /// 均匀交叉
        /// </summary>
        private string[] UniformPointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                chromosome[k] = "";
                for (int i = 0; i < numberOfBits; i++)
                {
                    chromosome[k] += random.NextDouble() <= 0.5 ? chromosome1[k][i] : chromosome2[k][i];
                }
            }
            return chromosome;
        }

        private string[][] Mutation(string[][] newChromosomeMatrix, int iterations)
        {
            Random random = new Random();
            for (int i = 0; i < chromosomeNum; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    double rate = random.NextDouble();
                    if (rate <= mutationRate * Math.Pow(1 - iterations / (iteratorNum - 1), 2)) // 变异率应该逐步减少，，以达到稳定状态
                    {
                        switch (mutationType)
                        {
                            case "single":
                                newChromosomeMatrix[i][j] = OnePointMutation(newChromosomeMatrix[i][j]); // 单点变异
                                break;
                            case "uniform":
                                newChromosomeMatrix[i][j] = UniformPointMutation(newChromosomeMatrix[i][j]); // 均匀变异
                                break;
                            default:
                                newChromosomeMatrix[i][j] = OnePointMutation(newChromosomeMatrix[i][j]); // 单点变异
                                break;
                        }
                    }
                }
            }
            return newChromosomeMatrix;
        }

        private string OnePointMutation(string chromosome)
        {
            Random random = new Random();
            int point = random.Next(0, numberOfBits);
            return chromosome.Substring(0, point) + (chromosome[point] ^ '1').ToString() + chromosome.Substring(point + 1);
        }

        private string UniformPointMutation(string chromosome)
        {
            string newChromosome = "";
            Random random = new Random();
            for (int i = 0; i < numberOfBits; i++)
            {
                newChromosome += random.NextDouble() <= 0.5 ? chromosome[i] : (chromosome[i] ^ '1').ToString();
            }
            return chromosome;
        }

        private void InitY_hat()
        {
            this.Y_hat = new double[this.chromosomeNum][];
            for (int i = 0; i < this.chromosomeNum; i++)
            {
                this.Y_hat[i] = new double[this.N];
            }
        }

        private static void Verify(double[][] X, double[] Y)
        {
            if (X.Length != Y.Length)
            {
                throw new Exception("X和Y的行数不一致");
            }
            for (int i = 1; i < X.Length; i++)
            {
                if (X[i].Length != X[i - 1].Length)
                {
                    throw new Exception("X的列数不一致");
                }
            }
        }

        /// <summary>
        /// 初始化第一代染色体
        /// </summary>
        private void GenerateFirstGeneration()
        {
            switch (encodeType)
            {
                case "Binary":
                    chromosomeMatrix = GenerateFirstGenerationBianry();
                    break;
                case "Double":
                    chromosomeMatrixDouble = GenerateFirstGenerationDouble();
                    break;
            }
        }

        private string[][] GenerateFirstGenerationBianry()
        {
            string[][] newChromosomeMatrix = new string[chromosomeNum][];
            for (var chromosomeIndex = 0; chromosomeIndex < chromosomeNum; chromosomeIndex++)
            {
                newChromosomeMatrix[chromosomeIndex] = new string[K];
                for (var i = 0; i < K; i++)
                {
                    newChromosomeMatrix[chromosomeIndex][i] = RandomBinaryString(numberOfBits);
                }
            }
            return newChromosomeMatrix;
        }
        private double[][]? GenerateFirstGenerationDouble()
        {
            double[][] newChromosomeMatrix = new double[chromosomeNum][];
            for (var chromosomeIndex = 0; chromosomeIndex < chromosomeNum; chromosomeIndex++)
            {
                newChromosomeMatrix[chromosomeIndex] = new double[K];
                for (var i = 0; i < K; i++)
                {
                    newChromosomeMatrix[chromosomeIndex][i] = RandomDouble(minValue, maxValue);
                }
            }
            return newChromosomeMatrix;
        }

        private double RandomDouble(double minValue, double maxValue)
        {
            Random random = new Random();
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }

        private double[] CalculateAdaptability()
        {
            if (encodeType == "Binary")
            {
                if (chromosomeMatrix == null)
                {
                    throw new Exception("染色体矩阵为空！");
                }
                CalculateY_hat(chromosomeMatrix, function);
            }
            else if (encodeType == "Double")
            {
                if (chromosomeMatrixDouble == null)
                {
                    throw new Exception("染色体矩阵为空！");
                }
                CalculateY_hat(chromosomeMatrixDouble, function);
            }
            double[] sumDeviation = new double[chromosomeNum];
            double[] adaptability = new double[chromosomeNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                sumDeviation[i] = CalculateSumDeviation(Y, Y_hat[i]);
                adaptability[i] = 1.0 / sumDeviation[i];
            }

            // 用来求解计算过程中最佳的染色体
            int index = -1;
            double maxAdaptability = double.MinValue;

            for (int i = 0; i < chromosomeNum; i++)
            {
                if (adaptability[i] > maxAdaptability)
                {
                    maxAdaptability = adaptability[i];
                    index = i;
                }
            }

            if (encodeType == "Binary")
            {
                bestChromosome = chromosomeMatrix[index];
            }
            else
            {
                bestChromosomeDouble = chromosomeMatrixDouble[index];
            }

            return adaptability;
        }

        /// <summary>
        /// 计算Y_hat
        /// </summary>
        /// <param name="chromosomeMatrix">染色体矩阵</param>
        /// <param name="function">计算函数 使用Lambda表达式</param>
        /// <returns>Y_hat</returns>
        private void CalculateY_hat(string[][] chromosomeMatrix, Func<double[], double[], double> function)
        {
            for (int i = 0; i < chromosomeNum; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Y_hat[i][j] = function(X[j], DecodeChromosome(chromosomeMatrix[i]));
                }
            }
        }

        private void CalculateY_hat(double[][] chromosomeMatrix, Func<double[], double[], double> function)
        {
            for (int i = 0; i < chromosomeNum; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Y_hat[i][j] = function(X[j], chromosomeMatrix[i]);
                }
            }
        }

        private double[] DecodeChromosome(string[] strings)
        {
            if (strings.Length != K)
            {
                throw new Exception("染色体长度不正确");
            }
            double[] param = new double[K];
            for (int i = 0; i < K; i++)
            {
                param[i] = Decode(strings[i]);
            }
            return param;
        }

        private double CalculateSumDeviation(double[] Y1, double[] Y2)
        {
            if (Y1.Length != Y2.Length || Y1.Length != N)
            {
                throw new Exception("Y1和Y2的长度不一致或长度不等于N");
            }
            double sumDeviation = 0.0;
            for (int i = 0; i < N; i++)
            {
                sumDeviation += Math.Abs(Y1[i] - Y2[i]);
            }
            return sumDeviation;
        }
    }
}
