/**
 * @date: 2024-03-04 21:11:41
 * @author: WangKang
 * @blog: https://wangkang1717.github.io
 * @email: 1686617586@qq.com
 * @filepath: GeneticAlgorithm.cs
 * @description: 遗传算法
 * Copyright 2024 WANGKANG, All Rights Reserved.
 */

using System.Diagnostics;

namespace GA
{
    public class GeneticAlgorithm
    {
        private string encodeType = "Binary"; // 编码方式 Binary / Double
        private string geneticStrategy = "random"; // 遗传策略 random: 随机策略 best：最优保留

        private double[][] X;
        private double[] Y;
        private double[][]? Y_hat; // 行为chromosomeNum，列为N
        private int N; // =X的行数 / =Y的行数
        private int M; // =X的列数
        private int K; // 需要优化的参数个数
        private double[] minValue; // 参数取值范围
        private double[] maxValue;
        private int iteratorNum; // 迭代次数
        private int chromosomeNum; // 染色体数量
        private double mutationRate; // 变异概率
        private double crossoverRate; // 交叉概率
        // 最优保留策略相关参数
        private double reserveRate; // 保留概率
        private int reserveNum; // 保留数量 根据reserveRate计算得到

        private string[][]? chromosomeMatrix; // 染色体矩阵 二进制编码

        private double[][]? chromosomeMatrixDouble; // 染色体矩阵 浮点数编码

        private double accuracy = 0.001; // 精度
        private string crossType; // 交叉方式 single/twoPoint/uniform
        private string mutationType; // 变异方式 single/uniform
        private int[] numberOfBits; // 二进制编码的位数 需要根据需要的精度进行动态计算

        private Func<double[], double[], double> function; // 计算函数 使用Lambda表达式

        private double maxAdaptability = double.MinValue;
        public string[]? bestChromosome; // 最大适应度 二进制编码 // 全局最优
        public double[]? bestChromosomeDouble; // 最大适应度 浮点数编码 // 全局最优

        public string[]? bestChromosomeLocal; // 最大适应度 二进制编码 // 局部最优
        public double[]? bestChromosomeDoubleLocal; // 最大适应度 浮点数编码 // 局部最优

        string returnType = "Local"; // 返回值类型 Local/Global

        /// WANGKANG 2024-05-08 15:56:48
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="X">输入参数矩阵</param>
        /// <param name="Y">输出结果向量</param>
        /// <param name="K">需要优化的参数个数</param>
        /// <param name="minValue">参数取值范围下限 数组类型，指明每一个变量的范围</param> 
        /// <param name="maxValue">参数取值范围上限 数组类型，指明每一个变量的范围</param>
        /// <param name="iteratorNum">迭代次数</param>
        /// <param name="chromosomeNum">染色体数量</param>
        /// <param name="crossoverRate">交叉概率</param>
        /// <param name="mutationRate">变异概率</param>
        /// <param name="function">计算函数</param>
        /// <param name="accuracy">精度</param>
        /// <param name="crossType">交叉方式 single/twoPoint/uniform</param>
        /// <param name="mutationType">变异方式 single/uniform</param>
        /// <param name="geneticStrategy">遗传策略 random/best</param>
        /// <param name="reserveRate">最佳保留率 只有在geneticStrategy为best时才有效</param>
        /// <param name="returnType">返回值类型 分为全局最优和局部最优 Local/Global 经过测试，局部最优效果更好</param>
        public GeneticAlgorithm(
            double[][] X,
            double[] Y,
            int K,
            double[] minValue,
            double[] maxValue,
            int iteratorNum,
            int chromosomeNum,
            double crossoverRate,
            double mutationRate,
            Func<double[], double[], double> function,
            double accuracy = 0.001, // 二进制编码有用
            string crossType = "single",
            string mutationType = "single",
            string encodeType = "Binary",
            string geneticStrategy = "random",
            double reserveRate = 0.1,
            string returnType = "Local")
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

            if (minValue.Length != maxValue.Length || minValue.Length != K)
            {
                throw new Exception("参数取值范围下限和上限的长度必须等于参数个数");
            }

            this.minValue = minValue;
            this.maxValue = maxValue;
            this.iteratorNum = iteratorNum;
            this.chromosomeNum = chromosomeNum;
            this.crossoverRate = crossoverRate;
            this.mutationRate = mutationRate;
            this.accuracy = accuracy;
            this.crossType = crossType;
            this.mutationType = mutationType;
            this.reserveRate = reserveRate;
            this.reserveNum = (int)(chromosomeNum * this.reserveRate);


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

            if (geneticStrategy == "random" || geneticStrategy == "best")
            {
                this.geneticStrategy = geneticStrategy;
            }
            else
            {
                throw new Exception("不支持的遗传策略(仅支持random和best)");
            }

            if (returnType == "Local" || returnType == "Global")
            {
                this.returnType = returnType;
            }
            else
            {
                throw new Exception("不支持的返回值类型(仅支持Local和Global类型)");
            }
        }

        private int[] calculateNumberOfBits()
        {
            int[] length = new int[K];
            for (int i = 0; i < K; i++)
            {
                int tmp_length = (int)Math.Ceiling(Math.Log2((maxValue[i] - minValue[i]) / accuracy + 1));
                length[i] = tmp_length;
                if (length[i] > 31)
                {
                    length[i] = 31;
                    Console.WriteLine("Warning: 二进制编码位数超过31位，将使用31位");
                }
            }
            return length;
        }

        private double Decode(string binaryString, int index)
        {
            if (binaryString.Length != numberOfBits[index])
            {
                throw new Exception("二进制编码长度不正确");
            }
            return minValue[index] + (maxValue[index] - minValue[index]) / (Math.Pow(2, numberOfBits[index]) - 1) * StringToNumber(binaryString);
        }

        public double[] Decode(string[]? chromosome)
        {
            if (chromosome == null)
            {
                throw new Exception("chromosome is null");
            }
            double[] res = new double[K];
            for (int i = 0; i < K; i++)
            {
                res[i] = Decode(chromosome[i], i);
            }
            return res;
        }

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
            string binaryString = "";
            for (int i = 0; i < length; i++)
            {
                binaryString += (randomNumber % 2).ToString();
                randomNumber /= 2;
            }
            binaryString = ReverseString(binaryString);
            return binaryString;
        }

        private string ReverseString(string str)
        {
            return new string(str.ToCharArray().Reverse().ToArray());
        }

        private int RandomNumber(int length)
        {
            Random random = new Random();
            return random.Next(0, (int)(Math.Pow(2, length) - 1));
        }

        private object[] reserveBestChromosome(string[][]? chromosomeMatrix, double[] naturalSelectionRate)
        {
            if (chromosomeMatrix == null)
            {
                throw new Exception("chromosomeMatrix is null");
            }
            if (chromosomeMatrix.Length != chromosomeNum)
            {
                throw new Exception("染色体数量不正确");
            }
            KeyValuePair<int, double>[] index_selectRate = new KeyValuePair<int, double>[chromosomeNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                index_selectRate[i] = new KeyValuePair<int, double>(i, naturalSelectionRate[i]);
            }
            Array.Sort(index_selectRate, (x, y) => y.Value.CompareTo(x.Value));

            List<string[]> bestReserveChromosomeMatrix = new List<string[]>();
            List<string[]> chromosomeMatrixTmp = new List<string[]>();

            double[] newNaturalSelectionRate = new double[chromosomeNum - reserveNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                if (i < reserveNum)
                    bestReserveChromosomeMatrix.Add(chromosomeMatrix[index_selectRate[i].Key]);
                else
                {
                    chromosomeMatrixTmp.Add(chromosomeMatrix[index_selectRate[i].Key]);
                    newNaturalSelectionRate[i - reserveNum] = index_selectRate[i].Value;
                }
            }
            object[] result = [bestReserveChromosomeMatrix.ToArray(), chromosomeMatrixTmp.ToArray(), newNaturalSelectionRate];
            return result;
        }

        public double[] Run()
        {
            Console.WriteLine("########### start ###########");
            Stopwatch stopwatch = new Stopwatch();
            // 开始计时
            stopwatch.Start();

            GenerateFirstGeneration();
            // 迭代繁衍
            for (int itIndex = 0; itIndex < iteratorNum; itIndex++)
            {
                // 计算各条染色体的适应度
                double[] adaptability = CalculateAdaptability();
                double[] naturalSelectionRate = CalculateNaturalSelectionRate(adaptability);
                if (encodeType == "Binary")
                {
                    if (geneticStrategy == "random")
                    {
                        string[][] newChromosomeMatrix = SelectCrossMutation(chromosomeMatrix, naturalSelectionRate, itIndex);
                        // 更新chromosomeMatrix
                        chromosomeMatrix = newChromosomeMatrix;
                        if (chromosomeMatrix.Length != chromosomeNum)
                        {
                            throw new Exception("染色体数量不正确");
                        }
                    }
                    else if (geneticStrategy == "best")
                    {
                        object[] result = reserveBestChromosome(chromosomeMatrix, naturalSelectionRate);
                        string[][] bestReserveChromosomeMatrix = (string[][])result[0];
                        chromosomeMatrix = (string[][])result[1];
                        naturalSelectionRate = (double[])result[2];

                        string[][] newChromosomeMatrix = SelectCrossMutation(chromosomeMatrix, naturalSelectionRate, itIndex);

                        chromosomeMatrix = mergeChromosomeMatrix(bestReserveChromosomeMatrix, newChromosomeMatrix);
                        if (chromosomeMatrix.Length != chromosomeNum)
                        {
                            throw new Exception("染色体数量不正确");
                        }
                    }
                    else
                    {
                        throw new Exception("不支持的遗传策略");
                    }
                }
                else if (encodeType == "Double")
                {
                    if (geneticStrategy == "random")
                    {
                        double[][] newChromosomeMatrixDouble = SelectCrossMutationDouble(chromosomeMatrixDouble, naturalSelectionRate, itIndex);

                        chromosomeMatrixDouble = newChromosomeMatrixDouble;
                        if (chromosomeMatrixDouble.Length != chromosomeNum)
                        {
                            throw new Exception("染色体数量不正确");
                        }
                    }
                    else if (geneticStrategy == "best")
                    {
                        object[] result = reserveBestChromosomeDouble(chromosomeMatrixDouble, naturalSelectionRate);
                        double[][] bestReserveChromosomeMatrixDouble = (double[][])result[0];
                        chromosomeMatrixDouble = (double[][])result[1];
                        naturalSelectionRate = (double[])result[2];

                        double[][] newChromosomeMatrixDouble = SelectCrossMutationDouble(chromosomeMatrixDouble, naturalSelectionRate, itIndex);

                        chromosomeMatrixDouble = mergeChromosomeMatrixDouble(bestReserveChromosomeMatrixDouble, newChromosomeMatrixDouble);
                        if (chromosomeMatrixDouble.Length != chromosomeNum)
                        {
                            throw new Exception("染色体数量不正确");
                        }
                    }
                    else
                    {
                        throw new Exception("不支持的遗传策略");
                    }
                }
                else
                {
                    throw new Exception("不支持的类型(仅支持Binary和Double类型)");
                }
            }

            CalculateAdaptability();

            // 停止计时
            stopwatch.Stop();
            // 打印运行时间（hh:mm:ss）
            Console.WriteLine("program running time: " + stopwatch.Elapsed.ToString(@"hh\:mm\:ss"));
            Console.WriteLine("########### end ###########");
            // 返回值
            if (encodeType == "Binary")
            {
                if (returnType == "Local")
                {
                    if (bestChromosomeLocal == null)
                    {
                        throw new Exception("bestChromosomeLocal is null");
                    }
                    return Decode(bestChromosomeLocal);
                }
                else if (returnType == "Global")
                {
                    if (bestChromosome == null)
                    {
                        throw new Exception("bestChromosome is null");
                    }
                    return Decode(bestChromosome);
                }
                else
                {
                    throw new Exception("不支持的返回值类型");
                }
            }
            else if (encodeType == "Double")
            {
                if (returnType == "Local")
                {
                    if (bestChromosomeDoubleLocal == null)
                    {
                        throw new Exception("bestChromosomeDoubleLocal is null");
                    }
                    return bestChromosomeDoubleLocal;
                }
                else if (returnType == "Global")
                {
                    if (bestChromosomeDouble == null)
                    {
                        throw new Exception("bestChromosomeDouble is null");
                    }
                    return bestChromosomeDouble;
                }
                else
                {
                    throw new Exception("不支持的返回值类型");
                }
            }
            else
            {
                throw new Exception("不支持的类型(仅支持Binary和Double类型)");
            }
        }

        private object[] reserveBestChromosomeDouble(double[][]? chromosomeMatrixDouble, double[] naturalSelectionRate)
        {
            if (chromosomeMatrixDouble == null)
            {
                throw new Exception("chromosomeMatrixDouble is null");
            }
            if (chromosomeMatrixDouble.Length != chromosomeNum)
            {
                throw new Exception("染色体数量不正确");
            }
            KeyValuePair<int, double>[] index_selectRate = new KeyValuePair<int, double>[chromosomeNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                index_selectRate[i] = new KeyValuePair<int, double>(i, naturalSelectionRate[i]);
            }
            Array.Sort(index_selectRate, (x, y) => y.Value.CompareTo(x.Value));

            List<double[]> bestReserveChromosomeMatrixDouble = new List<double[]>();
            List<double[]> chromosomeMatrixDoubleTmp = new List<double[]>();

            double[] newNaturalSelectionRate = new double[chromosomeNum - reserveNum];
            for (int i = 0; i < chromosomeNum; i++)
            {
                if (i < reserveNum)
                    bestReserveChromosomeMatrixDouble.Add(chromosomeMatrixDouble[index_selectRate[i].Key]);
                else
                {
                    chromosomeMatrixDoubleTmp.Add(chromosomeMatrixDouble[index_selectRate[i].Key]);
                    newNaturalSelectionRate[i - reserveNum] = index_selectRate[i].Value;
                }
            }
            object[] result = [bestReserveChromosomeMatrixDouble.ToArray(), chromosomeMatrixDoubleTmp.ToArray(), newNaturalSelectionRate];
            return result;
        }

        private double[][] mergeChromosomeMatrixDouble(double[][] bestReserveChromosomeMatrixDouble, double[][] newChromosomeMatrixDouble)
        {
            if (bestReserveChromosomeMatrixDouble == null)
            {
                throw new Exception("bestReserveChromosomeMatrixDouble is null");
            }
            List<double[]> chromosomeMatrixDoubleTmp = new List<double[]>();
            for (int i = 0; i < bestReserveChromosomeMatrixDouble.Length; i++)
            {
                chromosomeMatrixDoubleTmp.Add(bestReserveChromosomeMatrixDouble[i]);
            }
            for (int i = 0; i < newChromosomeMatrixDouble.Length; i++)
            {
                chromosomeMatrixDoubleTmp.Add(newChromosomeMatrixDouble[i]);
            }
            return chromosomeMatrixDoubleTmp.ToArray();
        }

        private double[][] SelectCrossMutationDouble(double[][]? chromosomeMatrixDouble, double[] naturalSelectionRate, int itIndex)
        {
            if (chromosomeMatrixDouble == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            double[][] newChromosomeMatrixDouble;
            newChromosomeMatrixDouble = SelectDouble(chromosomeMatrixDouble, naturalSelectionRate);
            newChromosomeMatrixDouble = CrossDouble(newChromosomeMatrixDouble);
            newChromosomeMatrixDouble = MutationDouble(newChromosomeMatrixDouble, itIndex);
            return newChromosomeMatrixDouble;
        }

        private string[][] SelectCrossMutation(string[][]? chromosomeMatrix, double[] naturalSelectionRate, int itIndex)
        {
            string[][] newChromosomeMatrix;
            newChromosomeMatrix = Select(chromosomeMatrix, naturalSelectionRate);
            newChromosomeMatrix = Cross(newChromosomeMatrix);
            newChromosomeMatrix = Mutation(newChromosomeMatrix, itIndex);
            return newChromosomeMatrix;
        }

        private string[][] mergeChromosomeMatrix(string[][]? bestReserveChromosome, string[][] newChromosomeMatrix)
        {
            if (bestReserveChromosome == null)
            {
                throw new Exception("bestReserveChromosome is null");
            }
            List<string[]> chromosomeMatrixTmp = new List<string[]>();
            for (int i = 0; i < bestReserveChromosome.Length; i++)
            {
                chromosomeMatrixTmp.Add(bestReserveChromosome[i]);
            }
            for (int i = 0; i < newChromosomeMatrix.Length; i++)
            {
                chromosomeMatrixTmp.Add(newChromosomeMatrix[i]);
            }
            return chromosomeMatrixTmp.ToArray();
        }

        private double[][] MutationDouble(double[][] newChromosomeMatrixDouble, int iterations)
        {
            Random random = new Random();
            for (int i = 0; i < newChromosomeMatrixDouble.Length; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    double rate = random.NextDouble();
                    if (rate <= mutationRate * Math.Pow(1 - iterations / (iteratorNum - 1), 2)) // 变异率应该逐步减少，，以达到稳定状态
                    {
                        newChromosomeMatrixDouble[i][j] = RandomMutationDouble(newChromosomeMatrixDouble[i][j], j);
                    }
                }
            }
            return newChromosomeMatrixDouble;
        }

        private double RandomMutationDouble(double val, int index)
        {
            Random random = new Random();
            return random.NextDouble() * (maxValue[index] - minValue[index]) + minValue[index];
        }

        private double[][] CrossDouble(double[][] newChromosomeMatrixDouble)
        {
            if (newChromosomeMatrixDouble == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            Random random = new Random();
            double[][] tmpChromosomeMatrix = new double[newChromosomeMatrixDouble.Length][];
            for (int i = 0; i < newChromosomeMatrixDouble.Length; i++)
            {
                double rate = random.NextDouble();
                // 判断该染色体是否需要交叉
                if (rate <= crossoverRate)
                {
                    // 当前染色体作为父本
                    int j = random.Next(0, newChromosomeMatrixDouble.Length); // 选择需要交叉的母本
                    tmpChromosomeMatrix[i] = RandomCrossDouble(newChromosomeMatrixDouble[i], newChromosomeMatrixDouble[j]);// 随机浮点数交叉
                }
                else // 不交叉，直接保留
                {
                    tmpChromosomeMatrix[i] = newChromosomeMatrixDouble[i];
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

        private double[][] SelectDouble(double[][]? chromosomeMatrixDouble, double[] naturalSelectionRate)
        {
            if (chromosomeMatrixDouble == null)
            {
                throw new Exception("染色体矩阵为空！");
            }
            if (naturalSelectionRate == null)
            {
                throw new Exception("自然选择率未计算！");
            }
            double[][] newChromosomeMatrix = new double[chromosomeMatrixDouble.Length][];
            int[] selectedIndex = SelectChromosomeIndex(naturalSelectionRate, chromosomeMatrixDouble.Length);
            for (int i = 0; i < chromosomeMatrixDouble.Length; i++)
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
            string[][] newChromosomeMatrix = new string[chromosomeMatrix.Length][];
            int[] selectedIndex = SelectChromosomeIndex(naturalSelectionRate, chromosomeMatrix.Length);
            for (int i = 0; i < chromosomeMatrix.Length; i++)
            {
                newChromosomeMatrix[i] = chromosomeMatrix[selectedIndex[i]];
            }
            return newChromosomeMatrix;
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
            Random random = new Random();
            string[][] tmpChromosomeMatrix = new string[newChromosomeMatrix.Length][];
            for (int i = 0; i < newChromosomeMatrix.Length; i++)
            {
                double rate = random.NextDouble();
                // 判断该染色体是否需要交叉
                if (rate <= crossoverRate)
                {
                    // 当前染色体作为父本
                    int j = random.Next(0, newChromosomeMatrix.Length); // 选择需要交叉的母本
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
        private string[] OnePointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                int point = random.Next(1, numberOfBits[k] - 1);
                chromosome[k] = chromosome1[k].Substring(0, point) + chromosome2[k].Substring(point);
            }
            return chromosome;
        }

        private string[] TwoPointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                int mid = numberOfBits[k] / 2;
                int point1 = random.Next(0, mid);
                int point2 = random.Next(mid, numberOfBits[k]);
                chromosome[k] = chromosome1[k].Substring(0, point1) + chromosome2[k].Substring(point1, point2 - point1) + chromosome1[k].Substring(point2);
            }
            return chromosome;
        }
        private string[] UniformPointCross(string[] chromosome1, string[] chromosome2)
        {
            Random random = new Random();
            string[] chromosome = new string[K];
            for (int k = 0; k < K; k++)
            {
                chromosome[k] = "";
                for (int i = 0; i < numberOfBits[k]; i++)
                {
                    chromosome[k] += random.NextDouble() <= 0.5 ? chromosome1[k][i] : chromosome2[k][i];
                }
            }
            return chromosome;
        }

        private string[][] Mutation(string[][] newChromosomeMatrix, int iterations)
        {
            Random random = new Random();
            for (int i = 0; i < newChromosomeMatrix.Length; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    double rate = random.NextDouble();
                    if (rate <= mutationRate * Math.Pow(1 - iterations / (iteratorNum - 1), 2)) // 变异率应该逐步减少，，以达到稳定状态
                    {
                        switch (mutationType)
                        {
                            case "single":
                                newChromosomeMatrix[i][j] = OnePointMutation(newChromosomeMatrix[i][j], j);
                                break;
                            case "uniform":
                                newChromosomeMatrix[i][j] = UniformPointMutation(newChromosomeMatrix[i][j], j);
                                break;
                            default:
                                newChromosomeMatrix[i][j] = OnePointMutation(newChromosomeMatrix[i][j], j);
                                break;
                        }
                    }
                }
            }
            return newChromosomeMatrix;
        }

        private string OnePointMutation(string chromosome, int index)
        {
            Random random = new Random();
            int point = random.Next(0, numberOfBits[index]);
            return chromosome.Substring(0, point) + (chromosome[point] ^ '1').ToString() + chromosome.Substring(point + 1);
        }

        private string UniformPointMutation(string chromosome, int index)
        {
            string newChromosome = "";
            Random random = new Random();
            for (int i = 0; i < numberOfBits[index]; i++)
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
                    newChromosomeMatrix[chromosomeIndex][i] = RandomBinaryString(numberOfBits[i]);
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
                    newChromosomeMatrix[chromosomeIndex][i] = RandomDouble(minValue[i], maxValue[i]);
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
            if (Y_hat == null)
            {
                throw new Exception("Y_hat未计算！");
            }
            for (int i = 0; i < chromosomeNum; i++)
            {
                sumDeviation[i] = CalculateSumDeviation(Y, Y_hat[i]);
                adaptability[i] = 1.0 / sumDeviation[i];
            }

            // 用来求解计算过程中最佳的染色体
            int index = -1;

            double maxAdaptabilityLocal = double.MinValue;
            int index_local = -1;
            for (int i = 0; i < chromosomeNum; i++)
            {
                // 计算全局最优解
                if (adaptability[i] > maxAdaptability)
                {
                    maxAdaptability = adaptability[i];
                    index = i;
                }
                // 计算局部最优解
                if (adaptability[i] > maxAdaptabilityLocal)
                {
                    maxAdaptabilityLocal = adaptability[i];
                    index_local = i;
                }
            }

            if (encodeType == "Binary")
            {
                if (chromosomeMatrix == null)
                {
                    throw new Exception("染色体矩阵为空！");
                }
                if (index != -1)
                {
                    bestChromosome = chromosomeMatrix[index];
                }
                if (index_local != -1)
                {
                    bestChromosomeLocal = chromosomeMatrix[index_local];
                }
            }
            else
            {
                if (chromosomeMatrixDouble == null)
                {
                    throw new Exception("染色体矩阵为空！");
                }
                if (index != -1)
                {
                    bestChromosomeDouble = chromosomeMatrixDouble[index];
                }
                if (index_local != -1)
                {
                    bestChromosomeDoubleLocal = chromosomeMatrixDouble[index_local];
                }
            }

            return adaptability;
        }

        private void CalculateY_hat(string[][] chromosomeMatrix, Func<double[], double[], double> function)
        {
            if (Y_hat == null)
            {
                throw new Exception("Y_hat未计算！");
            }
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
            if (Y_hat == null)
            {
                throw new Exception("Y_hat未计算！");
            }
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
                param[i] = Decode(strings[i], i);
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
