namespace GA_OLD
{
    //void initRandomArray(int[] array)
    //{
    //    HashSet<int> set = new HashSet<int>();
    //    Random random = new Random();
    //    while (true)
    //    {
    //        if (set.Count == array.Length)
    //        {
    //            break;
    //        }
    //        set.Add(random.Next(1, array.Length + 1));
    //    }
    //    int[] setArray = set.ToArray();
    //    for (int i = 0; i < array.Length; i++)
    //    {
    //        array[i] = setArray[i];
    //    }
    //}


    //int[] tasks = new int[100];
    //int[] nodes = new int[3];

    //initRandomArray(tasks);
    //initRandomArray(nodes);

    //GeneticAlgorithm ga = new GeneticAlgorithm(tasks, nodes, 10000, 5, 0.6);

    public class GeneticAlgorithm
    {
        private int taskNum; // 任务数
        private int nodeNum; // 节点数
        private int iteratorNum; // 迭代次数
        private int chromosomeNum; // 染色体数量 
        private double cp; // 染色体复制的比例
        private int crossoverMutationNum; // 参与交叉变异的染色体数量 = chromosomeNum - adaptabilityMaxNum
        private int reserveNum; // 应该保留复制的N条染色体 = chromosomeNum * cp

        private int[] tasks; // 任务数组
        private int[] nodes; // 节点数组
        private double[][]? timeMatrix; // 时间矩阵
        private int[][]? chromosomeMatrix; // 染色体矩阵，每一行为一条染色体，chromosomeMatrix[i][j]=k表示第i条染色体中的任务j分配给节点k
        private double[]? timeArray_oneIt; // 每条染色体的任务长度
        private double[]? adaptability; // 适应度矩阵(下标：染色体编号、值：该染色体的适应度)
        private double[]? selectionProbability; // 自然选择的概率矩阵(下标：染色体编号、值：该染色体被选择的概率)

        private List<double[]> resultTimeArray_oneIt = []; // 任务长度结果集
        private List<double[]> resultAdaptability = []; // 适应度结果集
        private List<double[]> resultSelectionProbability = []; // 自然选择的概率矩阵结果集
        private List<int[][]> resultChromosomeMatrix = []; // 染色体矩阵结果集

        public GeneticAlgorithm(int[] tasks, int[] nodes, int iteratorNum, int chromosomeNum, double cp)
        {
            this.tasks = tasks;
            this.nodes = nodes;
            this.taskNum = tasks.Length;
            this.nodeNum = nodes.Length;
            this.iteratorNum = iteratorNum;
            this.chromosomeNum = chromosomeNum;
            this.cp = cp;
            this.crossoverMutationNum = chromosomeNum - (int)(chromosomeNum * cp);
            this.reserveNum = (int)(chromosomeNum * cp);

            printArray(tasks, "tasks:");
            printArray(nodes, "nodes:");

            this.ga();
        }

        public void ga()
        {
            // 初始化任务执行时间矩阵
            this.timeMatrix = this.initTimeMatrix(this.tasks, this.nodes);
            printMatrix(timeMatrix, "timeMatrix:");

            // 迭代搜索
            gaSearch(this.iteratorNum);

            // 结果输出
            Console.WriteLine("----------------start output---------------");
            for (int i = 0; i < this.resultTimeArray_oneIt.Count; i++)
            {
                if (i == 0 || i == this.resultTimeArray_oneIt.Count - 1)
                {
                    printArray(this.resultTimeArray_oneIt[i], "TimeArray_oneIt:" + (i + 1).ToString());
                    printArray(this.resultAdaptability[i], "Adaptability:" + (i + 1).ToString());
                    printArray(this.resultSelectionProbability[i], "SelectionProbability:" + (i + 1).ToString());
                    printMatrix(this.resultChromosomeMatrix[i], "ChromosomeMatrix:" + (i + 1).ToString());
                    Console.WriteLine("-------------------------------");
                }
            }
            Console.WriteLine(this.resultTimeArray_oneIt.Count);
            Console.WriteLine(this.resultAdaptability.Count);
            Console.WriteLine(this.resultSelectionProbability.Count);
            Console.WriteLine(this.resultChromosomeMatrix.Count);

        }

        private double[][] initTimeMatrix(int[] tasks, int[] nodes)
        {
            double[][] timeMatrix = new double[tasks.Length][];
            for (int i = 0; i < tasks.Length; i++)
            {
                timeMatrix[i] = new double[nodes.Length];
                for (int j = 0; j < nodes.Length; j++)
                {
                    timeMatrix[i][j] = (double)tasks[i] / (double)nodes[j];
                }
            }
            return timeMatrix;
        }

        private void gaSearch(int iteratorNum)
        {
            // 初始化第一代染色体
            this.chromosomeMatrix = createGeneration(null);
            printMatrix(chromosomeMatrix, "chromosomeMatrix:first");

            // 迭代繁衍
            for (int itIndex = 0; itIndex < iteratorNum; itIndex++)
            {
                // 计算上一代各条染色体的适应度
                this.adaptability = calAdaptability(this.chromosomeMatrix);

                // 计算自然选择概率
                this.selectionProbability = calSelectionProbability(this.adaptability);

                // 生成新一代染色体
                this.chromosomeMatrix = createGeneration(this.chromosomeMatrix);
            }
        }

        /**
         * 繁衍新一代染色体
         * @param chromosomeMatrix 上一代染色体
         */
        private int[][] createGeneration(int[][]? chromosomeMatrix)
        {
            int[][] newChromosomeMatrix = new int[this.chromosomeNum][];
            // 第一代染色体，随机生成
            if (chromosomeMatrix == null)
            {
                Random random = new Random();
                for (var chromosomeIndex = 0; chromosomeIndex < this.chromosomeNum; chromosomeIndex++)
                {
                    newChromosomeMatrix[chromosomeIndex] = new int[this.taskNum];
                    for (var taskIndex = 0; taskIndex < this.taskNum; taskIndex++)
                    {
                        newChromosomeMatrix[chromosomeIndex][taskIndex] = random.Next(0, this.nodeNum);
                    }
                }
                /*// 计算当前染色体的任务处理时间
                this.timeArray_oneIt = calTime_oneIt(newChromosomeMatrix);

                // 记录运行结果
                this.resultTimeArray_oneIt.Add(this.timeArray_oneIt);
                this.resultChromosomeMatrix.Add(newChromosomeMatrix);*/

                return newChromosomeMatrix;
            }
            // 交叉生成{crossoverMutationNum}条染色体
            int[][] newCrossMutationChromosomeMatrix = cross(chromosomeMatrix);

            // 变异
            newCrossMutationChromosomeMatrix = mutation(newCrossMutationChromosomeMatrix);

            // 复制
            newChromosomeMatrix = copy(chromosomeMatrix, newCrossMutationChromosomeMatrix);

            // 记录运行结果
            this.resultChromosomeMatrix.Add(newChromosomeMatrix);

            return newChromosomeMatrix;
        }

        /**
         * 计算所有染色体的任务处理时间
         * @param chromosomeMatrix
         */
        private double[] calTime_oneIt(int[][] chromosomeMatrix)
        {
            // 计算每条染色体的任务长度
            double[] timeArray_oneIt = new double[this.chromosomeNum];
            for (int chromosomeIndex = 0; chromosomeIndex < this.chromosomeNum; chromosomeIndex++)
            {
                double sumLength = 0.0;

                // 计算一条染色体处理任务所需的时间
                for (int taskIndex = 0; taskIndex < this.taskNum; taskIndex++)
                {
                    int nodeIndex = chromosomeMatrix[chromosomeIndex][taskIndex];
                    sumLength += this.timeMatrix[taskIndex][nodeIndex];
                }

                timeArray_oneIt[chromosomeIndex] = sumLength;
            }
            return timeArray_oneIt;
        }

        /**
         * 计算 染色体适应度
         * @param chromosomeMatrix
         * 这里可能和上面那个计算任务长度的函数存在冗余处理，可能需要进一步优化
         */
        private double[] calAdaptability(int[][] chromosomeMatrix)
        {
            // 计算每条染色体的适应度
            double[] adaptability = new double[this.chromosomeNum];

            // 计算每条染色体的任务长度
            double[] timeArray_oneIt = calTime_oneIt(chromosomeMatrix);

            // 适应度 = 1/任务长度
            for (int chromosomeIndex = 0; chromosomeIndex < this.chromosomeNum; chromosomeIndex++)
            {
                adaptability[chromosomeIndex] = 1.0 / timeArray_oneIt[chromosomeIndex];
            }

            // 记录运行结果
            this.timeArray_oneIt = timeArray_oneIt;
            this.resultTimeArray_oneIt.Add(timeArray_oneIt);
            this.resultAdaptability.Add(adaptability);

            return adaptability;
        }

        /**
         * 计算自然选择概率
         * @param adaptability
         */
        private double[] calSelectionProbability(double[] adaptability)
        {
            double[] selectionProbability = new double[this.chromosomeNum];

            // 计算适应度总和
            double sumAdaptability = 0;
            for (var i = 0; i < this.chromosomeNum; i++)
            {
                sumAdaptability += adaptability[i];
            }

            // 计算每条染色体的选择概率
            for (var i = 0; i < this.chromosomeNum; i++)
            {
                selectionProbability[i] = adaptability[i] / sumAdaptability;
            }

            // 记录运行结果
            this.resultSelectionProbability.Add(selectionProbability);

            return selectionProbability;
        }

        /**
         * 交叉生成{crossoverMutationNum}条染色体
         * @param chromosomeMatrix 上一代染色体矩阵
         */
        private int[][] cross(int[][] chromosomeMatrix)
        {
            Random random = new Random();
            int[][] newChromosomeMatrix = new int[this.crossoverMutationNum][];

            for (int i = 0; i < this.crossoverMutationNum; i++)
            {
                newChromosomeMatrix[i] = new int[this.taskNum];

                // 采用轮盘赌选择父母染色体
                int[] chromosomeBaba = chromosomeMatrix[RWS(this.selectionProbability)];
                int[] chromosomeMama = chromosomeMatrix[RWS(this.selectionProbability)];

                // 交叉
                var crossIndex = random.Next(0, this.taskNum);
                for (var j = 0; j < this.taskNum; j++)
                {
                    newChromosomeMatrix[i][j] = (j < crossIndex) ? chromosomeBaba[j] : chromosomeMama[j];
                }
            }
            return newChromosomeMatrix;
        }

        /**
         * 变异
         * @param newChromosomeMatrix 新一代染色体矩阵
         */
        private int[][] mutation(int[][] newChromosomeMatrix)
        {
            Random random = new Random();
            // 随机找一个任务
            var taskIndex = random.Next(0, this.taskNum);
            // 随机找一条染色体
            int chromosomeIndex = random.Next(0, this.crossoverMutationNum);

            // 变异主过程
            int oldNodeIndex = newChromosomeMatrix[chromosomeIndex][taskIndex];
            // 随机找一个节点
            int nodeIndex = random.Next(0, this.nodeNum);
            while (nodeIndex == oldNodeIndex) // 确保变异到位
            {
                nodeIndex = random.Next(0, this.nodeNum);
            }
            // 赋值
            newChromosomeMatrix[chromosomeIndex][taskIndex] = nodeIndex;

            return newChromosomeMatrix;
        }

        /**
         * 复制(复制上一代中优良的染色体)
         * @param chromosomeMatrix 上一代染色体矩阵
         * @param newChromosomeMatrix 新一代染色体矩阵
         */
        private int[][] copy(int[][] chromosomeMatrix, int[][] newCrossMutationChromosomeMatrix)
        {
            // 寻找适应度最高(选择率最高的)的N条染色体的下标(N=染色体数量*复制比例)
            int[] chromosomeIndexArr = maxN(this.selectionProbability, this.reserveNum);

            int[][] newChromosomeMatrix = new int[this.chromosomeNum][];

            // 复制
            for (int i = 0; i < this.chromosomeNum; i++)
            {
                newChromosomeMatrix[i] = new int[this.taskNum];
                // 复制数组
                int[] chromosome;
                if (i < this.reserveNum)
                {
                    chromosome = chromosomeMatrix[chromosomeIndexArr[i]];
                }
                else// 交叉变异数组
                {
                    chromosome = newCrossMutationChromosomeMatrix[i - this.reserveNum];
                }

                for (int j = 0; j < chromosome.Length; j++)
                {
                    newChromosomeMatrix[i][j] = chromosome[j];
                }
            }

            return newChromosomeMatrix;
        }

        /**
         * 从数组中寻找最大的n个元素
         * @param array
         * @param 原数组中最大的元素所在的下标
         */
        private int[] maxN(double[] array, int n)
        {
            KeyValuePair<double, int>[] map = new KeyValuePair<double, int>[array.Length];
            // 将一切数组升级成二维数组，二维数组的每一行都有两个元素构成[原一位数组的下标,值]
            for (var i = 0; i < array.Length; i++)
            {
                map[i] = KeyValuePair.Create(array[i], i);
            }

            // 对二维数组排序
            KeyValuePair<double, int>[] sortedMap = map.OrderByDescending(pair => pair.Key).ToArray();

            // 取最大的n个元素
            int[] maxIndexArray = new int[n];
            for (var i = 0; i < n; i++)
            {
                maxIndexArray[i] = sortedMap[i].Value;
            }

            return maxIndexArray;
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

        private void printArray<T>(T[] array, string msg)
        {
            Console.WriteLine(msg);
            for (int i = 0; i < array.Length; i++)
            {
                Console.Write(array[i]);
                Console.Write(',');
            }
            Console.WriteLine();
        }

        private void printMatrix<T>(T[][] matrix, string msg)
        {
            Console.WriteLine(msg);
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    Console.Write(matrix[i][j]);
                    Console.Write(',');
                }
                Console.WriteLine();
            }
        }
    }
}