# 遗传算法（Genetic Algorithm）

## 介绍
遗传算法是一种基于进化的优化算法，它通过模拟自然界的进化过程，在一定条件下，能够找到最优解。

遗传算法的基本思想是，通过一定的变异和交叉，产生新的种群，并通过适应度函数来评估这些种群的适应度，从而选择出适应度最高的个体作为新的父代，并将其作为下一代种群的基因。

遗传算法的优点是可以解决复杂的优化问题，并在一定时间内找到全局最优解，但同时也存在一些局限性，比如需要预先设定问题的解空间、目标函数、搜索空间等，并且需要多次迭代才能找到全局最优解。

## 算法步骤
1. 初始化种群：随机生成初始种群，每个个体的基因由随机的初始值表示。
2. 适应度函数：计算每个个体的适应度，适应度越高，个体的概率越大被选中作为父代。
3. 选择：以适应度为选择概率,随机的抽取一定数量的个体作为父代。
4. 交叉：将父代个体的基因进行交叉，产生新的子代个体。
5. 变异：将子代个体的基因进行变异，产生新的种群。
6. 重复步骤2-5，直到收敛。

## 适用场景
我写的此版本的遗传算法适合用来 场景: **未知函数求参**

### 未知函数求参
已知函数y = f(x1, x2, x3,..., xn),其中有多个未知参数param = [param1, param2, param3,..., paramn]

有一组初始值[X_1, X_2, X_3,..., X_k]（其中X_*为1*n的向量）, [y1, y2, y3,..., yn]，对于任意X_*, 有y* = f(X_*)

算法求解目标：求出未知参数[param1, param2, param3,..., paramn]在指定范围内的最优解


## 使用方法
### 示例代码：

```csharp
double[][] X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
double[] Y = [121, 441, 961, 1681, 2601, 3721, 5041, 6561, 8281, 10201];

Func<double[], double[], double> function = (x, param) => // 要优化的函数
{
        if (x.Length != 1 || param.Length != 2)
        {
            throw new Exception("invalid input");
        }
        return param[1] * x[0] * x[0] + 2 * param[0] * x[0] + 1;
    };
GeneticAlgorithm ga = new GeneticAlgorithm( // 初始化遗传算法实例
    X: x3,
    Y: y3,
    K: 2,
    minValue: [8, 90],
    maxValue: [11, 110],
    iteratorNum: 100,
    chromosomeNum: 1000,
    crossoverRate: 0.5,
    mutationRate: 0.05,
    function: function,
    encodeType: "Binary",
    accuracy: 0.000001,
    geneticStrategy: "best", // "best" or "random"
    reserveRate: 0.1,
    returnType: "Global");
double[] res = ga.Run(); // 运行遗传算法
for (int i = 0; i < res.Length; i++) // 输出结果
{
    Console.WriteLine(res[i]);
}
```
在上面这段示例代码中，X，Y为已知的一组数据，其中对于每个元素X[i], Y[i]都有`Y[i] = f(X[i])`

function为未知函数，其输入为函数的变量和参数param，输出为函数的预测值y^

在这里，function为一元方程，未知参数的长度为2，此算法可以手动定义更加高维的函数，只需要传递正确的X，Y，function即可

### 参数详解
```
X:输入参数矩阵
Y:输出结果向量
K:需要优化的参数个数
minValue:参数取值范围下限 数组类型，指明每一个变量的范围 
maxValue:参数取值范围上限 数组类型，指明每一个变量的范围
iteratorNum:迭代次数
chromosomeNum:染色体数量
crossoverRate:交叉概率
mutationRate:变异概率
function:计算函数
accuracy:精度 （Bianry编码有效）
crossType:交叉方式 single/twoPoint/uniform （Bianry编码有效）
mutationType:变异方式 single/uniform （Bianry编码有效）
encodeType:编码方式 Binary/Double 二进制编码/双精度编码
geneticStrategy:遗传策略 random/best 随机遗传策略/最佳保留策略
reserveRate:最佳保留率 只有在geneticStrategy为best时才有效
returnType:返回值类型 分为全局最优和局部最优 Local/Global 经过测试，局部最优效果更好
```

## 算法功能介绍
### 编码方式：encodeType
#### 1. Binary编码

Binary编码是指将自变量和因变量都编码为01字符串，适用于离散型变量。速度较double编码慢，好处是可以指定精度以及使用多种交叉，变异方式。

#### Double编码
Double编码是指将自变量和因变量都编码为实数，适用于连续型变量。速度较binary编码快,不可指定精度、交叉, 变异方式，

好处是运行速度比较快

### 交叉方式：crossType（Bianry编码有效）
使用crossoverRate指定交叉率

#### 1. single
单点交叉是指在两个染色体之间随机选择一个交叉点，将两个染色体的基因序列分成两段，返回：父本的前半段+母本的后半段接。

#### 2. twoPoint
两点交叉是指在两个染色体之间随机选择两个交叉点，将两个染色体的基因序列分成两段，返回：父本的前半段+母本的中间段+父本的后半段。

#### 3. uniform
均匀交叉是指在父本和母本之间随机选取基因作为下一代染色体。


### 变异方式：mutationType（Bianry编码有效）
使用mutationRate指定变异率

#### single
单点变异是指在染色体的某个基因上随机选取一个位置，将该位置的基因进行变异。

#### uniform
均匀变异是指在染色体的每个基因都有几率发生变异。

### 遗传策略：geneticStrategy
#### random
随机遗传策略是指在每一代中，随机选择父本和母本，产生子代。

#### best
使用reserveRate指定最佳保留率

最佳保留策略是指在每一代中，按照一定的比例保留适应度最高的一些个体作为保留样本，同时使用随机策略选择其他个体，两者共同作为子代。

### 返回值类型：returnType  
#### Local
局部最优是指返回最后一代中适应度最高的个体作为最优解。

#### Global
全局最优是指记录迭代过程中每一代的最优解，并返回结果中适应度最高的个体作为最优解。

这种方法可能会返回局部最优解。因此建议使用Local类型。
