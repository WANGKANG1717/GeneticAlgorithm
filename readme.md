# 遗传算法（Genetic Algorithm）

## 介绍
是一种基于进化的优化算法，
它通过模拟自然界的进化过程，在一定条件下，能够找到最优解。
遗传算法的基本思想是，通过一定的变异和交叉，产生新的种群，并通过适应度函数来评估这些种群的适应度，
从而选择出适应度最高的个体作为新的父代，并将其作为下一代种群的基因。
遗传算法的优点是可以解决复杂的优化问题，并在一定时间内找到全局最优解，但同时也存在一些局限性，
比如需要预先设定问题的解空间、目标函数、搜索空间等，并且需要多次迭代才能找到全局最优解。

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
已知函数y = f(x1, x2, x3,..., xn),其中有多个未知参数[param1, param2, param3,..., paramn]<br />
有一组初始值[x1_0, x2_0, x3_0,..., xn_0], [y1, y2, y3,..., yn]<br />
目标：求出未知参数[param1, param2, param3,..., paramn]的最优解<br />

## 使用方法
已知:
X = [x1, x2, x3,..., xn], y = [y1, y2, y3,..., yn]<br />
其中 x1 = [1, 2, 3, 4, 5], y1 = [10]<br />
y=function(x)的结构,如y = param[0] * Math.Sin(x[0]) + param[1] * Math.Cos(x[1]) + param[2] / x[2] + param[3]<br />
其中param为未知参数,x为自变量,y为因变量<br />
使用如下代码即可进行遗传算法的训练:<br />
GeneticAlgorithm ga = new GeneticAlgorithm(<br />
&nbsp;&nbsp;&nbsp;&nbsp;X: X, // 自变量<br />
&nbsp;&nbsp;&nbsp;&nbsp;Y: Y, // 因变量<br />
&nbsp;&nbsp;&nbsp;&nbsp;K: 4, // 未知参数数量<br />
&nbsp;&nbsp;&nbsp;&nbsp;minValue: 1, // 未知参数最小值<br />
&nbsp;&nbsp;&nbsp;&nbsp;maxValue: 10, // 未知参数最大值<br />
&nbsp;&nbsp;&nbsp;&nbsp;iteratorNum: 100000, // 迭代次数<br />
&nbsp;&nbsp;&nbsp;&nbsp;chromosomeNum: 100, // 种群数量<br />
&nbsp;&nbsp;&nbsp;&nbsp;crossoverRate: 0.6, // 交叉概率<br />
&nbsp;&nbsp;&nbsp;&nbsp;mutationRate: 0.01, // 变异概率<br />
&nbsp;&nbsp;&nbsp;&nbsp;function: function, // 目标函数<br />
&nbsp;&nbsp;&nbsp;&nbsp;encodeType: "Double" // 编码方式 Double/Binary<br />
);<br />
double[] res = ga.Run(); // 运行遗传算法<br />
for (int i = 0; i < res.Length; i++) // 输出结果<br />
{<br />
    Console.WriteLine(res[i]);<br />
}<br />

## 注意事项
此遗传算法的编码方式有两种，Double编码和Binary编码，Double编码是指将自变量和因变量都编码为实数

### Binary编码
Binary编码是指将自变量和因变量都编码为0或1，适用于离散型变量。<br />
速度较double编码慢,但是可以使用多种交叉,变异方式<br />

### Double编码
Double编码是指将自变量和因变量都编码为实数，适用于连续型变量。<br />
速度较binary编码快,不可指定交叉, 变异方式<br />