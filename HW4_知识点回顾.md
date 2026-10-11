# DSCI 552 HW4 知识点回顾

## 1. 这份作业在做什么

HW4 使用 AReM 传感器数据识别人类活动。每个实例有 6 条时间序列，先把每条序列压缩成统计特征，再用分类模型判断活动类型。

本作业主要比较：

- 普通逻辑回归和 L1 正则逻辑回归；
- 二分类和多分类；
- Gaussian Naive Bayes 与 Multinomial Naive Bayes。

## 2. 训练集和测试集

按照题目规定：

- `bending1`、`bending2` 的 dataset1、dataset2 是测试集；
- 其他活动的 dataset1、dataset2、dataset3 是测试集；
- 其余数据是训练集。

最终得到 69 个训练实例和 19 个测试实例。

测试集只能在最后评估一次。选择分段数、正则强度和特征时，只使用训练集和交叉验证，避免 data leakage（数据泄漏）。

## 3. 时间序列分段和特征

把一条 480 个观测值的序列切成 `l` 段后，每一段计算：

- minimum
- maximum
- mean
- median
- standard deviation
- first quartile
- third quartile

一共有 6 条序列，所以 `l=1` 时每个实例有 `6 × 7 = 42` 个特征；`l=2` 时有 84 个特征。分段可以保留局部变化，但也会增加维度和过拟合风险。

## 4. 二分类逻辑回归

二分类任务把 `bending1` 和 `bending2` 合并为 `bending`，其他活动合并为 `other`。

逻辑回归输出属于 bending 的概率：

`P(Y=1|X) = 1 / (1 + exp(-(beta0 + beta1 X1 + ...)))`

题目要求测试 `l=1,...,20`。使用 stratified 5-fold cross-validation 可以保证每个验证折中仍有合理比例的 bending 和 other 样本。

## 5. p-value 筛选和 separation

普通逻辑回归中，p-value 小通常表示该变量在控制其他变量后仍有统计证据支持其作用。

如果类别几乎可以被特征完全分开，就会出现 separation。此时系数可能非常大，标准误和 p-value 不稳定。作业要求识别并讨论这种情况，而不是隐藏警告。

## 6. 混淆矩阵、ROC 和 AUC

混淆矩阵记录：

- true positive：正确识别 bending；
- true negative：正确识别 other；
- false positive：把 other 判成 bending；
- false negative：漏掉 bending。

ROC 曲线展示阈值变化时 true-positive rate 和 false-positive rate 的关系。AUC 越接近 1，排序区分能力越强；AUC 约为 0.5 表示接近随机。

## 7. 类别不平衡与 case-control sampling

训练数据中 bending 是少数类。如果只看 accuracy，模型可能全部预测为 other 仍然得到看似不错的结果。

case-control sampling 从多数类中抽取与少数类相同数量的样本，使训练更平衡。最终性能仍应在原始测试分布上评估。

## 8. L1 正则化

L1 logistic regression 在目标函数中加入系数绝对值惩罚：

`loss + lambda × sum(|beta_j|)`

它会把部分系数压到 0，同时完成变量选择。正则强度需要通过交叉验证选择。L1 通常比手动逐个做 p-value 筛选更稳定、更容易自动化。

## 9. 多分类模型

多分类 L1 logistic regression 使用 one-vs-rest 方式区分 7 种活动，并报告测试误差、混淆矩阵和 macro one-vs-rest AUC。

Gaussian Naive Bayes 假设每个特征在每个类别内服从高斯分布，适合连续测量值。Multinomial Naive Bayes 通常适合非负计数型特征，因此代码先进行 MinMax scaling。

最终比较应以测试误差为主要依据，同时结合模型假设解释结果。

## 10. ISLR 4.8.3

QDA 允许不同类别拥有不同方差。比较两个类别的 discriminant 时，会出现不同的 `x^2` 系数，因此决策边界一般是二次的。只有当各类别方差相等时，二次项抵消，边界才退化为线性，这就是 LDA 的情形。

## 11. ISLR 4.8.7

使用 Bayes 定理：

`P(Y=1|X=x) = P(Y=1) f(x|Y=1) / [P(Y=1)f(x|Y=1)+P(Y=0)f(x|Y=0)]`

在题目给出的 `X=4`、两个均值分别为 10 和 0、共同方差为 36、先验概率为 0.8 和 0.2 时，结果约为 `0.981`。原因是 4 更接近均值 10，同时 dividend 类别的先验概率也更高。

## 12. 最重要的检查原则

1. 先划分训练集和测试集。
2. 只用训练集选择 `l`、特征和正则强度。
3. 标准化或 MinMax scaling 必须在训练折中拟合。
4. 测试集只用于最后报告结果。
5. notebook 必须从头运行并保留输出。
