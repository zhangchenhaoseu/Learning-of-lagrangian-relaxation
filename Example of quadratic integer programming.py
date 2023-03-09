# 靡不有初，鲜克有终
# 开发时间：2023/3/6 15:13
import numpy as np


class Primary_Problem():  # 原问题
    def __init__(self, object_coeff, constraint_coeff, object_constant):
        self.variables_n = len(object_coeff)  # 变量个数
        self.constraints_n = len(constraint_coeff)  # （被松弛的）约束个数
        self.object_coeff = object_coeff  # 目标函数中的变量系数
        self.constraint_coeff = constraint_coeff  # 约束中的变量系数
        self.object_constant = object_constant  # 约束中的常数项

    def relaxed_object(self, multiplicator):  # 输入乘子，输出松弛之后，目标函数中变量的系数
        self.relaxed_object_coeff = []  # 松弛之后，目标函数中变量的系数（含二次项和一次项）
        for i in range(0, self.variables_n):
            self.relaxed_object_coeff.append({'quadratic': self.object_coeff[i], 'linear': sum(multiplicator*self.constraint_coeff[:, i]) })
        self.relaxed_object_constant = sum(multiplicator*self.object_constant)  # 松弛之后，目标函数中的常数项
        # print("松弛后，目标函数中变量的系数（含二次项和一次项）", self.relaxed_object_coeff)
        # print("松弛后原问题的常数项", self.relaxed_object_constant)

    def solve_subproblem(self):  # 求解原问题对应的子问题，子问题数量等于变量个数
        self.results = np.zeros(self.variables_n)  # 子问题的解，不一定满足被松弛之后的约束
        for i in range(0, self.variables_n):
            result = -1 * self.relaxed_object_coeff[i]['linear'] / (2 * self.relaxed_object_coeff[i]['quadratic'])  # 让导数等于0来求解未知量的求解结果
            self.results[i] = max(round(result), 0)  # 满足未被松弛的非负整数约束
        # print("子问题的解", self.results)

    def upper_bound(self):  # 采用启发式算法，调整求解子问题所得到的解，使其可行，得到原问题解的上界
        for i in range(0, self.constraints_n):  # 第i个被松弛掉的约束
            result_lst = list(self.results)
            while sum(self.results * self.constraint_coeff[i, :]) + self.object_constant[i] > 0:  # 当被松弛的约束不满足要求时
                modified_index = result_lst.index(min(result_lst))  # 启发式算法，从值最小的开始调整
                modified_coeff = self.constraint_coeff[i, modified_index]  # 需要被调整的未知量在约束中对应的系数
                if self.results[modified_index] - modified_coeff/abs(modified_coeff) >= 0:  # 保证在调整的时候满足非负约束
                    self.results[modified_index] = self.results[modified_index] - modified_coeff/abs(modified_coeff)  # 若是正系数，则进行-1操作，若负系数则+1操作，使得约束满足
                    result_lst[modified_index] = result_lst[modified_index] - modified_coeff/abs(modified_coeff)
                else:
                    result_lst[modified_index] = 999999  # 将其值调整为较大值，以便不满足非负约束的时候，再寻次小值
        self.modified_results = self.results
        # print("修正后的子问题的解", self.modified_results)
        self.upper_bound = sum(self.modified_results * self.modified_results * self.object_coeff)
        # print("修正后的原问题的上界为", self.upper_bound)

    def report(self, i):  # 展示算法计算过程中的信息
        print("第", i, "次迭代")
        print("子问题最优解为：", self.results)
        print("调整后的可行最优解为：", self.modified_results)
        print("上界为：", self.upper_bound)


class Dual_Problem():  # 对偶问题
    def __init__(self, object_coeff, constraint_coeff, subproblem_outcome, multiplicator):  # 输入目标函数的系数，子问题的约束（被松弛的）,计算结果,乘子
        self.object_coeff = object_coeff
        self.constraint_coeff = constraint_coeff
        self.variables_n = len(object_coeff)
        self.constraints_n = len(constraint_coeff)  # （被松弛的）约束个数
        self.object_constant = object_constant  # 约束中的常数项
        self.results = subproblem_outcome
        self.multiplicator = multiplicator

    def relaxed_object(self):
        self.relaxed_object_coeff = []  # 松弛之后，目标函数中变量的系数（含二次项和一次项）
        for i in range(0, self.variables_n):
            self.relaxed_object_coeff.append({'quadratic': self.object_coeff[i], 'linear': sum(self.multiplicator * self.constraint_coeff[:, i])})
        self.relaxed_object_constant = sum(self.multiplicator * self.object_constant)  # 松弛之后，目标函数中的常数项

    def lower_bound(self):
        self.constant = self.relaxed_object_constant
        self.quadratic_results = self.results*self.results*self.object_coeff
        linear_results_lst = []
        for i in range(0, self.constraints_n):
            linear_results_lst.append(sum(self.multiplicator[i]*self.constraint_coeff[i, :]*self.results))
        self.lower_bound = self.constant + sum(self.quadratic_results) + sum(linear_results_lst)
        # print("检查对偶问题的下界", self.lower_bound)

    def subgradient(self):  # 输出乘子对应的梯度
        self.subgradient = np.zeros(self.constraints_n)  # 乘子的次梯度，作为方向
        for i in range(0, self.constraints_n):
            self.subgradient[i] = np.dot(self.results, self.constraint_coeff[i, :])

    def update_stepsize(self):  # 按照公式输出新的步长
        linear_results_lst = []
        for i in range(0, self.constraints_n):
            linear_results_lst.append(sum(self.multiplicator[i] * self.constraint_coeff[i, :] * self.results))
        self.function_value = self.constant + sum(self.quadratic_results) + sum(linear_results_lst)
        self.stepsize = (417 - self.function_value) / np.linalg.norm(self.subgradient) ** 2
        # print("步长为", self.stepsize)

    def update_multiplier(self):  # 更新乘子
        for i in range(0, self.constraints_n):
            self.multiplicator[i] = self.multiplicator[i] + self.subgradient[i]*self.stepsize
        # print("乘子的更新结果", self.multiplicator)

    def report(self):  # 展示算法计算过程中的信息
        print("下界为：", self.lower_bound)
        print("乘子更新为：", self.multiplicator)
        print()


def gap(upper,lower):
    gap = (upper-lower)/upper
    return gap


if __name__ == "__main__":
    object_coeff = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
    constraint_coeff = np.array([[-1, 0.2, -1, 0.2, -1, 0.2], [-5, 1, -5, 1, -5, 1]])
    object_constant = np.array([48, 250])
    multiplicator = np.array([10, 10])

    subproblem = Primary_Problem(object_coeff, constraint_coeff, object_constant)  # 实例化原问题
    subproblem.relaxed_object(multiplicator)  # 描述松弛后的原问题表达式
    subproblem.solve_subproblem()  # 求解松弛后原问题对应的若干子问题，得到子问题最优解
    subproblem.upper_bound()  # 使用启发式方法对原问题对应子问题的最优解进行调整，使其变为可行解，得到问题的上界

    subproblem_outcome = subproblem.results  # 将原问题对应子问题的最优解传入对偶问题
    dualproblem = Dual_Problem(object_coeff, constraint_coeff, subproblem_outcome, multiplicator)  # 实例化对偶问题
    dualproblem.relaxed_object() # 描述松弛后的对偶问题表达式
    dualproblem.lower_bound()  # 得到在当前乘子和子问题最优解情况下，问题的下界
    dualproblem.subgradient()  # 根据子问题的最优解，得到乘子的次梯度，即前进方向
    dualproblem.update_stepsize()  # 根据计算得到的梯度，使用相关公式，得到对应的步长
    dualproblem.update_multiplier()  # 根据方向和步长，更新乘子
    # dualproblem.report()  # 显示迭代解的细节

    epsilon = 0.001
    k = 1
    while gap(subproblem.upper_bound, dualproblem.lower_bound) > epsilon:
        subproblem.report(k)
        dualproblem.report()

        multiplicator = dualproblem.multiplicator
        subproblem = Primary_Problem(object_coeff, constraint_coeff, object_constant)
        subproblem.relaxed_object(multiplicator)
        subproblem.solve_subproblem()
        subproblem.upper_bound()

        subproblem_outcome = subproblem.results
        dualproblem = Dual_Problem(object_coeff, constraint_coeff, subproblem_outcome, multiplicator)
        dualproblem.relaxed_object()
        dualproblem.lower_bound()
        dualproblem.subgradient()
        dualproblem.update_stepsize()
        dualproblem.update_multiplier()

        k += 1