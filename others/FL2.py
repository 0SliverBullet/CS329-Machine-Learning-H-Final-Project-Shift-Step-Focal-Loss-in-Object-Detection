import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def FL_0(pt):
    return - np.log(pt)

def FL_1(pt):
    return -((1 - pt) ** 2) * np.log(pt)

def FL_2(pt, alpha):
    return -((1 - pt) ** alpha) * np.log(pt) / (pt ** alpha)
    #return -((1 - (0.9*pt+0.1)) ** alpha) * np.log(pt) / ((0.9*pt+0.1) ** alpha)

pt = np.linspace(0.1, 0.99, 1000) # 生成0.01到0.99之间的100个等间距点

alpha_values = [0.125, 0.25, 0.5, 1, 2]
labels = [f'$\\gamma $= 0.125', f'$\\gamma $= 0.25', f'$\\gamma $= 0.5', f'$\\gamma $= 1', f'$\\gamma $= 2']
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))

fl_0 = FL_0(pt)
fl_1 = FL_1(pt)
sns.lineplot(x=pt, y=fl_0, label='CE')
sns.lineplot(x=pt, y=fl_1, label=f'FL $\\gamma$ = 2')
count=0
for alpha, label in zip(alpha_values, labels):
    count+=1
    fl_2 = FL_2(pt, alpha)

    sns.lineplot(x=pt, y=fl_2, label='GFL* ' + label)
    if count==3:

       break
plt.text(0.5, 5, r"$GFL*(p_t) = -\alpha_t \left( \frac{1-p_t}{p_t}  \right)^\gamma  log(p_t)$",
         fontsize=12, ha="left")

plt.text(0.5, 6, r"$FL(p_t) = -\alpha_t \left( 1-p_t \right)^\gamma  log(p_t)$",
         fontsize=12, ha="left")

plt.text(0.5, 7, r"$CE(p_t) = -\alpha_t log(p_t)$",
         fontsize=12, ha="left")
plt.xlabel('probability of ground truth class')
plt.ylabel('loss')
plt.legend()
plt.show()