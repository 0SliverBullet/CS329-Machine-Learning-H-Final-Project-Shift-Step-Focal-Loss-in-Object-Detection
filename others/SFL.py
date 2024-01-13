import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def FL_0(pt):
    return - np.log(pt)

def FL_1(pt,alpha=0.5):
    return -((1 - pt) ** alpha) * np.log(pt) / (pt ** alpha)

def FL_2(pt,alpha=0.5,beta=0.1):
    #return -((1 - pt) ** alpha) * np.log(pt) / (pt ** alpha)
    return -((1 - ((1-beta)*pt+beta)) ** alpha) * np.log(pt) / (((1-beta)*pt+beta) ** alpha)

def FL_3(pt):
    return -((1 - pt) ** 2) * np.log(pt)

pt = np.linspace(0.1, 0.99, 1000) # 生成0.01到0.99之间的100个等间距点

alpha_values = [0.5]
labels = [f'$\\gamma $= 0.125', f'$\\gamma $= 0.25', f'$\\gamma $= 0.5', f'$\\gamma $= 1', f'$\\gamma $= 2']
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))

fl_0 = FL_0(pt)
fl_1 = FL_1(pt)
fl_2 = FL_2(pt)
fl_3 = FL_3(pt)
sns.lineplot(x=pt, y=fl_0, label='CE')
sns.lineplot(x=pt, y=fl_3, label=f'FL $\\gamma$ = 2')
sns.lineplot(x=pt, y=fl_1, label=f'GFL $\\gamma$ = 0.5')
sns.lineplot(x=pt, y=fl_2, label=f'SSFL $\\gamma$ = 0.5, $\\beta$ = 0.1')
plt.text(0.4, 4, r"$SSFL*(p_t) = -\alpha_t \left( \frac{1-((1-\beta)p_t+\beta)}{(1-\beta)p_t+\beta}  \right)^\gamma  log(p_t)$",
         fontsize=12, ha="left")


plt.text(0.4, 5, r"$GFL*(p_t) = -\alpha_t \left( \frac{1-p_t}{p_t}  \right)^\gamma  log(p_t)$",
         fontsize=12, ha="left")

plt.text(0.4, 6, r"$FL(p_t) = -\alpha_t \left( 1-p_t \right)^\gamma  log(p_t)$",
         fontsize=12, ha="left")

plt.text(0.4, 7, r"$CE(p_t) = -\alpha_t log(p_t)$",
         fontsize=12, ha="left")
plt.xlabel('probability of ground truth class')
plt.ylabel('loss')
plt.legend()
plt.show()