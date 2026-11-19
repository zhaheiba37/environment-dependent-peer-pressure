# 环境依赖同伴压力与Q-learning合作演化模型

本项目提供论文 **Environment-dependent peer pressure in spatial public goods games with reinforcement learning** 的模拟代码，包含核心互动过程和参数扫描入口。运行后会生成实验结果。

## 安装与快速运行

使用 Python 3.10—3.12，在项目目录执行：

```bash
python -m pip install -r requirements.txt
python run.py --config examples/quickstart.json --output outputs/demo
```

快速示例运行10×10网络、100 MCS。首次运行会编译Numba核心。每次执行使用新的输出目录。

## 实验过程

个体位于具有周期边界的L×L方格网络。每个互动群体由中心个体及四名最近邻组成。代码中C=1、D=0。初始时个体以相同概率选择合作或背叛，Q值全部设为零。

每个MCS随机发起L²次群体互动，每次均匀随机选择中心个体，允许重复选择。每次互动按以下顺序执行：

1. 统计群体内合作者数量nC，满足nC≥T时为合作环境G，否则为背叛环境B。
2. 根据个体当前行为和群体环境确定状态，联合状态依次为CG、CB、DG、DB。
3. 五名成员分别按epsilon-greedy规则选择动作；探索和Q值相同时均匀随机选择。
4. 同时实施五名成员的新行为，重新判定同一群体的环境。
5. 计算奖励和下一状态，分别更新各自的Q表。

基础收益为 `pi = r*nC/5 - action`，合作成本为1。在合作环境中，背叛者受到 `beta*nC` 的压力；在背叛环境中，合作者受到 `beta*(5-nC)` 的压力；其余个体免受压力。奖励等于基础收益减去压力，每次更新采用当前群体互动的收益。

Q值更新为：

```text
Q(s,a) += alpha * (reward + gamma*max(Q(next_s,:)) - Q(s,a))
```

对照实验支持环境无关压力，即个体受到群体内所有相反行为成员施加的压力。学习状态可选择联合状态、仅行为状态或仅环境状态。

## 论文规模实验

默认参数为L=100、r=3.5、T=3、beta=0.1、alpha=0.1、gamma=0.9、epsilon=0.01。每次运行100万MCS，以最后10万MCS的平均合作比例作为最终结果，每1万MCS打印进度。

先生成参数配置，再运行：

```bash
python make_configs.py fig3-4 --output fig34.json
python run.py --config fig34.json --output outputs/fig34 --workers 3
```

| 扫描入口 | 参数设置 |
| --- | --- |
| fig2a | r=2.8至5.0，步长0.1；beta=0、0.05、0.1、0.15、0.2 |
| fig2b | r=3.2、3.4、3.6、3.8、4.0；beta=0至0.2，步长0.01 |
| fig3-4 | r=3.5、T=3；beta=0、0.1、0.2 |
| fig5a | r=3.5；T=0至5；beta=0、0.05、0.1、0.15、0.2 |
| fig5b | r=3.5；T=1至4；beta=0至0.2，步长0.01 |
| fig6a | T=3；r=2.9至5.0，步长0.1；beta=0.1、0.2；两种压力规则，联合状态 |
| fig6b | T=3、beta=0.1；r=2.9至5.0，步长0.1；三种状态表示 |

其他参数采用默认值。配置生成器默认为每个参数点生成一次独立实验。在 `make_configs.py` 命令中添加 `--repeats 5`，即可为每个参数点生成五次不同随机种子的实验。`--workers` 指定同时运行的独立实验数量。

需要延长演化时间的点，可编辑 `examples/long_run.json` 中的参数，然后运行：

```bash
python run.py --config examples/long_run.json --output outputs/long_run
```

该配置从初始状态开始运行1亿MCS，取最后1000万MCS平均。可结合合作轨迹判断是否需要延长运行。

## 自定义参数

JSON对象定义一次实验，对象列表定义多次实验；省略的参数采用run.py中的默认值。

| 字段 | 含义 |
| --- | --- |
| length、r、threshold、beta | 网络边长、协同因子、环境阈值和压力强度 |
| alpha、gamma、epsilon | 学习率、折扣因子和探索概率 |
| initial_cooperation | 初始合作概率，默认0.5 |
| pressure | dependent（环境依赖）或independent（环境无关） |
| state | joint（联合）、behavior（仅行为）或environment（仅环境） |
| mcs、tail | 总MCS和末段平均窗口长度 |
| seed | 初始化及互动过程的随机种子 |
| progress_every | 打印进度的MCS间隔 |
| record_every | 轨迹记录的MCS间隔，默认1 |
| snapshots | 快照时间，用逗号分隔，例如0,3000,50000,1000000 |

每次执行均从随机初始行为和零Q值开始。增大record_every可减少输出文件大小，最终平均仍使用尾段每一个MCS。轨迹中保存的是对应单个MCS的观测值。

## 输出文件

每次实验对应一个run_XXXX目录：

- metadata.json：参数、随机种子、软件版本、代码哈希和运行状态。
- trajectory.csv：MCS、合作比例、各状态访问次数和状态行为优势；第0行记录初始合作比例。
- actions_mcs_N.npy：运行时间范围内指定MCS的空间快照，0为背叛、1为合作。
- final_state.npz：最终行为和Q表，动作维度依次为D、C。
- summary.json：末段窗口内合作比例的均值和时间标准差。

状态行为优势为当前MCS实际访问事件中、更新前的Q(C)-Q(D)均值。同一个体多次访问分别计数，未访问的状态记为nan。状态顺序记录在metadata中。时间标准差描述单次实验内部的波动。

## 文件结构

- model.py：核心互动与学习过程。
- run.py：读取配置、运行模拟和保存结果。
- make_configs.py：生成论文参数网格。
- examples/：快速示例与长时间实验配置。
- tests/：小规模模型检查，可通过 `python -m unittest discover -s tests -v` 运行。

代码使用许可见LICENSE。
