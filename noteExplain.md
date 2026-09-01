# 任务列表

科研项目

1. Semester Arbeit: Quadruped+Reinforcement Learning

# 大体情况

认知误区：

复习不能只有输入，关键是建立输出的反应网络，

所以过一遍知识点是最轻松的第一步，最重要是建立输出+反馈+输入的机制

就是要做真题，遇到不会就回去输入知识点，然后反复输出+对答案反馈

高频、微小的“输出-反馈-输入”闭环，能像玩游戏一样让你上瘾，绝不会出现“脑子糊”的现象

本周任务：

每日任务：

我想下载以下数据集并在当前go2\_RL\_AMP的dataset\_Test的branch中测试不同的数据集并进行比较，

我想下载的数据集是：### 数据集链接

- \*\*Kine2Go (最推荐)\*\*：数据集存放在 Hugging Face 上，仓库 ID 是 `kine2go-review/kine2go`。官方提供了一个轻量的 `review\_sample` 用于测试[](https://huggingface.co/api/resolve-cache/datasets/kine2go-review/kine2go/3698e74f8f9e889697cafc872fb32dd5e8285fb7/README.md?download=true&etag=%22e27fcec52956a4846c87d17fc38607eccf68db39%22)[](https://huggingface.co/datasets/kine2go-review/kine2go?duplicate=true)。完整数据集包含了 40 个动作片段和 800 条策略轨迹[](https://arxiv-org.ezproxy.obspm.fr/html/2606.14433v1)[](https://huggingface.co/papers/2606.14433)。论文见 [arXiv:2606.14433](https://arxiv.org/abs/2606.14433)[](https://arxiv-org.ezproxy.obspm.fr/html/2606.14433v1)[](https://huggingface.co/papers/2606.14433)。

- \*\*Quadrupedal Agility (研究价值高)\*\*：这是 2026 年发表在 Nature Communications 上的工作，提供了已重定向到 Go2 的狗动作数据[](https://arxiv-org.ezproxy.obspm.fr/html/2606.14433v1)。你可以查找论文 "Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots"[](https://arxiv-org.ezproxy.obspm.fr/html/2606.14433v1) 的官方项目页或配套的 Zenodo 代码库[](https://jszx.cau.edu.cn/art/2025/10/27/art\_45097\_1086802.html)。

- \*\*DogML (多样性测试)\*\*：来自 T2QRM 项目，是第一个狗-机器人运动语言数据集，包含 8,048 段动作片段和 12,072 条文本描述[](https://dl.acm.org/doi/full/10.1145/3696409.3700230)[](https://github.com/SCUT-BIP-Lab/T2QRM)。官方 GitHub 仓库为 `SCUT-BIP-Lab/T2QRM`，可在其主页找到下载链接[](https://github.com/SCUT-BIP-Lab/T2QRM)。

注意这几个数据集的适用性，你帮我测试一下，最后给我在tensorboard呈现就行了，对了，训练不用这么多，执行cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

conda activate env\_isaacsim

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --headless --num\_envs 500 --max\_iterations 500就好

# 目标计划

1. Semester Arbeit: Quadruped+Reinforcement Learning

**快捷键**

**tmux attach -t Lingheng\_Test 重新进入会话**

学长在做什么？

**刘泽霖**SA做的deployGo2是新狗+cpg的sim2sim，用isaacgym和移植到mujoco用不同仿真器跑，因为不同仿真器的policy不一致，可以更好验证；现在上海实习做灵巧手

-sim2sim和sim2real就是接口不同，其他都是一样的

-学长提示reward func很重要，调参只是最后一步

-robot-viewer可以在网页端显示三维模型

-SA报告写8页就好了

-提示isaacSim稳定版本是4.5，lab是2.1

**黄钰琪**做的董学长的狗加上cpg算法

-base\_height是第一关节角度

-8月初前交MA论文

**刘天赐**做的MA，7月1号工作了，之前还在写论文，懂得多，可多点联系，做的amp+GO2的sim2sim

-isaacGym上训练效果好，mujoco上训练效果差，分析的大概率原因是因为：amp这个方法不好+数据清洗retargeting不行

-要一下论文和答辩PPT，论文写了42页还觉得不够

-论文上科研图可用PPT打草稿，然后交给AI画连线与构图；但容易被发现图框之间连线对不齐

-建议我先看amp\_hardware源代码；另外nvidia本身isaacLab有很多库现成的

-sim2real可以找刘泽霖学长问怎么sim2sim找接口，跑流程要一周

**具体目标**：(确定下来的用加粗表示)

看amp对抗运动先验和苏黎世那篇walk likedog论文

(Amp是刘天赐学长做的，walk likedog论文主要读retargeting部分为以后MA洗数据用)

**熟悉AMP代码，看懂RL怎么做的输入输出推理模块权重网**

**熟悉issac sim，跑一遍示例**

**用isaacSim复现的AMP的sim2sim**

**到宇树Go2的sim2real**

熟悉自己做的狗代码

做sim2real移植

题目

Cross-Platform Sim-to-Sim-to-Realdeployment for Adversarial Motion Priors on Quadrupedal Robots

# 项目日志

1、Semester Arbeit: Quadruped+Reinforcement Learning

**第4周：2026年5月5日-2026年5月8日**

硬软件介绍

**VS Code (SSH)** 是用来干活的：写代码、改配置、查看文本日志。它很快，延迟极低。

远程连SSH：

远程主机ip: ssh tumi6@131.159.60.37

密码：tumi6

**AnyDesk** 是用来看画的：只有当你要看 Rviz 里的机器人模型、Gazebo 仿真或者 Matplotlib 画出的图表时，才去翻 AnyDesk 窗口

anydesk远程画面

远程ID：1294010250

密码：tuminfo6

**Tmux**实时更新功能，程序课挂在主机跑，彩条及时报错不用管

如果主机重启后tmux默认丢失，重建一个新的

tmux new -s LinghengsTest

重新进入会话

tmux attach -t LinghengsTest

配置的美化工具starship没有正确加载，可以重新 source 一下：

source ~/.zshrc

常用检查命令

检查显卡（GPU）占用：

nvidia-smi

检查 CPU 和内存占用：

htop

查看当前有谁登录了这台机器:

who

揪出后台隐藏的 Python/训练任务:

ps -ef | grep python

重启 Ubuntu 系统:

sudo reboot

**第5周：2026年5月11日-2026年5月17日**

Conda环境配置

学长仓库都放在home下面Repo的文件夹了，我自己新建了一个LinghengTest文件夹

为了不污染base环境,

如果直接在 base 环境里装插件，万一版本冲突导致服务器 Python 崩了，所有人都要停工

新建了一个Lingheng\_env环境，

环境（Environment）和文件夹（Folder/Project）在物理上是分开存放的，但在逻辑上是“绑定”的关系:

项目文件夹：住在 ~/repo/Projects\_Lingheng\_Kong。这里只放你的源代码（.py, .cpp）和配置文件。

Conda 环境：统一住在 ~/miniconda3/envs/ 目录下。这里放的是巨大的 Python 解释器、各种库（PyTorch, numpy）

用direnv绑定自己的文件夹

direnv (.envrc 文件)：这是一个“进出门自动切换”的工具。

原理：你在某个目录下放一个 .envrc 文件，里面写好 conda activate xxx。当你用命令行 cd 进入这个目录时，它会自动帮你把虚拟环境切好；离开时自动切回来。这能防止你“人在 A 项目，却运行了 B 环境”，比如：

输入cat.envrc

输出source ~/miniconda3/etc/profile.d/conda.sh

conda activate spine\_gym%

因为isaacgym比较老旧，torch版本老旧不支持用5090跑，所以学长建议参考这篇文章：https://blog.csdn.net/m0\_56706433/article/details/148902144

学长在home/download下载了一个已经编译好的pytorch补丁，当自己的conda环境配置完之后在pip install home/download/torch\_xxx.whl 把这个torch安装进自己的环境里

Whl 文件：你可以把它想象成一个“离线安装包”

学长安装的是他“房间”里的家具，你现在是在布置你“房间”里的家具

驱动的真相：即便 Isaac Gym 版本旧，nvidia-smi 是正常的，我们还能通过替换那几个 .so 补丁来“骗”过系统。

多花时间在显卡驱动本身就是坏的（Mismatch 报错），nvidia-smi错误，系统现在的 `Mismatch` 是因为新旧文件混杂，然后我把所有 NVIDIA 相关的包全部删干净，

sudo apt-get purge nvidia -y

sudo apt-get autoremove -y

添加官方驱动源，

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-get update

ubuntu-driversdevices

sudo apt-get install nvidia-driver-580 nvidia-utils-580 nvidia-settings-580 -y

强制对齐并重启了主机就好了

sudodpkg --configure -a

sudo reboot

**Isaac gym示例**

1、在 AnyDesk 的 Ubuntu 桌面里，点开终端（Terminal）

2、输入命令：

cd ~/Repo/isaacgym/python/examples

python 1080\_balls\_of\_solitude.py

或者

在远程桌面输入xhost +local:tumi6

echodISPLAY

再在vscode输入exportdISPLAY=:0

python 1080\_balls\_of\_solitude.py

现在我的env\_kongstest 环境里已经整齐地摆好了三件大件：

isaacgym：物理仿真引擎

rsl\_rl：强化学习算法

legged\_gym：机器人具体任务和场景

学长还说cpg jump是另外一位同学做的，建议是看看就行，主要还是找自己的思路

学长说deploy这个仓库是sim2sim和sim2real用的，你可以在自己电脑上跑一下试试，需要安装ros2

这个deploy部署 仓库是一个中转站。它把你刚才在 legged\_gym 里训练出来的那个神经网络模型（Policy），通过 ROS2 通讯协议，“喂”给物理机器人（Real）或者更真实的机器人模拟器（Sim2Sim，比如专门的 Unitree 模拟器）

source install/setup.bash 到底在干什么

理解为“修改当前终端的通行证”。

注册路径：告诉系统，你的自定义 ROS2 节点、模型文件、可执行脚本都在 my\_Deploy\_Go2Robot/install 里面。

设置变量：它会自动修改 PATH（找程序）、LD\_LIBRARY\_PATH（找库文件）和 PYTHONPATH（让 Python 能 import 你写的代码）。

给自己的ProjectsTest\_Lingheng\_Kong仓库建立了github

安装IsaacSim

先执行 git clone，再执行 pip install

因为IsaacLab 是一个开源项目，它的代码库（Repo）里包含了大量的：

示例代码 (Tutorials/Examples)：比如如何控制 Go2 机器人，如何训练小球。

配置文件：环境设置、强化学习算法的超参数。

安装蓝图：当你运行 pip install isaaclab[...] 时，pip 只是安装了“库”。如果你没有克隆这个文件夹，你就没有地方去写你的代码，也找不到那些现成的demo 脚本。

git clone https://github.com/isaac-sim/IsaacLab.git .

然后再跟这个网站一步步来：

<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html> 注意改为"isaaclab[isaacsim,all]==2.3.2.post1"

问题：有时候会闪退

初次加载：Isaac Sim 在首次启动时会从服务器下载部分资产（Assets）并编译缓存。

5090 兼容性：RTX 5090 性能太强，有时候引擎的预热速度跟不上指令下达的速度。

窗口闪退或提示“Not Responding”通常是因为 Isaac Sim 在启动时需要进行大量的 着色器编译（Shader Compilation）。第一次运行由于需要把 5090 的驱动和引擎进行适配，系统压力会瞬间飙升，容易被操作系统误判定为“无响应”。

**IsaacSim示例**

介绍：

Isaac Sim 是英伟达开发的“游戏引擎（渲染+物理）”，它提供物理世界、CUDA加速和 5090 显卡渲染；IsaacLab 是建立在它之上的“机器人算法框架”，专门用来写强化学习（RL）代码、控制宇树狗子

底层的物理引擎其实都叫 PhysX：不管是全新的 Isaac Sim、IsaacLab 还是老旧的 Isaac Gym，它们背后负责计算“骨骼刚体碰撞、摩擦力、重力”的底层物理引擎，全部是英伟达大名鼎鼎的 PhysX（完全由显卡加速）。

第一代：Isaac Gym（已废弃/停更）：

前几年火遍学术界（比如强化学习跑宇树狗、四足机器人横行无忌的大部分经典论文都基于它）。它是英伟达当年做的一个“概念验证版”强化学习环境，画面很简陋（没有高级光追），没有图形界面，完全是在显存里纯代码裸跑。目前官方已经彻底停止维护。

第二代：Isaac Sim + IsaacLab（当前王牌）：

英伟达发现 Isaac Gym 饱受好评，于是把它和商业级 3D 渲染引擎（Omniverse）进行了史诗级大融合，诞生了 Isaac Sim（解决了画面逼真度、ROS2 接口和复杂传感器仿真问题）。随后为了让大家能像以前在 Isaac Gym 里一样爽快地写 RL 算法，又在 Isaac Sim 之上套了 IsaacLab。

本地：

真正的 Isaac Sim 仿真软件，现在就像 numpy 或 torch 一样，作为底层第三方二进制庞大依赖，默默躺在你的 conda env\_isaacsim 环境的 site-packages 文件夹深处

conda activate env\_isaacsim

python -c "import isaacsim; print(isaacsim.\_\_file\_\_)"

输出：/home/tumi6/miniconda3/envs/env\_isaacsim/lib/python3.11/site-packages/isaacsim/\_\_init\_\_.py在这里，找出来基本没用，千万不要去动它

find source/ -name "teleoperation.py"

cd ~/Repo/isaacSim

conda activate env\_isaacsim

isaacsim

在Isaac/5.1/Isaac/Robots/Unitree/Go2/找到了

**第6周：2026年5月18日-2026年5月24日**

阅读论文walk likedogs

learning steerable imitation controllers for legged robots from unlabeled motiondata

核心目标：是让四足机器人（如 Unitree Go2）从真实狗的未标注动作数据中，自动学会多种步态（Pace、Trot 等），并能根据用户摇杆指令实时切换步态，同时保持狗的自然风格

特点是Without manual labeling, predefined mode counts, or explicit switching rules, maintaining the stylistic coherence of thedata

传统模仿学习面临的三大挑战：

形态/物理差异：狗和机器人身体结构、重量、执行器不同，直接套用动作会失败（脚滑、穿地、关节超限）。

可控性：单纯回放轨迹无法响应用户实时指令。

多样性：真实数据里有很多不同步态（慢走、快跑等），怎么让机器人自动发现这些模式并在合适速度下切换，而不需要人工标注？

本文解决方案：

Kino-dynamic Motion Retargeting：把狗的动作“翻译”成机器人能物理执行的版本。

Steerable Motion Synthesis（基于 VAE）：学习一个“动作生成器”，输入用户期望的速度（前进+转弯），输出合适的参考动作序列，能自动切换步态。

Residual RL Tracking Controller：强化学习训练一个跟踪控制器，让机器人真实执行上面生成的参考动作。

整体框架分为三个阶段：

Stage 1: Kino-dynamic MR(Motion Retargeting)

核心任务是解决把真狗（源数据 source）爬行、奔跑的动捕数据，完美转录到机器狗上

Retargeting拆成了两步 ：

第一阶段：Kinematics Stage（纯几何与运动学映射）

这一步的目标是先在数学和几何上粗暴地把狗的动作等比例缩放到机器狗身上

1. 提取真狗的特征（每帧提取）：身体中心（Base）的三维坐标 [^srcx, ^srcy, ^srcz]和旋转角度（姿态：翻滚、俯仰、偏航角）

算出狗的三维速度（前后速度、左右速度、转向角速度）

提取肢体向量（Limb Vectors）：论文里明确写道，这指的是从肩膀/髋关节指向对应足端（Foot）的单位方向向量

1. 初步映射到机器狗（初级重定向 UVM 方法） ：

身体高度和姿态：直接乘上一个缩放因子alpha（因为机器狗比真狗矮小），直接安在机器狗身体上

身体的前后左右位置：通过把刚刚提取的速度乘上时间步长Delta t进行数值积分，累加算出来机器狗每一时刻应该走到哪里

脚的位置（公式 3）：机器狗的足端目标位置 = 机器狗自己的肩膀位置 + 缩放后的肢体向量（Limb Vector）

算出了身体位置和脚的位置后，用传统的逆运动学（IK Solver）反推，就能算出机器狗 12 个关节各自应该转多少度 (^tgtq) 。

但产生的 Artifacts（缺陷）这种传统的直接缩放方法被称为 Unit Vector Method (UVM，单位向量法) 。它的效果往往很差，容易带来三大穿帮镜头（如图 3a 所示） ： Limb penetration（肢体穿地）：脚或者膝盖直接插进地里去了 。 Contact foot slip（脚底打滑）：原本应该踩死在地面的脚，在地上像滑冰一样摩擦 。 Violation of joint limits（超出关节限位）：算出来的关节角度机器狗根本转不过去 。

第二阶段：Dynamics Stage（动力学优化与约束过滤）

因此避免人为矫正模型，解决方法是知道最终targeted motion通过加入限制constraints来逆运动IK求解，同时用MPC(白盒：模型中每一环都可以用物理数学公式解释)离线(算一次读一次)优化系统动态和物理极限的因素

步骤 A：带约束的逆运动学优化（Constrained IK Optimization）

作者不再直接用公式硬算关节角度，而是把这一步写成了一个带约束条件的优化问题（公式 4）。

优化的目标：让机器狗的机身姿态、空中挥舞的腿，尽量去贴近真狗的动捕轨迹。

死命令约束（Constraints）：

不许打滑（公式 4a）：只要检测到某只脚此时是踩地的（Stance foot），它的 x, y, z坐标在当前时间步内必须死死固定在地面锚点上，绝对不许动。

不许穿地（公式 4b & 4c）：所有空中的腿和膝盖，其高度 z必须大于 0（必须在地面以上）。

不许脱臼（公式 4d）：算出来的关节度数必须在机器狗电机的物理极限内。

结果：这一步解完后，得到了一套在几何上绝对不打滑、不穿地的完美动作序列

步骤 B：基于模型预测控制的动力学过滤（MPC Physics Filter）

虽然步骤 A 解决了打滑和穿地，但它依然只是个“几何魔术”。在真实物理世界里，机器狗有质量、有惯性、电机有马力（扭矩）上限。如果动捕数据里有真狗突然猛烈摆腿的动作，机器狗的电机根本带不动，直接照搬就会导致硬件瞬间摔倒或过载。于是，作者引入了 MPC（模型预测控制）追踪器：

做法：在 MuJoCo 物理仿真环境中，放一只拥有真实质量、真实惯性、真实电机扭矩特性的虚拟机器狗。

算法：利用 iLQG 求解器，设置一个 2.0 秒的预测展望时间（Horizon），让这个虚拟机器人拼命去“追”步骤 A 算出来的动作。

目的：让物理引擎去当裁判。MPC 会在绝对不违反牛顿力学、不超出电机最大马力、不超出地面摩擦力极限的前提下，计算出一套最平滑、最合理的电机执行轨迹。

输出：参考运动轨迹（Kinematic Targets）包括参考关节角度（ref\_θ\_t）、参考基座位置、速度、脚的位置等；

不直接输出力矩，只负责提供“理想情况下应该是什么样子”的运动目标（带风格的参考动作）

其他知识：

原本这个方法是做四足动画的用的

Animal motiondata有base pose和limb vectors两项，Limb Vector肢体向量 就是用一个从“肢体根部（关节起点）”指向“肢体末端（足端/手部）”的带有方向和长度的数学向量，来直接描述一条腿或一只手臂在空间中的相对位置

常见步态：

Pace-溜蹄 / 同侧步-低速，同侧前后腿几乎同时抬起和落下频繁出现

Trot-小跑 / 对侧步-中速，斜对角的两条腿同时抬起（最稳定）

Gallop-奔驰 / 飞奔-高速有明显“飞行相”（四脚同时离地）

UVM = Unit Vector Method（单位向量法）

这是论文中提到的传统、简单、快速的运动重定向（Motion Retargeting）基线方法。

UVM 的具体做法（非常直观）：

把狗的身体（base）位置、高度、朝向按一定比例缩放，转移到机器人身上。

把狗的四肢方向用单位向量（unit vector）表示（从肩部指向脚尖）。

把这些单位向量按机器人肢体长度缩放后，直接加到机器人的肩部位置，得到脚的目标位置。

用普通逆运动学（IK）或直接计算关节角度，让脚去到目标位置。

为什么 UVM 会产生问题？

论文明确指出，UVM 会引入严重的运动学 artifact（伪影）：

Limb Penetration（肢体穿地）：腿部穿过地面。

Contact Foot Slip（脚滑）：支撑脚在地面滑动，而不是固定。

Joint Limit Violation（关节超限）：机器人关节角度超出物理限制。

动态不稳定：导致后续 RL 训练很难收敛。

狗和机器人的形态差异很大（腿长、身体比例、关节范围不同），简单缩放无法保证物理合理性，尤其在动态动作（跑、转弯）时问题更严重。

Stage 2: Steerable Motion Synthesis

Stage 2-1: Motion Embedding（动作嵌入：“压”到一个更低维、更结构化的空间里） —— 离线训练C-VAE带约束(Conditional Variational Autoencoder)

运动是一个连续的马尔可夫过程（Markov Process）未来只取决于现在，而与过去无关输入 x\_t-1相当于给了decoder 一个物理锚点

本身Autoencoder（普通自编码器）是Encoder 把输入压缩成一个“瓶颈”向量（latent code），Decoder 再把它解码回原输入；

VAE是在原有基础上不学一个固定点，而是学一个概率分布（通常是高斯分布），Encoder 输出的是均值 μ 和 方差 σ（描述一个分布），这种概率分布的连续性是平滑动作和步态切换的关键，同时梯度下降会逼迫编码器化身为一个极度挑剔的“分类大师”，保证了多模态；训练时从这个分布中采样一个 z，再喂给decoder又因为加入 KL 散度正则项，让学到的分布尽量接近一个简单先验（如标准正态分布）

KL 散度（Kullback-Leiblerdivergence）

衡量两个概率分布之间的差异，计算分布 A 和分布 B 之间有多不像，用标准正态分布N~(0, I)做标准，KL 散度硬生生把所有动作的编码分布都往“标准正态分布”z\_mid去靠拢时

VAE的优势是（尤其在运动生成中）可以生成新样本（不是简单复制）；隐空间更平滑连续，相邻点对应相似的动作，便于插值和控制

本文特别用了 Hyperspherical VAE（超球面 VAE，用 von Mises-Fisher 分布），让隐变量落在单位球面上（有界、稳定），非常适合后面 RL 策略去“探索”

输入：retargeted 后的机器人动作数据库中的状态对，前一帧 vae\_x\_t-1，当前帧 vae\_x\_t，也称为state transition

Encoder：把这一对状态编码输出成隐变量 z 的概率分布（18 维超球面）。

Decoder：给定 z 和前一帧，预测当前帧，从隐变量“还原”出合理的动作

训练目标：重构误差 + KL 正则（让分布规整）。

额外技巧：逐渐从单步预测变成自回归预测（用自己预测的结果作为下一帧输入），提升长期稳定性。

结果：得到一个训练好的 VAEdecoder，它能从隐变量 z “解码”出合理的下一帧动作。同时隐空间自动捕捉了不同步态的聚类，也就是相似动作在隐空间里会聚集在一起，不同步态（Pace、Trot、Gallop）会自然形成不同的“区域”

先用 VAE （变分自编码器）把动作“嵌入”到一个超球面隐空间（hyperspherical latent space），不会无限探索，VAE 自动学习数据的多模态分布，不同步态自然聚成不同“簇”，因为数据本身包含多种步态，RL 策略在优化速度跟踪时，会自动发现并利用不同的隐空间区域（对应不同步态），从而实现无监督的步态发现和切换，不需要 explicit switching rules（显式切换规则），也不需要预设模式数量，全部 emergent（自然涌现）

Stage 2-2: Steerable Motion Synthesis Module（可控动作合成模块） —— RL 训练策略

用 RL 训练一个 Motion Synthesis Policy，它学会在隐空间里“走”，根据用户指令生成下一帧参考动作，这是offline可控的部分，训练一个 Motion Synthesis Policy（也是 RL，用 PPO）：

输入：用户指令 c\_t（前进速度 + 转弯速度） + 上一个生成的参考状态，得到一个向量 → 投影到超球面得到 z\_t。

1. 状态（State）：超球面z\_t
2. 动作（Action）：输出超球面地图上的一个坐标z\_t
3. 奖励函数（Reward Function）：

3-1任务追踪奖励（Task Tracking Reward）：Decoder 会当场吐出一个预测的当前帧动作^x\_t, 计算decoder 吐出来的速度，和用户手柄输入的目标控制指令c\_t之间的物理差距

3-2动作先验约束/风格重构奖励（Style / Prior Preservation Reward）：最小化两个 vMF（von Mises-Fisherdistribution）分布之间的 KL 散度来实现

4. 状态转移（State Transition / Environment）：把 z\_t 喂给上面训练好的 VAEdecoder，得到新的参考状态

作用：策略学会在隐空间里“导航”，找到哪个 z 能产生对应速度的动作（自然就用了不同步态）

Stage 3: Motion Tracking只它online

RL 训练残差策略（residual policy），把合成出来的参考动作真实执行在机器人上。

输入：当前机器人状态 + 高层给的参考轨迹

输出：关节角度残差（residual joint positions）

然后由底层 PD 控制器根据这个目标计算实际关节力矩，驱动电机

其他知识：

Heuristic Control“启发式控制”就是指“传统控制中人工凭经验拍脑袋写出的控制死规则”

DeepMimic (2018, Xue Bin Peng 等)

- 开创性工作：最早把“模仿运动数据 + RL”大规模用于物理角色控制。

- 方法：给 RL 策略一个密集的模仿奖励（当前姿态和参考姿态越接近，奖励越高）。

- 优势：能学复杂技能（走、翻跟头、打拳、恐龙走路等），支持多段不同 clip。

- 局限：

- 需要精心设计的奖励和参考状态初始化。

- 单个策略通常只能学一种或有限几种行为。

- 切换行为时需要额外机制。

- 地位：很多后续工作（包括本文）的基础。

GAN生成对抗网络Generative Adversarial Networks

由两个神经网络组成，

生成器 (Generator, 简称 G)：负责凭空制造数据（比如生成一张人脸图片）。它的目标是让自己的作品越来越逼真，从而欺骗判别器。

判别器 (Discriminator, 简称d)：负责做分类。它同时输入“真实数据”和“生成器做出来的假数据”，它的目标是识破假货。

在不断的对抗中，生成器做假钞的技术越来越高，判别器认假钞的火眼金睛也越来越厉害。最终达到平衡时，生成器生成的动作或图像逼真到连判别器也分不出真假

AMP: Adversarial Motion Priors (2021, Xue Bin Peng 等)

在传统的机器人强化学习里，我们要让机器狗学会走，得手写几十个生硬的惩罚项；而 AMP 彻底抛弃了这些手写规则，套用了 GAN 的对抗架构：

生成器 = RL Policy（策略网络）：机器狗在 Isaac Sim / Isaac Gym 仿真环境里自己尝试输出关节力矩。它不仅要完成任务（比如往前走，获得任务奖励 Task Reward），还要想办法让自己的走姿看起来像真狗

判别器 =discriminator Network：它去读取“真狗的动捕数据”和“此时机器狗在仿真里跑出来的动作”。如果它觉得机器狗跑得太机械、不像真狗，就给低分；如果觉得很逼真，就给高分（这个分数就是风格模仿奖励 Style Reward）

最终公式：Total Reward = Task Reward（走得快不快）+ Style Reward走得像不像真狗

优势：风格很强，能从少量数据学出多样行为；不需要精确的逐帧模仿奖励。

- 局限：

- 训练不稳定（GAN 常见问题）。

- 容易 mode collapse（只学会少数几种动作，丢失多样性）。

- 端到端训练，调试较难。

- 本文明确提到 AMP 的问题，并用 VAE + 结构化隐空间 + 简单速度奖励 来避免这些缺点。

本文 vs 它们：

-deepMimic：需要强模仿奖励 + 较难处理多模态。

- AMP：风格好但不稳定、多样性差。

- 本文：结合 VAE 的结构化表示 + 简单奖励 → 稳定 + 多模态自动发现 + 易控。

重点读 Section IV（Retargeting） 和 Section V（Synthesis）（技术核心）

**第7周：2026年5月25日-2026年5月31日**

周一二放假

周三去实验室和董学长再次确定论文方针路线：

移植amp到isaacSim做sim2sim，最后sim2real到Go2上

会议

周四答辩排练，天赐学长给我讲代码

周五正式答辩

**确认整个 pipeline。**

我理解现在是

go2\_secamp：legged\_gym\envs\go2\_secamp

训练 motion prior

学会 pace / trot / canter 这些基础步态

go2\_RoughTerrain：legged\_gym\envs\go2\_RoughTerrain

冻结 motion prior

再训练一个 residual policy

让机器人能在 rough terrain 上更稳地跟踪速度指令

`go2\_secamp` 是PPO训练 AMP motion prior，

`go2\_rough` 在 prior 上训练 residual policy，

`depoly\_mujoco` 做 MuJoCo sim2sim。

重点追问：

```text

datasets/camp/ 是什么数据？左前、右前、左后、右后，每条腿 3 个关节

frame[0:3] base 位置

frame[3:7] base 四元数

frame[7:19] 12 个关节角

frame[19:31] 4 只脚位置

frame[31:34] base 线速度

frame[34:37] base 角速度

frame[37:49] 12 个关节速度

frame[49:61] 4 只脚速度

**> 如果我只想验证现有结果，不重新训练，应该按什么命令跑？**

> 如果我要重新训练，分别跑哪些命令？

> 哪个 checkpoint 是你认为能用的版本？

你需要得到类似这样的答案：

python legged\_gym/scripts/play\_go2\_rough.py --task=go2\_rough

pythondepoly\_mujoco/deploy\_mujoco\_rough.py go2\_rough.yaml

python legged\_gym/scripts/train.py --task=go2\_secamp

python legged\_gym/scripts/train.py --task=go2\_rough

问关节顺序和坐标系

这个非常关键。MuJoCo、Isaac Gym、Isaac Sim 的关节顺序/四元数顺序/坐标系只要错一个，机器狗就会抽搐。

> **现在 Isaac Gym 和 MuJoCo 的 joint order 是不是不一样**？`REORDER` 这个数组是怎么来的？迁移 Isaac Sim 时我应该以哪个顺序为准？

重点追问：

Isaac Gym 训练时的 joint order 是什么？

MuJoCo XML 的 joint order 是什么？

Isaac Sim 里 USD/URDF 导入后 joint order 怎么确认？

四元数顺序是 xyzw 还是 wxyz？

base\_lin\_vel / base\_ang\_vel 是 world frame 还是 body frame？

projected\_gravity 怎么算？

command 是 body frame 还是 world frame？

**第8周：2026年6月1日-2026年6月7日**

刘天赐学长给的参考资料

NVIDIA的ProtoMotion的库

https://github.com/NVlabs/ProtoMotions

他自己的my\_AMP\_for\_hardware\_V1库

https://github.com/escontra/my\_AMP\_for\_hardware\_V1?tab=readme-ov-file

强化学习B站教程

https://space.bilibili.com/1337590903?spm\_id\_from=333.337.0.0

看学长仓库改动过的文件：（标志是2 months ago）
datasets - 重新添的自己的数据库

depoly\_mujoco

legged\_gym

Pretrained

Resources

rsl\_rl

README.md

setup.py

**AMP项目代码架构**

环境与配置分離：每个训练任务由两个核心文件决定：

legged\_robot.py（环境主体）：定义机器人的动作空间和状态。

legged\_robot\_config.py（配置文件）：包含 LeggedRobotCfg（环境参数，如重力、地形）和 LeggedRobotCfgPPO（训练参数，如学习率、AMP 奖励权重及参考数据集路径）。

介绍概念：SECAMP: Skill-Conditioned Adversarial Motion Priors for Quadruped Locomotion

它将对抗运动先验（AMP）与基于技能条件的策略学习（Skill-Conditioned Policy Learning）相结合。单个策略即可学会在小跑（trot）、侧对步（pace）和奔跑（canter）等自然步态之间进行顺畅切换，同时能够追踪基于航向点（waypoint）的导航指令，且全过程无需人工设计复杂的奖励函数

该框架完全在 IsaacGym 仿真环境中针对 Unitree Go2 机器狗训练而成，支持向 MuJoCo 的“仿真到仿真（sim-to-sim）”迁移，以及直接进行“仿真到实机（sim-to-real）”的部署

核心特性

* 多技能运动：通过一个离散的独热编码（one-hot skill code），单个策略即可同时控制小跑（约 1.5 m/s）、侧对步（约 1.0 m/s）和奔跑（约 2.5 m/s）。
* 基于航向点的导航：采用截断位置指令（Clamped position commands）将步态与速度解耦，使机器人在到达目的地时能够自然减速。
* 双航向点前瞻（Lookahead）：通过在完全到达当前航向点之前预先锁定下一个航向点，从而鼓励机器人实现平滑的转弯过渡。
* 条件判别器（Conditionaldiscriminator）：受条件生成对抗网络（cGANs）启发的投影式架构，在不发生梯度消失的情况下，提供特定技能的模仿奖励。
* 对称性增强（Symmetry Augmentation）：镜像奔跑动作，使机器人在任何步态相位下都能实现无缝的左/右技能状态切换。
* 非对称演员-评论家架构（Asymmetric Actor-Critic）：在训练期间，评论家（Critic）可以获取特权状态（如真实值速度）；而部署的演员（Actor）仅使用机载传感器。

系统架构总览（System Overview）

运动库 (MoCap 动捕数据集) IsaacGym 仿真环境

小跑 / 侧对步 / 奔跑 Go2 机器人 + 航向点 (W1, W2)

(trot / pace / canter) │

基于技能条件的专家数据采样 │ 本体感受观测 (Proprioceptive Obs - 45维)

│ 特权观测 (Privileged Obs - 51维)

│ AMP 状态过渡 (AMP Transitions - 43×2维)

▼

┌───────────────────────────────── ┐

│ 技能感知条件变量 c │

│ [ 步态热码 (3维) | 控制指令 (3维) ] │

└─────────────────────────────────┘

│ │

┌───────────▼────┐ ┌────────▼────────┐

│ 演员网络 π\_θ │ │ 评论家网络 V\_ψ │

│ (Actor 策略网络) │ │ (Critic 价值网络) │

│ [1024, 512, 256] │ │ [1024, 512, 256] │

│ 45维 → 12 自由度│ │ │

└───────────┬────┘ └────────┬────────┘

│ │

关节目标值 价值估计值

Joint Targets Value Est.

│ │

└────────┬────────┘

│

PPO 更新

│

┌────────────────────▼────────── ──┐

│ 判别器d\_φ │

│ (Discriminator 网络) │

│ 骨干网络 + 技能投影层 │

│ [1024, 512] → 模仿奖励 r^I ∈ [0, 1] │

└──────────────────────────────── ─┘

输入解释：

Proprioceptive Obs (本体感受观测/45维)：指机器狗自身传感器能直接测得的数据（如关节角度、角速度、IMU 姿态等），这也是实机部署时 Actor（演员网络） 唯一能用的数据

Privileged Obs (特权观测/51维)：只有在仿真环境里才能拿到的上帝视角数据（如足端与地面的真实接触力、外部物体的精确距离等），专门喂给 Critic（评论家网络） 用来稳定训练

Skill-Aware Condition (技能感知条件变量)：把 3 维的独热步态编码（对应小跑/侧对步/奔跑）和 3 维的导航指令（方向与位置）拼在一起，作为条件控制信号同时输入给策略和判别器

架构基准：

首先建立标准 Actor-Critic DRL的前向流程：

状态 s ──→ Actor ──→ 动作 a ──→ 环境 ──→ 奖励 r，新状态 s'

└─────→ Critic ──→ V(s)

Actor：策略 π(a|s)，决定动作

Critic：价值函数 V(s)，评估"这个状态未来能拿多少累计奖励"

奖励 r：由人手写的公式从环境状态算出来

PPO 在这个框架里干的事：用 Critic 的 V(s) 算出优势 A = 实际回报 − V(s)（"这一步比预期好多少"），再用这个优势指导 Actor 更新，同时加 clip 限幅防止一次更新迈太大步。

GAN 的对抗结构

专家动捕数据（真狗）───┐

├──→ Discriminator ──→ 这段动作"真"的程度

Actor 跑出来的动作 ─────┘ │

↑ │

└────────── r\_imitation ←─────────────────────┘

判别器努力分辨"真狗做的" vs "机器人做的"

Actor 努力让判别器分不出来（判别器给的高分 = 高奖励）

两者对抗，Actor 的动作风格被逼着往专家数据上收敛

标准框架有两个痛点，正好被 AAC 和 AMP 分别解决：

痛点 1：Actor 和 Critic 不共用输入，Asymmetric非对称，为了 Actor 能上真机，被迫把 Critic 也限制在真机可测的信息里，Critic 变笨，训练变慢

痛点 2：奖励不必手写，"走得像真狗"这种传统风格性目标style reward，人根本写不出公式，AMP的判别器解决了

**AAC**

(Asymmetric Actor-Critic，非对称演员-评论家)：Actor和Critic关心的是"怎么完成任务"（走到目标点、用什么步态）。之所以叫 Asymmetric（非对称），核心就在于：Actor 和 Critic 进出的门道完全不同，Actor 最终要装进机器人，它的输入必须是真机测得到的；Critic 训练完就丢弃，永远不上真机，所以可以在仿真中开上帝视角。 我们把这些观测数据分成两派，它们通过网络前向传播（Forward）和直接拼接（Concat）的方式进入网络。

Actor：负责"决策"——看当前状态，决定下一步该怎么动（输出12维action）。这是唯一部署到真机上的部分。进入的数据：Proprioceptive Obs（本体感受 45维）输出：这 51 维的数据输入给 Actor 的 [1024, 512, 256] 网络，最后输出 12 维的关节目标角度Joint Targets（12doF，控制四足机器人的 12 个电机）

Critic：负责"打分"——评估当前状态未来能拿多少奖励，帮助训练Actor。只在训练时存在。（价值网络）输出一个 1 维的标量（Value Est. 价值估计，用来评价当前狗子混得好不好），直接承担了奖励函数的设置

奖励传递更新方式 Critic:PPO（回归） 和 Actor：PPO（clip）：

传递给 Critic：这个 r^I 会和环境本身的奖励（比如有没有走到航向点）加总。Critic 看到这个总奖励后，去更新自己的“价值评估”能力，更新自己的参数。

传递给 Actor：Actor 听到 Critic 说：“你刚才往左扭屁股的动作，教练（判别器）给了超高分，而且也接近目标了！” Actor 就会在接下来的策略更新中，调大这种动作的出现概率。

**AMP**

(Adversarial Motion Priors，对抗运动先验)：是系统的步态美化机制 / 奖励来源，它规定了系统里必须塞进一个判别器（Discriminator）和一套真狗的动捕数据库（MoCap）

Discriminator (判别器)：判别器把这段动作和 MoCap 动捕库里的真狗动作做对比。如果觉得“像真狗”，就给出接近 1.0 的高分；如果觉得“动作畸形”，就给出接近 0.0 的低分。Discriminator完全不关心任务目标，它只关心这个动作的运动学模式，风格上像不像专家（真狗）数据。这个分数，在代码里被称为模仿奖励（Imitation Reward r^I）它也只在训练时存在，不会被部署。

更新方式：LSGAN

独立的小屋：AMP Transitions（43×2维）

它是连续两个时间步（t 和 t+1）的机器人姿态剪影（每步 43 维，包含关节位置、速度等）。它被打包成 86 维的向量，单独送进判别器d中去打分

**PPO**

update 就是那个统筹全局、负责让演员演得更好、评论家看得更准的“总导演”

工作 A：升级 Actor（改策略）

利用 PPO 算法的裁剪目标函数（Clips Objective），计算当前策略和旧策略的差异。如果某套步态既能快速走到航向点，又能拿到判别器的高分，PPO 就会强行修改 Actor 神经网络的权重，让机器狗未来更倾向于做出这种优雅、快速的动作。

工作 B：修正 Critic（精算师）

对比 Critic 预测的价值和机器狗实际跑出来的累计得分。PPO 会修改 Critic 的权重，让它对形势的估计越来越准。

工作 C：对抗升级（给判别器充能）

这是最精彩的部分。如果机器狗（Actor）变得越来越聪明，学会了“作弊”和“伪造步态”来骗取判别器的高分，PPO 会在这个阶段同时更新判别器的权重。让判别器的眼神变得更毒辣，强迫机器狗向更高精度的动捕细节去演进。

什么是 PPO 算法的裁剪目标函数（Clip Objective）

在强化学习中，策略网络（Actor）是通过梯度上升来寻找能获得最高奖励的动作。但传统的策略梯度有一个致命弱点：如果某次策略更新的步长过大，导致网络权重突变，策略可能会直接崩塌（比如机器狗突然直接“散架”再也站不起来），模型就彻底废了

为了让策略更新更稳健，PPO（Proximal Policy Optimization）引入了裁剪目标函数（Clips Objective），它就像是一个“策略安全气囊”，把新旧策略的变化幅度死死限制

传统最优控制的策略往往是一个确定性函数；

现代强化学习的策略（神经网络π\_θ）输出一套“概率分布的参数”：对于连续动作空间，通常是输出高斯分布的均值和标准差，然后调用一个采样函数输出动作（比如a是某个电机的转动角度），一个是为了“探索”（Exploration），一个是为了算概率，能最终逐步确定最佳动作的均值，同时包含一个惩罚项（叫做熵惩罚 Entropy Penalty），或者随着训练步数增大，算法会自动强行让标准差逐渐变小，也就是最终确定最佳动作

整个更新流程：

第一阶段：数据收集与概率记录

将一个虚拟狗导入isaacGym这个物理仿真环境中，不涉及MoCap，自然获得狗的各项数据，即Proprioceptive Obs和Privileged Obs，

Actor神经网络只接收Proprioceptive Obs + Skill Code + Command（没有Privileged Obsactor），actor基于现阶段状态输出每个关节的概率分布的两个核心参数mean and ，采样函数拿着这两个数在后台“摇号”，然后就可以知道动作和做完后的状态

第二阶段：Discriminator 判别器的权重更新

判别器D神经网络根据第一阶段后虚拟狗的前后状态和真狗在跑步时的AMP Transition（录下的 MOCap 数据其实是一帧一帧的连贯动作，“当前帧的关节位置+速度”以及“下一帧的关节位置+速度”拼在一起的一个长向量43×2 = 86维）进行打分，也是AMP的精髓，生成了一个实打实的奖励 r^I模仿奖励，只丢给critic神经网络

判别器（Discriminator）的更新不归 PPO 管，判别器是用标准 GAN（生成对抗网络）的二分类交叉熵损失（Binary Cross Entropy Loss）独立更新的；标准的 GAN 是由两部分组成的：一个生成器（Generator）负责造假，一个判别器（Discriminator）负责鉴宝，在这个机器狗项目里，Actor（机器狗的策略网络）就是那个隐形的生成器（Generator），系统算完这个独立的对抗损失后，利用反向传播，直接修改了判别器网络自己的权重，目的是为了把眼神练得更凶狠，把真狗和假狗分得清清楚楚

第三阶段：Critic 评论家网络的权重更新

Critic 接收的Privileged Obs+ Skill Code + Command，然后输出一个标量预测值Value Est.，自己来评判狗走得好不好，

然后用总奖励R =环境的任务奖励（比如离航向点近不近）+ 判别器刚吐出来的模仿奖励r^I，和Critic自己生成的Value Est.进行比较，得到优势A，是发给第四阶段 PPO 的。如果 A > 0，说明Actor刚才摇号摇出来的动作比预期好

第四阶段：PPO的上帝视角减速

神经网络，是通过梯度上升自行修改权重的，得到L^clip(theta)，通过神经网络的反向传播（Backpropagation）与梯度下降（Gradientdescent），对每一个神经元回推偏导数，让 Loss 的得分变得更高，计算完梯度后，优化器（如 Adam）会直接强行冲进显存/内存中，把 Actor 网络原本的权重更新

PPO的本质是一个类函数一个数学公式，输入actor的新概率、旧概率和优势A，使用裁剪目标函数（Clips Objective），取新概率、旧概率相除得到比率（保证在1-e,1+e内），然后乘以优势，得到L^clip(theta)

所以说PPO的clip用来限制神经网络的权重收敛，而PPO只负责更新Actor和Critic神经网络的参数

Repository Structure代码结构

my\_AMP\_for\_hardware\_V1/

├── legged\_gym/

│ ├── envs/

│ │ ├── go2\_secamp/ SECAMP environment + config

│ │ ├── go2/ Standard AMP environment

│ │ └── base/ Base legged robot class

│ └── scripts/

│ ├── train.py Training entry point

│ └── play.py Evaluation / visualization

├── rsl\_rl/rsl\_rl/

│ ├── algorithms/

│ │ ├── secamp\_ppo.py SECAMP PPO algorithm

│ │ └── amp\_ppo.py Standard AMP PPO

│ ├── modules/

│ │ ├── actor\_critic.py Shared actor-critic network

│ │ ├── secamp\_disc\_project.py Projectiondiscriminator

│ │ └── secamp\_disc\_concat.py Concatdiscriminator

│ ├──datasets/

│ │ └── secamp\_motion\_loader.py Skill-conditioneddata loader

│ └── runners/

│ └── go2\_secamp\_runner.py Training loop orchestration

├──datasets/

│ └── camp/ Reference motion clips (.json)

│ ├── trot0.json ... trot2.json

│ ├── pace0.json ... pace2.json

│ ├── canter0.json ... canter5.json includes mirrored

│ └── left\_turn0.json, right\_turn0.json ...

├──depoly\_mujoco/ MuJoCo sim-to-simdeployment

├── pretrained/ Pre-trained checkpoints

└── resources/ Go2 URDF / MJCF assets

整个项目由两大核心算法库（legged\_gym 和 rsl\_rl）以及三个支撑模块（datasets、resources、deploy）组成

2026年6月7日开会简要

代办：问刘学长sim2real接口流程：

读deployGo2的readme就知道sim2sim流程了

**第9周：2026年6月8日-2026年6月14日**

今周任务：利用天赐学长的代码中在sim2sim和sim2real跑通，让董学长把我an上

**已完成：**

**sim2sim（isaacGym）**

在~/R/ProjectsTest\_Lingheng\_Kong/**my\_AMP\_for\_hardware\_V1**跑狗：

在 AnyDesk 的 Ubuntu 桌面内的终端里运行输入这两行：

cd ~/Repo/ProjectsTest\_Lingheng\_Kong/my\_AMP\_for\_hardware\_V1

python legged\_gym/scripts/train.py --task=go2\_secamp --num\_envs=10

python legged\_gym/scripts/train.py --task=go2\_secamp --headless --num\_envs=500

接下来在mujoco中sim2sim和sim2real然后天赐学长的嘱咐：

-在my\_AMP\_for\_hardware\_V1跑狗时play是简单，难的是training训练调整的东西多

-叫我在sim2sim和sim2real时注意，ros用的是foxy版本，有环境信息后，机器人底层都由一个三状态状态机（3-state state machine）管理:趴下，起立和激活 RL 策略

-手柄默认的是xbox研发，ps4不行再改一下引索；A/B是切换狗的站姿，按住手柄两个LB+RB不放激活自己的policy

-激活policy有两种方法：自己带笔记本+网线连IP；或者宇树够自带jetson推理模块，带学长提示文件夹混乱，环境乱套

-总的流程（不管是sim2sim还是sim2real）是：

Deploy的必看文件在Desktop\Semester Arbeit\Projects\my\_Deploy\_Go2Robot，两个pdf一个reame.md, 还有宇树官方App的登录账号密码图片

1确保ROS Foxy和环境是对的

2配置SDK

3验证手柄

4在设置-服务状态中关闭宇树自己的policy:sport\_mode，密码是8个8

执行中矛盾点：

-学长说的是ROS Foxy+Python 3.8，但在主机上显示是ROS Humble+Python 3.10

-

**手柄**使用注意事项：

一、 手柄在sim2sim和sim2real模式下的“使用差异”

1. 在 Sim2Sim 仿真中使用

硬件接线：手柄直接插在运行仿真节点的主机

节点启动区别：通过传递 `is\_simulation:=true` 告诉状态机，手柄的指令应该发送给 MuJoCo 仿真器，而不是真实的狗（终端3/4）

ros2 rundeploy\_rl\_policy low\_level\_ctrl --ros-args -p is\_simulation:=true

ros2 rundeploy\_rl\_policy rl\_policy.py --is\_simulation True

2. 在 Sim2Real 实机部署中使用

文档的最后两页（第5、6页）重点强调了实机时的手柄使用方案 ：

安全前置条件：在通过手柄控制真机前，必须首先通过手机 App 登录授权账号，连接机器狗 Wi-Fi，进入设备服务状态，手动关闭机器人自带的官方运动模式（sport\_mode） 。否则手柄发出的自定义策略指令会与官方运控发生严重冲突。

部署方案 A（连接主机部署 - 有线）：

接线：你的电脑通过网线连接 Go2 拓展坞，静态 IP 设为 `192.168.123.xxx` 。

手柄：依然插在你的电脑/主机上。此时启动控制节点，将参数改为 `false`：

ros2 rundeploy\_rl\_policy low\_level\_ctrl --ros-args -p is\_simulation:=false

状态机会通过网线将手柄转换为实机低级控制命令（`lowcmd`）发给狗子。

部署方案 B（机载计算平台部署 - 无线）：

接线：把代码仓库传入机器人机载的 Jetson 计算平台并完成编译 。

手柄：断开显示器和网线，手柄直接通过无线（如蓝牙）或物理方式连接到机器人机载系统上，通过手柄完全无线控制机器狗移动 。

二、 控制使用方法（Sim2Sim 与 Sim2Real 完全相同）

无论是仿真还是实机，机器人底层都由一个三状态状态机（3-state state machine）管理。你需要通过手柄按键来控制它的物理行为切换：

1. 初始状态（趴下）：

启动系统后，机器人自动进入 Layingdown（趴下） 状态。

2. 控制起立：

按下 O 键：机器人执行内置算法，从“趴下”切换为 Standing Up（站立） 状态。

3. 激活 RL 策略（最关键的一步）：

机器人站稳后，同时按下 LB + RB：系统会激活你的强化学习策略节点（`rl\_policy.py`），机器人开始执行推理。

操作细节：当你想要通过手柄摇杆（Sticks）自由控制机器人前后左右移动时，必须一直保持按住 LB + RB 键

4. 切回趴下：

按下 X 键：机器人会从站立状态切回 Layingdown（趴下）。

**已完成：**

**sim2sim（Mujoco）**

在~/R/ProjectsTest\_Lingheng\_Kong/**my\_Deploy\_Go2Robot**中sim2sim跑狗：

第一步：将手柄插入主机

验证主机是否识别到 PS4 手柄

检查驱动

lsmod | grep joy

#如果没有输出，加载驱动

sudo modprobe joydev

#再次检查

ls /dev/input/js\*

正常情况：会输出/dev/input/js0（或者js1）

第二步（可跳过）：启动 Joy 节点并测试按键

由于你现在是在笔记本上通过 VSCode 远程操作，我们要在远程主机上让 ROS 2 抓取手柄的数据：

1. 启动手柄驱动节点：

ros2 run joy joy\_node

(如果提示找不到命令，先运行一下 `sudo apt install ros-humble-joy` 安装驱动。)

2. 测试按键映射（非常重要）：

保持上面那个终端，新建一个终端窗口，同时回到my\_Deploy\_Go2Robot，输入：

cd ~/Repo/ProjectsTest\_Lingheng\_Kong/my\_Deploy\_Go2Robot

手柄上的按键引索pdf都记下放在Semester Arbeit\Projects\my\_Deploy\_Go2Robot

代码在my\_Deploy\_Go2Robot\src\deploy\_rl\_policy\src\low\_level\_ctrl.cpp中

第三步：分终端启动 MuJoCo Sim2Sim 仿真

手柄测试没问题后，把整个仿真系统转起来，在远程主机中开 4 个终端窗口：

cd ~/Repo/ProjectsTest\_Lingheng\_Kong/my\_Deploy\_Go2Robot\_V1

点击Terminator，Ctrl + Shift + O向右扩展，Ctrl + Shift + H向下新增终端

终端 1：保持运行手柄节点

ros2 run joy joy\_node

终端 2：启动 MuJoCo 仿真器（进入你的项目文件夹并激活环境）

ros2 rundeploy\_rl\_policy mujoco\_simulator.py

此时实验室的主机屏幕上应该会弹出一个 MuJoCo 窗口，里面有一只趴着的 Go2 机器狗。

终端 3：启动低级控制状态机控制器（必须指定为仿真模式）

ros2 rundeploy\_rl\_policy low\_level\_ctrl --ros-args -p is\_simulation:=true

终端 4：启动强化学习策略节点

#ros2 rundeploy\_rl\_policy rl\_policy.py --is\_simulation True

使用deploy\_secamp.py强化学习策略节点

#ros2 rundeploy\_rl\_policydeploy\_secamp.py --is\_simulation True --mode joystick

python3 src/deploy\_rl\_policy/scripts/deploy\_secamp.py --is\_simulation True --mode joystick

#0X=pace, 1O=trot, 2△=canter

用自动路线模式，就改成：

#ros2 rundeploy\_rl\_policydeploy\_secamp.py --is\_simulation True --mode waypoint

python3 src/deploy\_rl\_policy/scripts/deploy\_secamp.py --is\_simulation True --mode waypoint

**出现的硬件问题**

**主机卡机后物理断电重启**

Nvidia显卡输入

lsmod | grep nvidia

输出下面这几行带有 nvidia 字样的结果：就说明驱动的内核模块（Kernel Modules）已经百分之百成功加载到 Linux 系统里了。软件代码已经就位，在内核层是完全健康的

nvidia\_uvm 2088960 0

nvidia\_drm 139264 0

nvidia\_modeset 1638400 1 nvidia\_drm

nvidia 104157184 2 nvidia\_uvm,nvidia\_modeset

video 77824 3 xe,i915,nvidia\_modeset

折腾了这么久，终于把这个藏在系统最深处的硬核 Bug 彻底揪出来并干掉了。这是一个非常经典的 “高版本硬件固件安全策略限制 + 系统非正常断电引起的驱动死锁” 案例

🔍 1. 核心病因（为什么显卡会突然“装死”？）

1物理层锁死：断电电涌导致 RTX 5090 显卡硬件在重启时触发了主板的 ACPI 电源保护，固件直接卡在了低功耗的挂起状态（D3 Cold），导致 `nvidia-smi` 找不到设备。

2软件层硬锁（真凶）：在通过软件强行唤醒 PCIe 链路后，NVIDIA 50 系列卡（GB202 架构）触发了其独有的底层硬件安全保护策略——5090 芯片在固件级别强行规定，在 Linux Ubuntu 6.8 内核下必须且只能配合官方的 Open 开源内核驱动分支（Open Kernel Modules） 运行。下面是5090 显卡硬件固件（GSP）写死的底层物理规则，旧的闭源驱动根本没有解开 5090 新架构暗号的钥匙

NVRM: The NVIDIA GPU 0000:01:00.0 ... requires use of the NVIDIA open kernel modules.

NVRM: GPU 0000:01:00.0: RmInitAdapter failed!

3逻辑死锁：当时系统里装的是传统的闭源商业驱动。开机时内核成功加载了代码（所以 `lsmod` 正常），但当驱动去敲显卡大门时，5090 硬件芯片发现暗号不对（拒绝闭源握手），直接报 `RmInitAdapter failed` 拒绝初始化，从而死锁。

🛠️ 2. 专业排查五步走（教科书级别的debug 流程）

我们在没有拆卸机箱的情况下，通过纯软件手段一层层扒开了内核：

1. 总线排查：通过 `lspci` 确认了主板在物理层其实是拉到了显卡（地址 `01:00.0`），排除了显卡掉电烧毁。

2. 内核模块排查：通过 `lsmod | grep nvidia` 确认了内核驱动模块加载成功，说明驱动文件没有损坏。

3. 环境对齐：通过 `dpkg-reconfigure` 和 `ldconfig` 刷新了全局动态链接库，排除了新旧驱动文件污染和缓存冲突。

4. 物理放电：拔掉电源线狂按 5 次开机键，强行压榨干净主板电容残余静电，排除了 ACPI 状态机死锁。

5. 一发入魂：使用最高权限 `sudo dmesg | grep -i nvidia` 调取 Linux 系统核心启动日志，直接抓到了显卡硬件抛出的绝杀报错：`requires use of the NVIDIA open kernel modules`。

🏁 3. 终极解决方案

1. 挖地三尺，把不合规的闭源驱动及依赖彻底 purge 干净

sudo apt purge "nvidia" -y

sudo apt autoremove -y

2. 迎合 5090 硬件锁，定向安装带 "-open" 的官方开源内核驱动分支

sudo apt install nvidia-driver-580-open nvidia-utils-580 -y

3. 强制对齐并重启大主机，让 5090 固件与 Open 驱动成功握手

Sudo dpkg --configure -a

sudo reboot

**Anydesk打不开isaacSim**

🔍 1. 核心病因（为什么明明连着 AnyDesk，却一直喊 `No windowing`？）

物理根源：大主机在断电重启时，机箱后面是没有插任何物理显示器（显示器线没接/屏幕没开）的，Ubuntu 的懒加载机制：Ubuntu 的显示管理器（GDM3）开机时检测不到物理显示器，就会自作聪明地拒绝初始化默认的 `:0` 号物理屏幕通道。

AnyDesk 的障眼法：当你通过 AnyDesk 远程连入时，AnyDesk 为了能让你看到桌面，在系统后台凭空创建了一个临时的虚拟屏幕通道（也就是 `:1` 号通道）。

矛盾点（为什么报错）：你在 AnyDesk 桌面里打开的新终端，默认的环境变量仍然在刻舟求剑地去寻找 `:0` 号物理屏幕。由于 `:0` 根本不存在，加上系统为了安全默认锁死了外来控制台对图形桌面的访问权限，导致 Isaac Sim 内部的图形库（GLFW）直接崩溃，疯狂报 `unable to opendisplay` 和 `Failed to acquire IWindowing`。

🛠️ 2. 核心解决三步走（抢回屏幕钥匙）

我们通过精确的变量对齐和权限下放，成功把终端的“视线”引向了 AnyDesk 的虚拟屏幕：

1. 定向爆破（重新定位屏幕）：

exportdISPLAY=:1

作用：明确告诉终端和 Isaac Sim：“别去找那个不存在的 0 号物理屏幕了，把画面给我渲染到 AnyDesk 正在使用的 1 号虚拟桌面上！”

2. 强夺钥匙（解除访问限制）：

xhost +local:

作用：彻底撕掉 Ubuntu 的图形安全保护锁，强行允许当前控制台的所有本地进程（包括 Conda 环境里的 Isaac Sim）直接向这个 1 号屏幕写入并弹出 3D 窗口。

3. 完美避坑（忽略系统拖拽）：

isaacsim --no-window-drag-drop

作用：由于 AnyDesk 是虚拟屏幕，不支持高阶的物理窗口跨屏拖拽（Externaldrag &drop）。加上 `--no-window-drag-drop` 可以让 Isaac Sim 绕过这个检测，直接把 3D 主舞台崩出来。

🏁 3. 完整通关指令流（以后开机 AnyDesk 弹不出窗口的通用公式）

如果在之后的项目推进中，大主机再次遇到重启找不到窗口的情况，你只需要在 AnyDesk 的终端里无脑甩出这套连招：

1. 灌入 ROS2 环境，防止 Bridge 报错

source /opt/ros/humble/setup.zsh

2. 强制绑定 AnyDesk 图形通道并解锁权限

exportdISPLAY=:1

xhost +local:

3. 正常启动你的仿真环境

isaacsim --no-window-drag-drop

**网络灵异事件**

🔍 问题根源：Linux 路由表的“精神分裂”

这次问题的核心，可以用一句话概括：物理上网线虽然插回了实验室外网，但 Ubuntu 系统大脑里的“导航地图（路由表）”还迷失在机器狗的局域网世界里。

1. 为什么会失联？

当你为了控狗，把网卡手动改成静态 IP 192.168.123.100 并在网关填了 192.168.123.1 时，系统在底层生成了一条默认路由（Default Route）。这条路由的意思是：“只要想去外网，一律走 192.168.123.1 这扇大门。”

2. 为什么接回外网改回 Auto 依然报错？

当你在没有物理拔线、或者网卡没有重新初始化的状态下，直接把图形界面改回 Auto 时，Ubuntu 的网络管理器（NetworkManager）产生了配置残留。

结果：你的主机虽然物理上插回了外网网线，甚至重新拿到了 TUM 的 IP。但由于那条顽固的 .123.1 默认路由没有被及时清理，导致系统想把响应你笔记本、VSCode、AnyDesk 的数据包，全部错误地从有线网口砸向已经不存在的狗子网段。数据有去无回，笔记本自然就显示主机“彻底失联”了。

🛠️ 核心解决方法与命令拆解

我们刚刚执行的三行组合拳，实际上是在 Linux 底层进行了一次网络环境的“全面强制刷新”：

1. 强行关闭有线网卡（断开所有错误的底层物理连接和残留握手）

sudo ip link set enp4s0down

2. 重新开启有线网卡（让网卡硬件重新上电，进入就绪状态）

sudo ip link set enp4s0 up

3. 彻底重启 Ubuntu 的网络核心管理器服务

sudo systemctl restart NetworkManager

💡 这一套操作的本质是：

强行让 enp4s0 网卡“瞎一次再睁开”。在重新睁眼的瞬间，NetworkManager 会彻底擦除之前残留的 192.168.123.X 路由表，重新向 TUM 实验室的交换机发起干净的dHCP 请求，从而拿到了正确的默认网关 131.159.61.254，把笔记本和外网的通道彻底打通。

**第10周：2026年6月15日-2026年6月21日**

在sim2sim时遇到学长的policy加载不上去的问题

2026年6月18日早上开会并录屏了

给了我最后一版的deploy项目文件夹（最原始的Deploy是嘉宝学长的）

给了很多参考文献，以便快速掌握amp和aac的算法

用tensorboard看效果，找到runner才是主循环程序

Policy记得还用pd30\_0.7的，适应真狗

讲解了整个项目的心路历程和解决policy加载问题

主要是要改4个地方：

1 config的go2.yaml的路径和scale（为了归一化处理）

2 rl\_policy.py的观察量维度

3 手柄中的引索

参考文献

1.

Peng, X. B., Ma, Z., Abbeel, P., Levine, S. & Kanazawa, A. AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control. ACM Trans. Graph. 40, 1–20 (2021).

2.

Peng, X. B., Abbeel, P., Levine, S. & Panne, M. vande.deepMimic: Example-Guideddeep Reinforcement Learning of Physics-Based Character Skills. ACM Trans. Graph. 37, 1–14 (2018).

3.

Peng, X. B., Guo, Y., Halper, L., Levine, S. & Fidler, S. ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters. ACM Trans. Graph. 41, 1–17 (2022).

4.

He, T. et al. ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills. Preprint at https://doi.org/10.48550/arXiv.2502.01143 (2025).

5.

Peng, X. B. et al. Learning Agile Robotic Locomotion Skills by Imitating Animals. Preprint at https://doi.org/10.48550/arXiv.2004.00784 (2020).

6.

Escontrela, A. et al. Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions. in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 25–32 (IEEE, Kyoto, Japan, 2022).doi:10.1109/IROS47612.2022.9981973.

7.

Huang, N., Xie, Z. & Li, Q. Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors. Preprint at https://doi.org/10.48550/arXiv.2509.21810 (2025).

8.

Li, C. et al. Learning Agile Skills via Adversarial Imitation of Rough Partialdemonstrations.

9.

Li, C. et al. Versatile Skill Control via Self-supervised Adversarial Imitation of Unlabeled Mixed Motions. Preprint at https://doi.org/10.48550/arXiv.2209.07899 (2023).

10.

Serifi, A., Grandia, R., Knoop, E., Gross, M. & Bächer, M. VMP: Versatile Motion Priors for Robustly Tracking Motion on Physical Characters. Computer Graphics Forum 43, e15175 (2024).

11.

Sood, S. et al. APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking on Legged Robots. Preprint at https://doi.org/10.48550/arXiv.2505.10022 (2025).

12.

Tessler, C. et al. CALM: Conditional Adversarial Latent Models fordirectable Virtual Characters. in Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Proceedings 1–9 (2023).doi:10.1145/3588432.3591541.

先看2和5，这是最原始的模仿学习的思路

然后看1，6，这俩是amp算法和部署

然后看4，讲如何解决sim2real

然后是7，8，9是amp的一些衍生

最后是3，10，11，12这些是一些更先进的方法

本周工作

这周试着用copilot修改以加载policy，并整理文件夹同步Git

重新整理~/Repo/ProjectsTest\_Lingheng\_Kong，重新整理文件夹：

my\_my\_AMP\_for\_hardware\_V1\_V1：重新下载的天赐的github仓库，并加入在主机上~/Repo/my\_AMP\_for\_hardware\_V1/logs/

my\_Deploy\_Go2Robot\_V1：自己改动的deploy库

my\_spine\_test：纯粹用来试用kongstest\_env\_issac\_gym环境的

original\_my\_AMP\_for\_hardware\_V1：原始老外的AMP

original\_deploy\_go2\_fromTianci：学长的deploy库

sim2real

**已完成：**

在~/R/ProjectsTest\_Lingheng\_Kong/**my\_Deploy\_Go2Robot**中sim2real跑狗：

第一阶段：实机安全整备

在让你的代码接管狗子之前，必须先解除官方自带算法的控制权，否则手柄发出的自定义策略指令会与官方运控发生严重冲突,导致狗子直接“疯掉”或摔飞

先双击（第二下长按）启动GO2电源，然后狗会自动站起来，此时要让它坐下防止过热，可以选择unitree自带手柄和手机app控制

方式一：

自带手柄也是双击（第二下长按）启动电源；两端L2按住点A让狗子坐下

方式二：

下载 Unitree Go APP，登录账号密码，在Semester Arbeit\Projects\my\_Deploy\_Go2Robot中

手机连接狗子的“家人” WiFi 信号 ，打开 Unitree Go APP

进入GO，点击趴下+阻尼

进入“设备 -> 服务状态”，手动关闭 sport\_mode/1.0.7.30

第二阶段：网络通信搭建

1. 物理连接：用网线将你的电脑与 Go2 扩展坞的 RJ45 网口相连

2. 主机静态 IP 配置：

方式一：Ubuntu

在Ubuntu 系统右上角设置setting中打开wired，将有线网卡设为手动mannual模式，然后选择ip4修改：

IP 地址：192.168.123.100

子网掩码：255.255.255.0

**方式二：Windows**

Windows键+S打开搜索，输入cmd，右键以管理员方式运行命令行

输入：（实际网口以输入ipconfig结果为准）

netsh interface ip set address name="以太网 6" static 192.168.123.100 255.255.255.0 192.168.123.1

进行ping 测试：ping 192.168.123.222

输入ssh unitree@192.168.123.222，密码是 123

然后会问ros:foxy(1) noetic(2)，选择1

第三阶段：代码传输与实机编译

首先将编译产物传输到 Go2

方式 A：使用纯命令行 SCP（推荐，不需要额外软件）

在 Windows 上打开终端（PowerShell），把 Linux 主机上编译好的整个工作空间传到 Go2：

SCP 到 Go2（实际以你的deploy文件为主）

输入：scp -r /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/my\_Deploy\_Go2Robot\_V1 unitree@192.168.123.18:/home/

然后输入 Go2 的密码 123，等待传输完成

**方式 B：使用 VSCode Remote-SSH**

在VSCode 中左下角连接远程，点击Remote-SSH: Connect to Host

输入 unitree@192.168.123.18

输入密码 123

连接成功后，在 VSCode 中打开 /home 文件夹

把my\_Deploy\_Go2Robot\_V1 文件夹拖拽到 Go2 的 /home/ 目录下

实机部署也有两种运行策略，你可以根据需求二选一 ：

策略 A：机载边缘计算部署（实操时无网线）

刚才已经把编译代码仓库下载到机器人机载的 Jetson 计算平台了

核心坑点提示：环境隔离与编译

机载系统由于是 Ubuntu 20.04 + ROS2 Foxy ，如果遇到 `ModuleNotFoundError` 等类型支持报错，在 `colcon build` 编译时必须强行指定 Python 3.8 路径 ：

终端输入

export PYTHON\_EXECUTABLE=/usr/bin/python3.8

colcon build --symlink-install

策略 B：PC 主机直接局域网控制（需拖着网线）

终端输入colcon build --symlink-install

直接利用你之前在 Ubuntu 22.04 主机上已经成功编译好的环境，只要能 `ping` 通狗子，就可以在主机上直接打火运行

第四阶段：实机高并发打火运行（Sim2Real 开跑）

因策略 A：机载边缘计算部署（实操时无网线）调试困难且需要接外设控制GO2的linux

所以实行策略 B：PC 直接局域网控制（需拖着网线）

不再需要启动 `mujoco\_simulator.py` ，只需要打开 3 个独立的终端窗口，依次执行以下 ROS2 节点：

终端 1：启动手柄硬件接口

ros2 run joy joy\_node

(接收你的 PS4/Xbox 手柄输入，并发布到 `/joy` 话题)

终端 2：启动状态机核心（核心参数切为实机！）

ros2 rundeploy\_rl\_policy low\_level\_ctrl --ros-args -p is\_simulation:=false

⚠️ 核心注意点：这里的参数必须修改为 `is\_simulation:=false` ！此时状态机才会把底层的控制流从小电流的仿真模式，切换成对准实机关节电机的真实低级控制字（lowcmd）

终端 3：拉起你的强化学习策略网络

#ros2 rundeploy\_rl\_policy rl\_policy.py --is\_simulation False

python3 src/deploy\_rl\_policy/scripts/deploy\_secamp.py --is\_simulation False --mode joystick

(将推理策略的标志位也同步设为 `False` )

2026年6月21日开会

刘泽霖学长现在写SA的论文，仿照iros期刊格式，8页双栏

接下来要深入看AMP怎么训练了，

看看unitree\_rl\_gym稳定的版本里奖励函数

https://github.com/unitreerobotics/unitree\_rl\_gym

让其更robust有三大方法，域随机化、观察里加噪声、mujoco里跑得好

**第11周：2026年6月24日-2026年6月28日**

学会如何跑harness engineering

Harness Engineering 的核心组成部分

1. Guides（引导）：系统提示、规则、技能（Skills）、工作流模板。
2. Sensors（传感器）：监控、验证机制（代码检查、测试、Lint、人工 Review 触发器）。
3. Tools & Environment：给 Agent 提供的工具链、沙箱、API、版本控制集成。
4. Feedback Loops（反馈循环）：Agent 出错 → 自动记录 → 下次改进（让错误不再重复）。
5. Orchestration（编排）：多 Agent 协作、分层任务分解、人类监督点。

先写.md，制定目标，说明仓库结构，要实现的约束

再写具体的实现细节，尽量细致

再写验收标准

**第12周：2026年6月29日-2026年7月5日**

https://github.com/Denys88/rl\_games

<https://github.com/leggedrobotics/rsl_rl>

两个机器人的库

https://isaac-sim.github.io/IsaacLab/main/index.html

https://docs.robotsfan.com/isaaclab/IsaacLab安装时的中文标准网站，点击左边栏开始-建立自己的项目或任务得教程

结论先说：unitree和amp的仓库的“奖励函数”层级不一样。

legged\_robot\_config.py:101 是典型的 locomotion reward：先在配置里定义各项 scale，再由 legged\_robot.py:163 自动收集同名的 reward\* 函数来算总奖励。核心项包括 tracking\_lin\_vel、tracking\_ang\_vel、feet\_air\_time、stand\_still、collision、action\_rate、torques、dof\_vel、dof\_acc、base\_height 和 termination。这个版本本质上是在“学会走得稳、跟得准、耗能低”。

go2\_secamp\_env.py:325 更像“任务奖励 + 模仿奖励”的混合。任务侧是 tracking\_pos：用世界系下的速度投影去追 waypoint，且按技能限制速度上限；模仿侧在 amp\_discriminator.py:11 里由判别器输出 AMP reward，再和 task reward 按 lerp 混合。配置信息在 go2\_secamp\_config.py:171 里，说明它训练时会先靠 imitation 稳住动作，再逐步让任务奖励把策略拉向目标 waypoint 和速度范围。

README.md:1 不是训练仓库，所以这里基本没有奖励函数。它的 rl\_policy.py:1 是直接读取传感器、拼 observation、加载 TorchScript policy、输出目标关节角；deploy\_secamp.py:1 也是同样逻辑，只是支持 waypoint/joystick 的部署控制。也就是说，这里关注的是“怎么把训练好的策略安全地跑起来”，不是“怎么定义训练目标”。

如果你要跟学长解释，可以直接概括成一句话：my\_unitree\_rl\_gym\_V1 是传统 locomotion 奖励，my\_AMP\_for\_hardware\_V1 是带 AMP 模仿先验的 waypoint 训练奖励，而 my\_Deploy\_Go2Robot\_V1 只负责部署，不再计算训练奖励。

详解：

我先把你点名的几段代码按“奖励定义、AMP 混合、部署推理”三条线对齐，补足每个术语在代码里的实际含义，然后逐条给你翻译成训练时会发生什么。

1. 各项 scale 是干什么的

在 legged\_robot\_config.py 里，`rewards.scales` 其实就是“每一项奖励/惩罚的权重”。训练时，代码会先算出每个 reward 函数的原始值，再乘上对应的 scale，最后加总成总奖励。你可以把它理解成：决定这项指标在训练里有多重要，以及它是奖励还是惩罚。

在 legged\_robot.py 的 `compute\_reward()` 里，每个项都按 `rew = reward\_function() \* reward\_scale` 进入总和，所以：

- 正数 scale = 鼓励这项行为

- 负数 scale = 惩罚这项行为

- 绝对值越大 = 影响越强

另外，`only\_positive\_rewards = True` 会把总奖励裁成不小于 0，避免训练早期大量负值把学习搞得很不稳定。

2. 这些核心项分别是什么奖励

对应的具体实现都在 legged\_robot.py 到 legged\_robot.py。

- tracking\_lin\_vel

这是线速度跟踪奖励，目标是让机器人 base 的 xy 线速度接近命令速度。代码里是对 `commands[:, :2]` 和 `base\_lin\_vel[:, :2]` 的误差做指数函数，误差越小奖励越大。

- tracking\_ang\_vel

这是角速度跟踪奖励，目标是让机器人 yaw 角速度接近命令值。代码里是 `commands[:, 2]` 和 `base\_ang\_vel[:, 2]` 的误差。

- feet\_air\_time

这是“抬脚时间”奖励。意思是：机器人在走路时，脚在空中的时间合适、落地节奏自然时会得分。它只在命令不为 0 时才给奖励，避免站着不动也拿到这项奖励。

- stand\_still

这是静止惩罚。命令接近 0 时，如果机器人关节还在乱动，就会被惩罚。它的目的就是让机器人在没有指令时站稳。

- collision

这是碰撞惩罚。某些被标记为 penalised 的身体部位如果碰到地面或其他物体，就会扣分。它在鼓励动作干净，避免摔、蹭、刮。

- action\_rate

这是动作变化率惩罚。当前动作和上一步动作差得越大，惩罚越大。它的作用是让控制更平滑，减少抖动。

- torques

这是力矩惩罚。力矩越大，惩罚越大。它通常用来鼓励省力，避免策略用特别猛的控制硬拽机器人。

- dof\_vel

这是关节速度惩罚。关节转得越快，惩罚越大。它能抑制不稳定、高频振荡。

- dof\_acc

这是关节加速度惩罚。代码里是相邻两步关节速度差除以 dt，所以本质是在惩罚“加速度太猛”。

- base\_height

这是机身高度惩罚。base 的实际高度偏离目标高度 `base\_height\_target` 越多，惩罚越大。它帮助机器人保持合适的身体高度。

- termination

这是终止惩罚或终止奖励。代码里是 `reset\_buf \* ~time\_out\_buf`，也就是非超时导致的重置才算终止项。通常表示摔倒、失稳这类提前结束。

你在配置里看到的 `termination = -0.0`、`orientation = -0.` 这类值，如果是 0 或接近 0，意思就是这项在当前配置下几乎没开或者权重很弱。

3. tracking\_pos 是什么意思，连代码一起解释

这个是 go2\_secamp\_env.py 里的任务奖励。

代码逻辑是：

- `v\_world = self.root\_states[:, 7:9]`

取机器人在世界系下的 xy 速度。

- `delta = self.pos\_target\_world - self.root\_states[:, :2]`

算当前位置到 waypoint 的世界系 xy 偏移。

- `d\_w = delta / norm(delta)`

把这个偏移归一化，得到“朝 waypoint 的单位方向”。

- `proj = (v\_world \* d\_w).sum(dim=-1)`

把速度投影到 waypoint 方向上。

这一步的意思是：只看机器人有没有沿着 waypoint 的方向往前走，不看横着抖不抖。

- `skill\_idx = self.skill\_cmd\_buf.argmax(dim=-1)`

看当前是哪一种技能，比如 pace / trot / canter。

- `v\_max = vmax\_table[skill\_idx]`

给不同技能设不同的速度上限，比如 pace 1.0、trot 1.5、canter 3.0 m/s。

- `return torch.clamp(proj, max=v\_max)`

最后把奖励上限截住。也就是说：

- 朝 waypoint 方向走，投影越大越好

- 但超过这个技能允许的速度后，奖励不再继续涨

- 这样策略不会为了追奖励而一味飙到不适合该 gait 的速度

你可以把它理解成一句话：不是“往某个绝对速度跑”，而是“朝 waypoint 方向有效前进，并且不要超过这个 gait 的合理速度上限”。

4. lerp 是什么，连代码一起解释

`lerp` 就是 linear interpolation，中文叫线性插值。这里它不是图形学里的插值玩具，而是“两个 reward 按比例混合”的系数。

在 amp\_discriminator.py 里：

- `amp\_reward = ...`

这是判别器给出来的 AMP imitation reward。

- `combined\_reward = self.\_lerp\_reward(amp\_reward, task\_reward.unsqueeze(-1))`

把 AMP reward 和 task reward 混合。

- `\_lerp\_reward()` 里是：

`r = (1.0 - self.task\_reward\_lerp) \* disc\_r + self.task\_reward\_lerp \* task\_r`

这句话的意思就是：

- 如果 `task\_reward\_lerp = 0`，那就是纯 AMP reward

- 如果 `task\_reward\_lerp = 1`，那就是纯 task reward

- 如果是 0.3，那么就是 70% AMP + 30% task

所以你之前那句“再和 task reward 按 lerp 混合”，本质就是“按比例加权求和”。

5. “先靠 imitation 稳住动作，再逐步让任务奖励拉向 waypoint 和速度范围”是什么意思

这个说法对应的是 go2\_secamp\_runner.py 的训练逻辑。

代码里有一段：

- 初始 `lerp\_init = 0.15`

- 最终 `lerp\_final = amp\_task\_reward\_lerp`

- 如果 `lerp\_schedule\_enabled = True`，训练过程中会把 `task\_reward\_lerp` 从 0.15 慢慢线性推到最终值

意思是：

- 训练初期，策略还很菜，先主要学“像示范动作一样动”

- 这样能避免一开始就只盯着 waypoint 乱跑，学出奇怪动作

- 等动作先学稳了，再逐步提高任务奖励比例，让策略更重视“往目标点走、按技能速度走”

但这里要注意一个现实细节：在 go2\_secamp\_config.py 里，`lerp\_schedule\_enabled = False`。这说明当前配置并不会真的动态 ramp，而是直接用固定的 `amp\_task\_reward\_lerp = 0.3`。

所以“逐步拉向任务奖励”是这套代码支持的训练思路，但你当前这份配置是关闭调度、固定比例混合。

6. rl\_policy.py 是什么意思，连代码一起解释

rl\_policy.py 不是训练代码，而是部署推理代码。它做的事情很简单：

- 从机器人或仿真读传感器数据

- 把这些数据拼成和训练时一样的 observation

- 把 observation 喂给已经训练好的 TorchScript policy

- policy 输出动作

- 再把动作转换成目标关节角，发给机器人执行

代码里关键步骤是：

- `self.policy = torch.jit.load(str(policy\_path))`

加载已经导出的策略模型，不再训练。

- `self.qj[i] = self.low\_state.motor\_state[i].q`

读每个关节的角度。

- `self.dqj[i] = self.low\_state.motor\_state[i].dq`

读每个关节的速度。

- `gravity\_orientation = self.get\_gravity\_orientation(quat)`

把 IMU 四元数变成重力方向表示，这是观测的一部分。

- `self.cur\_obs[:3] = self.cmd \* self.config.cmd\_scale`

把控制指令放进 observation。

- `self.cur\_obs[3:6] = gravity\_orientation`

放入姿态信息。

- `self.cur\_obs[6:9] = ang\_vel`

放入角速度。

- `self.cur\_obs[9:21] = qj\_obs`、`self.cur\_obs[21:33] = dqj\_obs`、`self.cur\_obs[33:45] = self.action`

把关节状态和上一时刻动作拼进去，构成和训练时一致的输入。

- `self.action = self.policy(obs\_tensor).detach().numpy().squeeze()`

用策略网络直接预测动作。

- `self.target\_dof\_pos = self.config.default\_angles + self.action \* self.config.action\_scale`

把网络输出转换成目标关节位置。

- 最后再按机械臂/机器人自己的关节顺序重排后发布。

所以这份脚本的核心不是“算奖励”，而是“让训练好的策略在真实系统上工作”。

7. deploy\_secamp.py 也是同样逻辑，只是支持 waypoint/joystick 控制，这句话什么意思

deploy\_secamp.py 也是部署脚本，但它更接近你前面说的 AMP/SECAMP 方案。

它和 rl\_policy.py 一样，都是：

- 读传感器

- 拼 observation

- 加载 TorchScript policy

- 输出目标关节角

- 发布到 `/rl/target\_pos`

不同点是它多了两种“给策略下指令”的方式：

- waypoint 模式

在代码里用时间表自动切换方向和 skill。也就是策略不是手动操控，而是按预设路线跑。

- joystick 模式

左摇杆控制方向，按键切换 pace / trot / canter。

这只是“部署时怎么给 policy 提示”的接口，不是训练奖励本身。

你看到的这段：

- `obs = np.concatenate([...])`

- `self.policy(obs\_t)`

- `target\_isaac = self.last\_action \* self.config.action\_scale + self.config.default\_angles`

- `target\_robot = target\_isaac[REORDER]`

意思就是：

1. 把观测拼好

2. 把它送进已经训练好的网络

3. 网络给出动作

4. 动作变成目标关节角

5. 再转成机器人实际需要的关节顺序

所以“这里关注的是怎么把训练好的策略安全地跑起来，而不是怎么定义训练目标”，就是这个意思：

训练目标已经在别的仓库里学好了，这里只负责把那个结果稳定地接到真实机器人/仿真上。

**第13周：2026年7月6日-2026年7月12日**

2026年7月4日组会

我的工作：

阅读了unitree的官方库，并clone到本地，查找了reward func.是由一系列的奖励函数堆积而成；而amp是有任务环境position-tracking+amp的discriminator的输出奖励而成

IsaacGym到IsaacLab迁移新创项目文件夹之类的，codex完成：https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html

开会提出的问题：

1. 为什么要用基于waypoint的tracking而不是跟踪速度或其他，（好像是因为这样设置什么速度都能切换）董博批评说还是没完全懂天赐学长的思路，要多问个为什么！
2. 泽霖学长和董博共同意见，要训练效果好，amp的结构没毛病，主要是数据集要洗干净，数据集量大质好，可以多看看其他数据集，比如walk like dog论文中的数据集，还有kine2GO的数据集（https://huggingface.co/datasets/MIMUW-Robotics/kine2go），好的数据集比设置奖励函数强，因为amp本身的架构是很好的训练步态的奖励源，

写论文时小技巧

1. 文字内容上重要的是baseline上比不同，然后他们的astra如果脊柱锁死和主动脊柱的效果有多明显
2. 整篇论文最重要是框架图，就是正篇文章都是围绕图展开
3. 论文图用GPT生成，再不行手动用PPT画，用blender渲染，或者其他画图工具（draw.io、tikz）

**然后是复习备考考试**

**第17周：2026年8月3日-2026年8月9日**

本周工作：

1. 成功完成isaacSim迁移
2. Git Repo管理
3. 获取并分析其他动作数据库
4. 着手训练，明白内部流程及原理

董博提醒：

1. amp之前有一个小型的神经网络，架构不太好，看看怎么改进
2. Reward function怎么设置，有多少项

**sim2sim（isaacSim）**

1. 首次安装

打开 AnyDesk 中 Ubuntu 桌面的终端：

conda activate env\_isaacsim

cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

/home/tumi6/Repo/isaacLab/isaaclab.sh -p -m pip install -e source/go2\_amp\_isaaclab

/home/tumi6/Repo/isaacLab/isaaclab.sh -p -m pip install -e ../rsl\_rl

二、确认 Isaac Lab 任务已注册

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/list\_envs.py

正常应显示：

Isaac-Go2-AMP-Direct-v0

Isaac-Go2-SECAMP-Direct-v0

Isaac-Go2-Rough-Residual-Direct-v0

三、先启动 Isaac Sim 做环境检查

下面命令会打开 Isaac Sim 窗口，使用零动作运行 SECAMP 环境；窗口自动结束前不要关闭终端：

cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

conda activate env\_isaacsim

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/zero\_agent.py --task Isaac-Go2-SECAMP-Direct-v0 --num\_envs 2 --steps 200

最后出现 PASS 即表示 Isaac Sim、Go2 资产、接触传感器和 SECAMP 环境正常。

四、启动 Isaac Sim 可视化训练（SECAMP）

不要加 --headless，Isaac Sim 会显示训练窗口：

cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

conda activate env\_isaacsim

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --num\_envs 10

建议第一次先限制为两轮，确认训练、奖励和窗口都正常：

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --num\_envs 10 --max\_iterations 2

五、正式无界面训练（SECAMP）

关闭 Isaac Sim 可视化以提高速度：

**cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab**

**conda activate env\_isaacsim**

**/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --headless --num\_envs 500 --max\_iterations 200**

默认训练 50000 次迭代。训练日志和 checkpoint 写入：

/home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/logs/go2\_secamp\_isaaclab/

六、其他任务

基础 AMP：

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-AMP-Direct-v0 --headless --num\_envs 500

Rough residual：

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-Rough-Residual-Direct-v0 --headless --num\_envs 500

完成情况

1. play.py 手柄输入检查

Lab\_my\_AMP\_for\_hardware\_V1/isaaclab/scripts/play.py 已经实现了手柄控制（--mode joystick），且映射与旧版 ROS2 /joy 控制完全一致（对照 Lab\_my\_Deploy\_Go2Robot\_V1/src/deploy\_rl\_policy/scripts/deploy\_secamp.py 和 low\_level\_ctrl.cpp 逐条核对）：

左摇杆：前后=轴1，左右=轴0（含死区）

L1(4)+R1(5) 同时按住 = 使能控制

X(0)/O(1)/△(2) = 切换 pace/trot/canter

L2+R2 同时拉到底 = 退出

不需要额外改动，代码已完整可用。

2. Headless 运行 — 修复了两个环境问题

isaaclab 之前的可编辑安装指向了旧路径 /home/tumi6/Repo/IsaacLab（大写），而实际目录已改名为 isaacLab（小写），导致 import isaaclab 失败 → 用 ./isaaclab.sh --install 在新路径重新安装。

系统里存在官方 rsl-rl-lib 包，与本项目自定义的 rsl\_rl（含 AMPOnPolicyRunner/Go2SECAMPRunner 等）包名冲突，导致一直加载到官方版而报 ImportError → 卸载 rsl-rl-lib，重新以 --force-reinstall --no-deps 安装项目自己的 rsl\_rl。

顺带把因中间步骤被升级的 psutil 固定回 5.9.8（避免和 isaacsim-kernel/rl-games 冲突）。

修复后已验证通过：

zero\_agent.py --headless → PASS

train.py --headless --max\_iterations 3 → 正常输出 loss/reward

play.py --headless（自主模式，用刚训练出的 checkpoint）→ 正常跑完

当前状态：已经在后台启动了正式无界面训练（Isaac-Go2-SECAMP-Direct-v0, --num\_envs 500, 50000 次迭代），进程仍在运行，约 24000 steps/s，预计 ETA ~7 小时，日志在 /tmp/go2\_secamp\_train.log，checkpoint 写入 Lab\_my\_AMP\_for\_hardware\_V1/logs/go2\_secamp\_isaaclab/。

简化笔记（更新版）

一、首次安装（只需一次，或环境路径变化后重装）

conda activate env\_isaacsim

cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

/home/tumi6/Repo/isaacLab/isaaclab.sh -p -m pip install -e source/go2\_amp\_isaaclab

/home/tumi6/Repo/isaacLab/isaaclab.sh -p -m pip install -e ../rsl\_rl

若 import isaaclab 失败（路径改名等）：

cd /home/tumi6/Repo/isaacLab && ./isaaclab.sh --install

若报 rsl\_rl 缺 AMPOnPolicyRunner（和官方 rsl-rl-lib 撞包名）：

pip uninstall -y rsl-rl-lib

cd /home/tumi6/Repo/ProjectsTest\_Lingheng\_Kong/Lab\_my\_AMP\_for\_hardware\_V1/isaaclab

/home/tumi6/Repo/isaacLab/isaaclab.sh -p -m pip install -e ../rsl\_rl --force-reinstall --no-deps

pip install "psutil==5.9.8" # 避免版本冲突

二、确认任务已注册

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/list\_envs.py

三、Headless 冒烟测试

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/zero\_agent.py --task Isaac-Go2-SECAMP-Direct-v0 --num\_envs 2 --steps 200 --headless

四、正式无界面训练

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Go2-SECAMP-Direct-v0 --headless --num\_envs 500

日志/checkpoint：Lab\_my\_AMP\_for\_hardware\_V1/logs/go2\_secamp\_isaaclab/

五、Play（回放/导出），支持手柄

自主模式

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/play.py --task Isaac-Go2-SECAMP-Direct-v0 --checkpoint <ckpt.pt> --num\_envs 1

手柄模式（PS4手柄，映射与旧版完全一致：L1+R1使能，摇杆定方向，X/O/△切换步态，L2+R2退出）

/home/tumi6/Repo/isaacLab/isaaclab.sh -p scripts/play.py --task Isaac-Go2-SECAMP-Direct-v0 --checkpoint <ckpt.pt> --num\_envs 1 --mode joystick

**环境介绍**

它们不是仿真本体，而是不同用途的 Python/conda 环境。真正的仿真平台是：Isaac Gym / Isaac Sim / MuJoCo；环境只是把对应库装好。

按你这些名字，通常可以这样理解：

环境 主要适配

env\_isaacgym Isaac Gym

kongstest\_env\_issac\_gym 也是 Isaac Gym（名字里写错了 issac）

env\_isaacsim Isaac Sim / Isaac Lab

kongstest\_env\_amp\_hardware 多半是硬件/部署/AMP 代码环境，不专指某一个仿真

kongstest\_env\_deploy 多半是部署/机器人运行环境，常见会配 ROS2、MuJoCo 或实机代码

关系可以简单记成：

Isaac Gym：老一代 NVIDIA 机器人仿真

Isaac Sim：新一代，基于 Omniverse

Isaac Lab：跑在 Isaac Sim 上的训练框架

MuJoCo：另一套独立仿真器，和 Isaac 系列不是一回事

所以：

训练 Lab\_my\_AMP\_for\_hardware\_V1/isaaclab 这套，主要用 env\_isaacsim

旧的 isaacgym 相关代码，用 env\_isaacgym

deploy\_mujoco / 部署侧脚本，通常更偏 MuJoCo / ROS2 / 实机

**Reward function设置巧思**

取材GPT-SA2

可以。你这个问题其实正好是理解 AMP / SECAMP 整个训练框架最关键的一环。我先基于你上传的学长 thesis 把 Reward 的逻辑完整拆开。这里尤其要注意：这个 Reward 不是单纯的“机器人跑得越快奖励越高”，而是两条完全不同的奖励通道最后汇合。

先给你整个数据流

你现在可以先把整个训练过程脑子里想成：

┌─────────────────────┐

│ Motion Dataset M │

│ "walk like dog" │

└──────────┬──────────┘

│

▼

Expert Motion

│

▼

┌─────────────────┐

│ Discriminator D │

└────────┬────────┘

│

imitation

reward

│

│

▼

Observation ──→ Policy π ──→ Action ──→ IsaacGym

▲ │

│ │

│ ▼

│ State'

│ │

│ ▼

│ Task Reward

│ │

└──── Total Reward ◄────┘

│

▼

PPO

│

▼

更新 Policy π

更准确一点，是：

Motion Dataset

│

▼

Expert transitions

│

│

▼

Discriminator

▲

│

│ Policy transition

│

Robot State ──→ Policy ──→ Action

│ │

│ ▼

│ IsaacGym

│ │

│ ▼

│ State t+1

│ │

│ ├────────→ Task Reward rT

│ │

│ └────────→ Imitation observation

│ │

│ ▼

└──────────────────→ Discriminator

│

▼

rI

│

┌─────────────────┴───────────────┐

│ │

▼ ▼

rT × ωT rI × ωI

│ │

└────────────┬────────────────────┘

▼

Total Reward r

│

▼

PPO

│

▼

Update π这个数据流基本就是你之后从 Isaac Gym 迁移到 Isaac Lab 时最应该保留的逻辑。

Reward 到底由什么组成？

你这套 SECAMP 最核心的公式是：r = ωT rT + ωI rI

也就是：总奖励 = Task Reward + Imitation Reward

论文明确把它定义为 task completion 和 imitation consistency 两部分的加权组合。

在 appendix 给出的最终参数里：

task reward weight = 0.3

imitation reward weight = 0.7

也就是说：r = 0.3rT + 0.7rI

这就非常重要了。

第一条：Task Reward 是干什么的？

Task Reward：rT 负责告诉机器人：“你有没有朝着我要你去的地方走？”而不是告诉机器人：“你的腿是不是像狗一样动。”后者是 AMP 的工作。

为什么 Task Reward 不直接用 velocity tracking？

传统四足 RL 很喜欢：rv = exp(-|v-v\_target|²)

比如：

target velocity = 1.5 m/s

robot = 1.4 m/s → reward高

robot = 0.5 m/s → reward低

robot = 2.5 m/s → reward低

但是你这个项目有：pace, trot, canter

不同 gait 本来就有不同的自然速度范围。

论文里给出的典型速度大概是：

pace ≈ 1.0 m/s

trot ≈ 1.5 m/s

canter ≈ 2.5 m/s

如果你强行让三个 gait 都追同一个 velocity：target = 1.5 m/s

那就会出现：

pace: 自然速度 1.0 ↓ 被迫跑 1.5

canter: 自然速度 2.5 ↓ 被迫降到 1.5

这和“学习不同自然 gait”是冲突的。

所以它用了 Waypoint Progress Reward

这是你这个项目很关键的设计。

机器人不是：“我要达到 1.5m/s”

而是：“我要往这个 waypoint 方向走。”

首先计算：d\_wp\_bar = (wp - p) / |wp - p|

其中：

p = robot base 当前世界坐标

wp = waypoint 世界坐标

也就是：从机器人当前位置指向 waypoint 的 normalized direction。

然后看机器人速度在这个方向上的投影

机器人速度：v

目标方向：d\_wp\_bar

做 dot product：⟨v, d\_wp\_bar⟩

其实就是：机器人当前速度，有多少是在“朝目标走”？

例如：

目标在 → 方向

robot velocity → →→→→→ dot = 很大 reward = 高

如果：

目标在 →

robot velocity ↑ ↑ ↑

机器人在横向跑：dot ≈ 0 reward ≈ 0

如果机器人甚至往反方向：

目标 → robot ←

那么：dot < 0

最终 Task Reward

论文给出的公式是：rT = min(⟨v, d\_wp\_bar⟩, vmax(cS))

这个设计特别聪明。

假设：trot vmax = 1.5 m/s

那么：

机器人朝目标 0.5 m/s → reward = 0.5

机器人朝目标 1.0 m/s → reward = 1.0

机器人朝目标 1.5 m/s → reward = 1.5

机器人朝目标 2.5 m/s → reward = 1.5

也就是说：超过这个 gait 的合理速度以后，不再额外奖励。

论文也明确解释了这一点：达到该 skill 的 nominal speed 后，继续加速不会获得额外 reward，从而避免机器人为了 reward 疯狂加速。

这就是为什么它不会把“跑得越快”当成唯一目标

整个逻辑变成：

Task Reward → 朝 waypoint 前进？ → velocity projection → skill-specific velocity ceiling

所以：

pace ↓ 允许比较慢

trot ↓ 允许中等速度

canter ↓ 允许高速

这样 gait 和 task 就不会互相打架。

第二条奖励：Imitation Reward

这才是你这个项目真正的 AMP 核心。

它叫：rI (Imitation Reward)。

它解决的是：“你虽然往 waypoint 走了，但是你走得像不像参考的狗？”

比如 Task Reward 只存在：前进

那么机器人可能学出来：四条腿疯狂划拉，身体上下乱晃，但是速度非常快。

从 Task Reward 看：很好 👍

但从动物运动角度：完全不像狗 ❌

论文也明确指出，仅优化 task performance 会产生功能上有效、但 gait pattern 不自然的运动。

所以 Discriminator 登场

这个东西你一定要理解。

它不是直接告诉机器人：“膝盖应该抬多少。”

它做的是一个分类任务：

Motion Transition → Discriminator Dφ → score d

它学习：Expert motion vs Policy motion

Expert 从哪里来？

就是你的：walk like dog motion dataset

比如一段狗走路：t0 → t1 → t2 → t3 (state → state → state → state)

从里面抽：(ŝt, ŝt+1) 作为 Expert Transition。

Policy 自己也产生 Transition

机器人实际跑：st → at → st+1

所以有：(st, st+1)

然后 discriminator 同时看：

Expert: ŝt → ŝt+1 → D

Policy: st → st+1 → D

论文明确说 discriminator 输入的是连续 observation pair / transition，而不是单独一个 state，因此它可以捕捉 limb coordination、body oscillation、gait timing 等时间上的运动特征。

为什么不是只看一个 state？

这个很关键。

假设某一瞬间：四条腿的位置看起来很像狗。

但是：t0 → t1 腿运动的顺序完全错了。

那它其实不是自然 gait。

所以：state 只能告诉你：“这一帧长得像不像。”

而：state\_t → state\_t+1 才能告诉你：“这一帧是怎么运动到下一帧的。”

所以 AMP discriminator 看 transition。

Discriminator 输出什么？

它输出：d = Dφ(fI(st), fI(st+1) | cS)

也就是说：

st → fI → imitation observation

st+1 → fI → imitation observation

→ Discriminator → d

而且还有：skill code cS

所以它不是单纯问：“像不像狗？”

而是：“在当前要求的 skill 下，这个运动像不像对应的 reference motion？”

论文明确强调 discriminator 是 skill-conditioned 的。

然后 d 怎么变成 reward？

论文给的是：rI = max(0, 1 - 0.25(d-1)²)

也就是 discriminator 给出的 score 经过一个 reward transformation，最终限制在 [0,1]

你可以简单理解成：

motion → Discriminator → "像不像 expert?" → score d → reward shaping → rI

最后两条线汇合

现在整个 Reward 就非常清楚了：

Task Reward (rT) × 0.3

Imitation Reward (rI) × 0.7

=

Total Reward (r)

所以：r = 0.3rT + 0.7rI

但是最容易搞混的一点：Discriminator 不是 Policy

这一点你迁移 Isaac Gym → Isaac Lab 的时候尤其重要。

有两个网络：

Policy (πθ)

负责：observation → Policy → action → robot

它最终要部署到 Go2 上。

Discriminator (Dφ)

负责：robot motion → Discriminator → imitation reward

它不是最终部署到机器人上的控制器。

PPO 又在哪里？

这里就是整个 RL loop 闭环：

Observation → Policy πθ → Action → IsaacGym → State t+1

State t+1 → Task Reward

State t+1 → Discriminator → rI

Total Reward → PPO → 更新 Policy π

论文的 training pseudocode 也是这个顺序：先 rollout，计算 task/imitation reward，存 transition；然后 discriminator 更新；最后 PPO 更新 policy 和 value function。

这里还有一个非常重要的“双训练”

你现在千万不要理解成：PPO训练一个网络，Discriminator只是辅助。

实际上每轮 training 都在同时更新两个东西：

第一条：

Expert motion + Policy motion → Discriminator → 更新 Dφ

让 discriminator 越来越会分辨：Expert vs Robot

第二条：

Total Reward → PPO → 更新 Policy π

而 policy 又努力让自己的 motion：越来越像 Expert

于是形成：

Discriminator ↑

│

Expert ←→ Policy

│

↓

Robot motion

这就是 AMP 的 adversarial loop。

为什么最终机器人会“像狗一样走”？

现在你就可以理解了。

假设 robot 学到了一个非常奇怪的策略：跑得很快

那么：Task Reward ↑

但是：Imitation Reward ↓ (因为不像 expert)

所以：r = 0.3rT + 0.7rI 并不会特别高。

反过来：

机器人走得非常像狗

那么：Imitation Reward ↑↑

但是如果它：原地走

那么：Task Reward ↓

所以总 reward 还是不高。

最终 Policy 被逼到一个折中点：

既要完成任务，又要保持 reference motion 的运动风格。

这正是 AMP 最大的意义。

你现在迁移 Isaac Lab，Reward 应该怎么理解？

这个对你当前工作特别重要。

不要把 Reward 理解成 Isaac Gym 的东西。

实际上 Reward 是你这个 RL algorithm 的逻辑：

RL Algorithm

├── Task Reward

└── AMP Reward (Discriminator)

→ Total Reward → PPO

所以从 Isaac Gym → Isaac Lab：

不应该改变：

rT 的数学逻辑

rI 的数学逻辑

r = 0.3rT + 0.7rI

Discriminator

PPO

主要改变：

IsaacGym Environment API → Isaac Lab Environment API

也就是：

Gym: compute\_reward(), compute\_observations(), step()

↓ migration

Lab: Observation terms, Reward terms, Termination terms, Action terms, Event terms

数学逻辑基本不变，代码接口和数据组织方式变。

你可以先记住这一张图

如果你现在要开始看学长的代码，我建议你脑子里一直放着这个：

Motion Dataset → Expert motion → Discriminator D

Observation → Policy → Action → Robot → State'

State' → Task Reward (rT)

State' → Imitation observation → Discriminator → rI

0.3 rT + 0.7 rI → PPO → 更新 Policy

其中最核心的一句话就是：

Task Reward 管“往哪儿走”，Imitation Reward 管“怎么走”。

而你现在这个 SECAMP 的创新，本质上就是在这个基础上再加入：

skill code → Policy → 选择不同 gait

以及：

skill code → Discriminator → 要求 motion 与对应 gait 的 reference 匹配

所以才可以做到 pace / trot / canter 一个 policy。论文也明确把 task reward 与 imitation reward 的冲突作为 multi-skill locomotion 的核心问题。

不过现在有一个很关键的下一步：你刚才问的是“这里的 Reward function 怎么设置”，如果你指的是你项目里实际代码中的 reward 函数，那我还需要把 my\_AMP\_for\_hardware\_V1 里的具体 compute\_reward / reward\_scales / AMP reward 代码找出来。现在我检索到的文件主要是学长 thesis，所以我能准确解释论文层面的 Reward，但还不能假装已经看到了你代码里的每一个 reward term。

如果你把项目代码文件夹/zip 上传过来，我可以直接按代码调用链给你拆成：

哪个文件 → 哪个函数 → 拿了哪个 observation → 算出了哪个 reward → 进入哪个 buffer → PPO 怎么拿到 → 最后怎么更新 Policy

这会正好对应你现在 Isaac Gym → Isaac Lab 迁移最需要搞懂的东西。

AMP + AAC + PPO 架构，那么不能简单理解成“AMP 前面还有一个额外的小神经网络”。

更准确地说，要区分 AMP 本身的 discriminator 和 Policy 前面的 observation encoder / 小网络

![](data:image/png;base64...)

**第18周：2026年8月10日-2026年8月16日**

1下载markitdown省token

**markitdown指令**

conda activate markitdown

markitdown “C:\Users\ASUS\Desktop\SA计划攻略.docx” -o “C:\Users\ASUS\Desktop\Semester Arbeit\Projects\noteExplain.md”

2重新学习架构以及数据流

我已经把论文、训练端和部署端的主数据流对上了。先给你一个最重要的结论：

> 这个项目本质上是：用动作捕捉数据教 Go2“怎么自然地走”，用强化学习教它“往哪里走”，最后只把 Actor 策略网络部署到 MuJoCo/真机。

## 1. 项目总架构

```text

动作捕捉数据 datasets/camp

│

▼

SECAMP Motion Loader ───────────────┐

提供 pace/trot/canter 专家动作 │

▼

Isaac Lab 中 4096 个并行 Go2 ──► 条件判别器 D

│ 判断动作是否像指定步态

│

├─ Actor：根据传感器和命令输出 12 个动作

├─ Critic：训练时评估动作长期收益

└─ Reward：导航奖励 + 模仿奖励

│

▼

PPO 更新策略

│

▼

checkpoint model\_xxx.pt

│

▼

导出 TorchScript + JSON

│

▼

MuJoCo → ROS2 → Unitree Go2

```

**两个项目文件的分工**

是：

- `go2\_RL\_AMP`：训练、动作数据、网络、PPO、AMP/SECAMP、策略导出。

- `go2\_Deploy`：构造真机观测、运行导出策略、关节重排、ROS2 状态机、MuJoCo/真机控制。

这三个词其实合起来就是"训练好的大脑，怎么真正指挥机器人的腿"这一整套流程。结合你文档里 `go2\_Deploy` 部分的具体代码，一个一个拆开讲。

## 1. 运行导出策略（Run the exported policy）

\*\*背景\*\*：`go2\_RL\_AMP` 训练完之后，用 `play.py` 把训练好的神经网络"导出"成两个文件：

- `go2\_secamp.pt` —— 网络本身（TorchScript 格式，脱离训练框架也能跑）

- `go2\_secamp.json` —— 说明书（关节顺序、输入输出维度、控制频率等）

\*\*"运行导出策略"就是指\*\*：在 `go2\_Deploy` 这边，写一个脚本（也就是文档里的 `deploy\_isaaclab.py`）把这个 `.pt` 文件加载进来，喂给它当前机器人的传感器数据，让它算出下一步的关节角度。

打个比方：训练阶段是"教一个人开车"，导出是把这个人的驾驶技能写成一本"操作手册"，而"运行导出策略"就是真的把这本手册拿出来，照着执行——传感器读数进去，动作指令出来。这一步本身不训练、不学习，只是"推理"（inference），纯粹的输入→计算→输出。

## 2. 关节重排（Joint reordering）

这是最容易踩坑、也最危险的一步，你文档第9节专门标出来了：

```

Isaac： FL, FR, RL, RR （训练时神经网络认的顺序：左前、右前、左后、右后）

Unitree：FR, FL, RR, RL （真实狗/官方SDK的顺序：右前、左前、右后、左后）

映射： [3,4,5, 0,1,2, 9,10,11, 6,7,8]

```

\*\*为什么需要重排\*\*：训练神经网络时，Isaac Lab 内部用了一套关节排列顺序（比如"第0-2号是左前腿"）。但宇树 Go2 真机（或 MuJoCo 仿真）的官方通讯协议，用的是另一套顺序（比如"第0-2号是右前腿"）。

如果不重排会怎样？神经网络本来想让"左前腿"做某个动作，结果这个指令被发到了"右前腿"的电机上——完全是错误的腿在乱动，机器人会瞬间摔倒甚至损坏。

所以流程是：

1. 从真机/仿真拿到的关节角度、速度是 Unitree 顺序 → 转换成 Isaac 顺序，喂给神经网络

2. 神经网络算出的动作是 Isaac 顺序 → 转换回 Unitree 顺序，才能发给真正的电机

这两次转换都发生在 `deploy\_isaaclab.py` 里，就是你文档里画的"Unitree顺序 → Isaac顺序"和"Isaac顺序 → Unitree顺序"那两行。

## 3. ROS2 状态机（ROS2 state machine）

对应你文档里的 `low\_level\_ctrl.cpp`。

\*\*先说 ROS2 是什么\*\*：可以理解成机器人内部各个程序之间"发消息"的邮局系统。比如"策略推理程序"算出了目标关节角度，不是直接去控制电机，而是发布一条消息（`/rl/target\_pos`），"底层控制程序"订阅这条消息后再去真正驱动电机。这样不同功能的程序可以独立开发、互不干扰，出问题也好排查是哪一段。

\*\*再说"状态机"（state machine）是什么\*\*：就是给机器人规定几种"模式"，同一时刻只能处于其中一种，并且规定了模式之间怎么切换。你文档里提到的模式包括：

- \*\*躺下\*\*（初始/安全姿态）

- \*\*站立\*\*（准备阶段，把腿撑起来）

- \*\*RL 运行\*\*（正式把神经网络算出来的动作发给电机）

为什么必须要有状态机？因为你不能让机器人一上电就直接跑神经网络——那样如果腿的初始姿势不对，网络可能瞬间输出一个离谱的动作，机器人会摔。所以必须严格按顺序：先安全地"躺下"，再缓慢"站立"到一个类似训练时的初始姿态，确认稳定后才切换到"RL"模式开始真正执行策略。状态机就是负责管这个切换逻辑、以及设置每个阶段该用什么样的 Kp/Kd（电机软硬程度）参数。

---

\*\*串起来看整条链路\*\*（对应你文档第8节的那张流程图）：

```

机器人/仿真发来传感器数据（LowState）

↓

deploy\_isaaclab.py：

- 读取IMU、关节角、关节速度

- 【关节重排】Unitree顺序 → Isaac顺序

- 拼成神经网络要的45维输入

- 【运行导出策略】把这45维喂给go2\_secamp.pt，网络输出12维动作

- 【关节重排】算出的动作 Isaac顺序 → Unitree顺序

↓

发布ROS2消息 /rl/target\_pos

↓

low\_level\_ctrl.cpp（【ROS2状态机】）：

- 判断当前处于躺下/站立/RL哪个状态

- 设置对应的Kp/Kd

- 发布最终的电机控制指令

↓

真正驱动MuJoCo仿真狗或真实Go2的电机

```

这就是为什么你的分工写"go2\_Deploy 负责关节重排、ROS2状态机、运行导出策略"——这三件事都发生在"网络已经训练好之后、真正上机器人"的这一段，跟训练本身没关系，纯粹是"怎么把算好的数字，安全、正确地变成电机转动"的工程问题。

论文对应的核心任务是 `go2\_secamp`。`go2\_amp` 是单技能基线；`go2\_rough` 是冻结 SECAMP 先验之上的崎岖地形残差策略，建议最后再学。

## 2. 零基础需要先懂的四个概念

- Observation：机器人当前“看到”的信息，例如身体倾斜、关节角、用户命令。

- Action：策略输出的 12 个数字，表示相对默认站姿的关节偏移。

- Reward：告诉策略这一拍做得好不好。

- Policy/Actor：从 observation 映射到 action 的神经网络。

SECAMP 又加入两个角色：

- Critic：训练时估计长期收益，帮助 Actor 学习；部署时删除。

- Discriminator：判断动作是否像指定步态；训练时提供模仿奖励，部署时也删除。

所以部署到机器人上的不是完整训练系统，只有 Actor。

## 3.

**Actor 的 45 维输入**

当前推荐的 Isaac Lab 实现位于 [go2\_env.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/source/go2\_amp\_isaaclab/go2\_amp\_isaaclab/tasks/go2\_amp/go2\_env.py:133>)。

| 索引 | 维度 | 内容 |

|---|---:|---|

| `0:3` | 3 | projected gravity，身体相对重力方向 |

| `3:6` | 3 | 机身坐标系下的目标方向，XY+0齐次坐标 |

| `6:18` | 12 | `当前关节角 - 默认关节角（设定的"标准站姿"）` |

| `18:30` | 12 | 关节速度 × 0.05 |//× 0.05是归一化（normalization）

| `30:42` | 12 | 上一拍实际 action |

| `42:45` | 3 | 步态 one-hot：pace/trot/canter |

重力在机身坐标系下的方向向量，3 维，对应机身自己的 x/y/z 三个轴。

在世界坐标系里，重力永远指向正下方，是一个固定不变的向量 [0, 0, -1]（单位化后）。但机器人自己不知道"世界的下方"在哪——它只有自己的身体作为参照。所以要把这个世界坐标的重力向量，旋转到机身坐标系，问的是：

从我身体自己的视角看，重力是往哪个方向拉我的？

几个具体数值，直观感受一下

机器人姿态 projected gravity 大致取值 含义

完全水平站立 [0, 0, -1] 重力垂直于我的身体底面，我是平的

头朝下俯冲（pitch 前倾） [-0.5, 0, -0.87] 重力有一个分量指向我的"后方"，说明我在往前倾

向左侧倾（roll） [0, 0.5, -0.87] 重力有一个分量指向我的"右侧"，说明我在往左歪

完全翻倒（肚皮朝天） [0, 0, +1] 重力指向我的"上方"，我翻过来了

-为什么不直接用姿态角（roll / pitch / yaw）？

① 避免万向锁和角度突变。 欧拉角在某些姿态下会出现数值奇异（万向锁）

② 天然剔除了 yaw（航向角）。 roll 和 pitch 决定"会不会摔"，yaw 只是"朝哪个方向"，跟平衡完全无关。

③ 表示更"平滑"，网络更好学。 向量的每个分量都在 [-1, 1] 之间连续变化，跟其他输入项的量级也接近，不需要额外归一化。

另外，我们看起来更直观；从 IMU 读姿态四元数，做一次坐标变换即可，不容易出错。

上一拍实际 action作用

1、观测本身缺少"我正在做什么"这个信息。

其他 44 维描述的都是结果：我现在什么姿态、关节在哪、转得多快。但有一个关键信息缺失——我上一刻发出了什么指令。

这两者不等价。因为 PD 控制器有响应延迟：你发出"转到 30 度"的指令，关节不会瞬间到 30 度，它需要时间。所以在某一时刻，关节可能在 20 度、正在往 30 度移动。如果网络不知道"目标是 30 度"，它只看到"现在 20 度、正在转"，就无法判断这个运动是不是自己造成的、还会持续多久。

给了上一拍 action，网络就知道"这个运动是我自己指挥的，且目标是 30 度"，可以据此规划下一步。

2、抑制动作抖动。

这是最主要的实际收益。如果没有这一项，每一拍网络都是独立做决策的——它不记得自己上一拍说了什么。理论上相邻两拍的观测很接近，输出也应该接近；但神经网络对输入的微小变化可能产生不小的输出变化，加上仿真里的随机噪声、域随机化，很容易出现相邻两拍指令跳来跳去的情况。

这在仿真里可能看起来只是"动作不太自然"，但在真机上是灾难：目标角度高频抖动会让电机反复正反转，产生剧烈震动和噪音，长期还会损伤减速器。

把上一拍 action 喂回去，网络就有了"参照点"——它倾向于在上一拍的基础上做小幅调整，而不是每次重新算一个绝对值。动作自然变得平滑连贯。

3、让策略具备有限的"记忆"。

Actor 是一个 MLP（前馈网络），本身没有任何记忆——它不像 RNN/LSTM 那样能记住历史。但周期性步态本质上是一个时序问题：现在该抬左前腿还是右前腿，取决于我处在步态周期的哪个相位。

上一拍 action 相当于给了这个无记忆的网络一个最小的历史窗口。虽然只有一步，但配合关节角和速度信息，足以让网络隐式地推断出"我现在处于步态周期的哪个阶段"，从而输出周期性连贯的动作。

这是用最低的成本（12 维输入）换取时序能力，比换成 RNN 便宜得多，也更容易部署（部署端只需要缓存上一次的输出，逻辑极简单）。

Actor 故意不使用**机身线速度和角速度，因为真机上这两个量不容易可靠获得**。

为什么偏偏是这 6 维被拿掉？

机身线速度：真机上只能靠 IMU 加速度积分得到。积分会让传感器噪声不断累积，几秒钟后估出来的速度就严重漂移。这是最典型的"仿真里免费、真机上昂贵"的量。

机身角速度：IMU 其实能直接测，比线速度可靠得多。但仿真值更干净无噪，且为了保持"Actor 只用低层本体感知"的一致原则，也被排除在外。

因此，不是"Actor 被削弱了"。 Actor 从来就不需要显式的机身速度——它可以从"关节角偏差 + 关节速度 + 上一拍动作"这些量里隐式推断出自己的运动状态（这本身就是网络要学的东西之一）

论文将其称为 asymmetric Actor-Critic，见 [Thesis.md](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/Thesis.md:1070>)。

**Critic 的 51 维输入**

在仿真里（Isaac Sim/Gym/MuJoCo），机身线速度和角速度是"作弊式"地拿到的——仿真引擎本来就精确知道机器人身体在虚拟世界里的每一个数据

训练时在 Actor 信息前再增加：

base linear velocity 3 //对应 x、y、z 三个方向

base angular velocity 3 //对应翻滚（roll）、俯仰（pitch）、偏航（yaw）

Actor 信息 45

总计 51

为什么这样能提升训练效果

Critic 的任务是"准确估计状态价值"。 而机器人的价值高度依赖它当前的运动状态——正以 1.2 m/s 朝目标平稳前进，和几乎静止但姿态相同，这两个状态的价值天差地别。可如果 Critic 看不到线速度，这两个状态在它眼里几乎一模一样，它只能给出一个模糊的平均估计。

Critic 估得不准 → 优势 A 算得不准 → Actor 收到的学习信号有噪声 → 收敛慢甚至学不出来。

给 Critic 补上这 6 维，它对状态价值的估计立刻清晰，Actor 的学习信号质量随之提升。

## 4. Waypoint 如何变成策略命令

论文不是直接命令“以 1.5 m/s 前进”，而是命令“目标在身体的哪个方向”。

训练环境进行：

```text

世界坐标目标位置 - 世界坐标机器人位置

│

▼

限制向量长度最大为 1 米

│

▼

从世界坐标旋转到机器人机身坐标

│

▼

乘以 0.5 //归一化，把数值压到一个网络更好处理的范围

//为了让这一项和其他输入项保持接近的数量级

│

▼

放入 observation45维中的[3:6]机身坐标系下的目标方向```

对应旧实现为 [go2\_secamp\_env.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/legged\_gym/envs/go2\_secamp/go2\_secamp\_env.py:165>)。

这种“clamped command”的意义是：

y = x (-1,1), 1 (>1)

- 目标很远时，命令长度最多为 1，避免超出训练分布。

- 目标小于 1 米时，命令长度随距离减小，策略可以自然减速。

**- 步态由独立的 skill code 决定，不再把“速度”和“步态”绑死。**

**这正是论文最核心的“任务与技能解耦”。**

## 5.

**Action到电机控制**

Actor 输出：

```text

a ∈ R¹²

```

训练代码执行：

```text

a = clamp(a, -1, 1)

q\_target = q\_default + 0.25 × a //x0.25是动作缩放（action scaling）经验调参数值

```

解释：

Actor 网络最后一层通常用 tanh 之类的函数，天然输出范围就是 [-1, 1]（a ∈ R¹²，加上 clamp 确保万无一失）。这个 a 本身没有物理单位，只是一个"意图强度"，网络也不知道它对应多少度。我们需要把这个 [-1,1] 的抽象值，变成"关节该转到哪个角度（弧度）"。做法是：在默认站姿基础上，加一个偏移量，这个偏移量最大是 0.25 弧度（约14.3度）

原因：

安全性：把动作限定为"围绕默认姿势的小幅度调整"，网络不可能一下子输出一个离谱的、让关节瞬间转到极限位置的指令，天然带了一层安全保护。

好学：网络只需要学"该往哪个方向、多用力地偏离标准站姿"，而不用从零学习"每个关节的绝对角度应该是多少度"，这是一种\*\*残差式控制（residual control）\*\*的思路，比直接学绝对值更容易训练收敛。

**残差式控制（residual control）**

不让网络从零开始学"绝对该怎么做"，而是让网络只学"在一个已知还不错的基础动作上，需要修正多少"。这个"修正量"就叫残差（residual）

基础动作（baseline）：机器人站立的默认关节角 q\_default。这是一个已知的、稳定的、不会摔倒的姿势，不需要学，是人为设定好的。

残差（network学的部分）：0.25 × a——网络只需要学"**在这个默认姿势基础上，往哪个方向偏一点、偏多少"，就能拼出走路、转身等各种动作**。

最终动作 = 基础动作 + 残差 = q\_default + 0.25 × a

**对比**一下如果不用残差式，会怎样：网络要直接输出"绝对关节角度"，也就是从完全空白开始学"站立是什么角度、走路第一步左前腿该在什么角度、第二步该在什么角度……"——网络等于要重新发明"怎么站得住"这件事,学习难度大得多,而且训练早期网络乱输出时，很容易直接输出一个让机器人瞬间摔倒或关节损坏的绝对角度。

而残差式的好处：

训练更容易：网络只需要学"小幅度修正"，起点已经是一个合理的姿势，不用从零摸索。

天然安全：因为残差被限制在 ±0.25，无论网络输出什么，最终动作都不会离"默认站姿"太远，天然带了保护。

更快收敛：训练初期网络参数是随机的，如果是残差式，随机输出也只是"站姿附近抖一抖"，不会直接摔倒，训练能正常进行下去；如果是绝对角度式，随机输出可能直接命令关节转到极限位置。

然后 PD 控制器计算：

```text

torque = Kp × (q\_target - q) - Kd × dq

```

解释：

组成弹簧-阻尼系统，PD控制本身，其实天然的输出就是力（或力矩），不是角度。位置控制模式，只是把这套算法包在了驱动器芯片里，每有偏差就算一下正负力矩，对外表现成"你给我角度目标，我内部自己搞定"

旧 Isaac Gym 的实现可见 [legged\_robot.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/legged\_gym/envs/base/legged\_robot.py:419>)；

Isaac Lab 版在 [go2\_env.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/source/go2\_amp\_isaaclab/go2\_amp\_isaaclab/tasks/go2\_amp/go2\_env.py:113>)。

频率关系是：

```text

物理仿真：200 Hz，dt = 0.005 s

decimation：4 //（每4个物理仿真步，才让策略重新更新算一次新动作）

策略运行：200 / 4 = 50 Hz

每 20 ms 生成一次新动作

```

## 6. SECAMP 奖励如何工作

**SECAMP 奖励**

**导航奖励r\_task**

```text

d = 指向 waypoint 的单位向量

progress = 世界坐标速度 · d //点积（dot product）物理意义：两个向量做点积衡量的是 "其中一个向量，在另一个向量方向上的投影分量有多大"

r\_task = min(progress, 当前步态速度上限)

```

其中速度上限：

- pace：1.0 m/s

- trot：1.5 m/s

- canter：3.0 m/s

这不是让机器人精确跟踪速度，只是防止它为了更多奖励无限加速。

代码在 [go2\_env.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/source/go2\_amp\_isaaclab/go2\_amp\_isaaclab/tasks/go2\_amp/go2\_env.py:176>)。

**模仿奖励r\_imitation**

**Discriminator的43维输入**

判别器每帧读取 43 维：

```text

关节角 12

机身坐标脚位置 12 // 4只脚 × 每只脚3个坐标（x, y, z）

//机身坐标系（body frame）：以机器人身体自己为原点、自己 的朝向为坐标轴

机身线速度 3

机身角速度 3

关节速度 12

机身高度 1

总计 43

//但其实`H=2`，所以判别器看到连续两帧，即 86 维运动片段，而不只是静态姿势。

还有加上skill（同维度）

对比AAC

| 索引 | 维度 | 内容 |

|---|---:|---|

| `0:3` | 3 | projected gravity，身体相对重力方向 |

| `3:6` | 3 | 机身坐标系下的目标方向，XY+0齐次坐标 |

| `6:18` | 12 | `当前关

节角 - 默认关节角（设定的"标准站姿"）` |

| `18:30` | 12 | 关节速度 × 0.05 |//× 0.05是归一化（normalization）

| `30:42` | 12 | 上一拍实际 action |

| `42:45` | 3 | 步态 one-hot：pace/trot/canter |

base linear velocity 3 //对应 x、y、z 三个方向

base angular velocity 3 //对应翻滚（roll）、俯仰（pitch）、偏航（yaw）

对比

没有"目标方向"没有"步态one-hot"没有"上一拍action"，因为它的唯一工作是判断"这一小段运动，看起来像不像真狗的动作"

多出来的：

1、机身线速度、角速度（各3维）：这里跟Critic的逻辑一样——判别器也只在仿真训练阶段运行，从不部署到真机，所以它同样没有"真机上测不准"的顾虑，可以放心用仿真里精确的速度数据。

2、关节角、关节速度用的是"原始值"而不是"相对默认站姿的差值"、不做×0.05归一化：这点文档里没有直接强调差异，但本质上，判别器要做的是跟专家数据库里存的动作做统计对比（**专家数据本身就是按什么格式存的，判别器就按同样格式读取**），所以它的特征提取方式，是跟着"专家动作数据集怎么记录的"来对齐的，**而不是跟着"Actor方便学习"的归一化方式来设计**的。这是两套完全独立的、各自服务于自己任务的特征工程。

3、机身高度（1维）：奔跑、行走时身体重心的起伏（比如奔跑时身体一颠一颠地往上蹦）是判断步态风格很重要的特征，这个信息对Actor决策没那么关键（Actor有关节角信息基本能间接反映），但对"判断动作真不真实"很直接，所以判别器单独加了这一维。

Projection discriminator 计算：

motion\_feature = F(两帧运动) //F 是一个小型神经网络（通常是几层全连接网络/MLP）， 把86维原始动作数据，映射成一个更抽象、更利于判别的 特征向量，不是简单的减法

skill\_feature = E(skill one-hot) //E 通常就是一个简单的线性层，维度是"skill种类数 × 特 征维度"，one-hot向量乘上这个矩阵，向量的维度跟 motion\_feature的维度

score = motion\_feature · skill\_feature //做点积，看投影,动作与它声称的步态配不配，但动作 本身却可能不自然

+ unconditional\_score //不看skill是什么，只问这段动作本身，像不像专家数 据库里自然、合理的运动（原始 AMP，步态无法区分）

SECAMP 带来的关键能力：步态与速度解耦。 传统方法里步态往往是速度的副产品（走快了自动变成跑）。SECAMP 用独立的 skill code 指定步态，速度上限只是各步态各自的封顶值（pace 1.0 / trot 1.5 / canter 3.0 m/s），于是可以做到"用 canter 的步态慢速移动"这种传统方法难以实现的事。

代码在 [secamp\_disc\_project.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/secamp\_disc\_project.py:55>)。

**总奖励**

目前实际为固定比例：

r\_total = 0.7 × r\_imitation + 0.3 × r\_task

//模仿权重更高（0.7），说明优先保证动作自然，其次才是把任务走完

你的理解基本正确。关键是把这三个东西分开：

- `D`：整个判别器

- `d`：判别器输出的一个原始标量分数

- `r\_imitation`：把原始分数 `d` 转换后得到的强化学习奖励

## 1. 判别器不是只有 F

完整的 SECAMP 判别器是：

\[

D(\tau,c\_S)=F(\tau)^TE(c\_S)+U(F(\tau))

\]

这里 `D` 是一个整体模块，由三个可训练部件组成：

```text

D = F + E + U

```

不是三个独立判别器，而是同一个判别器内部的三个子模块。

实际代码就是：

```python

feature = self.trunk(x) # F(τ)

embed = self.embed(z) # E(cS)

inner = (feature \* embed).sum(...) # F(τ)ᵀE(cS)

unconditional = self.amp\_linear(feature) # U(F(τ))

d = inner + unconditional

```

见 [secamp\_disc\_project.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/secamp\_disc\_project.py:55>)。

## 2. F、E、U 分别是什么？

### F：动作特征提取器

\[

h=F(\tau)

\]

其中 `τ` 是连续两帧运动：

```text

τ = [o\_amp(t), o\_amp(t+1)]

维度 = 43 × 2 = 86

```

代码中的 F：

```text

86

↓

Linear 86 → 1024

↓

ReLU

↓

Linear 1024 → 512

↓

ReLU

↓

h = motion\_feature，512维

```

所以 F 是两层隐藏 MLP，也叫 discriminator trunk。

F 不直接判断真假，它只负责把原始的86维运动转换为512维抽象运动特征。

它可能学习到类似这样的隐式特征：

```text

某两条腿是否同时运动

前后腿相位差

机身是否上下震荡

速度是否符合步态

脚的运动轨迹

动作是否连续

```

这些特征不是人工指定的，是训练出来的。

### E：技能嵌入

\[

e=E(c\_S)

\]

输入是3维 one-hot：

```text

pace = [1,0,0]

trot = [0,1,0]

canter = [0,0,1]

```

网络只有一层：

```text

Linear 3 → 512，无 bias

```

由于输入是 one-hot，这实际上相当于从一个矩阵中选取某一列：

```text

E([1,0,0]) → pace embedding

E([0,1,0]) → trot embedding

E([0,0,1]) → canter embedding

```

每种步态最终对应一个可学习的512维特征向量。

### U：无条件判别头

\[

u=U(F(\tau))

\]

U 也是神经网络层，但非常简单：

```text

Linear 512 → 1

```

没有隐藏层，也没有激活函数：

\[

U(h)=w^Th+b

\]

它回答的是：

> 不考虑声称的 skill，这段运动总体上是否像专家运动？

标准 AMP 可以近似看作只有：

\[

D\_{\mathrm{AMP}}(\tau)=U(F(\tau))

\]

SECAMP 在它上面增加了：

\[

F(\tau)^TE(c\_S)

\]

因此：

```text

标准 AMP：

运动特征 → 总体自然性

SECAMP：

运动特征 → 总体自然性

+ 与指定步态的匹配度

```

## 3. 点积项代表什么？

令：

```text

h = F(τ) # 运动特征

e = E(cS) # 指定步态特征

```

点积：

\[

h^Te=\sum\_{i=1}^{512}h\_ie\_i

\]

可以理解为比较两个512维特征的方向是否一致：

- 较大正数：运动特征与指定 skill 比较匹配。

- 较小或负数：运动特征与指定 skill 不匹配。

例如同一段 trot 动作：

```text

F(trot动作) · E(trot) → 可能较高

F(trot动作) · E(pace) → 可能较低

F(trot动作) · E(canter) → 可能较低

```

但你指出的问题是正确的：

> 只看匹配度，完全可能出现“动作符合某些步态特征，但动作本身不自然”。

所以还需要 U：

```text

最终 d =

skill 匹配分数

+ 通用自然性分数

```

不过要注意：这不是两个严格独立的规则，也不是逻辑上的：

```text

自然 AND skill匹配

```

两部分是共同训练、加法组合的。所谓“自然性”和“skill匹配度”是对其作用的直观解释，网络并没有被硬性规定只能分别学习这些信息。

## 4. d 是什么？

\[

d=D(\tau,c\_S)

\]

`d` 是判别器的原始输出，是任意实数：

```text

d ∈ (-∞, +∞)

```

它不是概率，因为最后没有 sigmoid。

该项目使用 LSGAN，将训练目标设置为：

```text

专家动作：d → +1

策略动作：d → -1

```

判别器损失为：

\[

L\_D=

\frac{1}{2}

\left[

(D(\tau\_\text{expert},c\_S)-1)^2+

(D(\tau\_\text{policy},c\_S)+1)^2

\right]

+L\_\text{gradient penalty}

\]

代码在 [secamp\_ppo.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/algorithms/secamp\_ppo.py:234>)。

所以判别器的训练逻辑是：

```text

专家数据 + 对应skill

↓

希望输出 d = +1

机器人策略数据 + 当前skill

↓

希望输出 d = -1

```

## 5. d 怎样变成 imitation reward？

判别器分数不能直接作为 PPO reward，所以需要转换：

\[

r\_I=

\max\left(0,\;1-\frac{1}{4}(d-1)^2\right)

\]

实际代码还乘了奖励系数和控制周期：

\[

r\_\text{imitation}

=

\text{amp\\_coef}

\cdot

\max\left(0,1-\frac14(d-1)^2\right)

\cdot dt

\]

代码在 [secamp\_disc\_project.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/secamp\_disc\_project.py:83>)。

不考虑 `coef` 和 `dt` 时：

| 判别器输出 d | 模仿奖励 |

|---:|---:|

| -1 | 0 |

| 0 | 0.75 |

| 1 | 1，最大值 |

| 2 | 0.75 |

| 3 | 0 |

| 小于 -1 或大于 3 | 截断到 0 |

这条函数的中心是 `d=1`，因为判别器对专家数据的目标就是 `+1`。

所以逻辑是：

```text

策略动作很不像专家

↓

D 输出接近 -1

↓

r\_imitation 接近 0

策略动作越来越像专家

↓

D 输出从 -1 向 +1 移动

↓

r\_imitation 增大

策略动作被 D 判断得像专家

↓

D 输出接近 +1

↓

r\_imitation 最大

```

## 6. Actor 是否直接反向传播穿过判别器？

这里容易误解：没有直接反向传播。

判别器计算出的 reward 会先交给 PPO：

```text

Actor 产生动作

↓

仿真得到下一状态

↓

判别器计算 d

↓

d 转换为 r\_imitation

↓

和 r\_task 合并

↓

PPO 计算 advantage

↓

PPO 更新 Actor

```

Actor 不会直接获得：

```text

∂D / ∂Actor

```

而是通过强化学习间接学会：

> 哪些动作最终能够获得更大的判别器奖励？

这与普通 GAN 有一点不同。普通 GAN 的生成器通常直接通过判别器反向传播；AMP 中的“生成器”是物理仿真里的控制策略，中间包含不可直接反传的环境动力学，因此使用 PPO 奖励进行学习。

## 7. 最终给 PPO 的奖励

当前代码将模仿奖励和任务奖励组合：

\[

r\_\text{total}

=

(1-\lambda)r\_I+\lambda r\_T

\]

当前 SECAMP 配置中：

```text

λ = 0.3

```

因此：

```text

r\_total = 0.7 × r\_imitation

+ 0.3 × r\_task

```

两部分的含义是：

```text

r\_imitation：

“你的动作像不像指定步态的专家动作？”

r\_task：

“你有没有朝 waypoint 前进？”

r\_total：

“既要以正确步态自然运动，也要完成导航任务。”

```

## 8. 完整逻辑图

```text

连续两帧运动 τ：86维

│

▼

F：86→1024→512

│

├──────────────► U：512→1

│ │

│ 通用自然性分数

│

▼

motion feature：512维

│

│ 点积

▼

skill feature：512维

▲

│

E：3→512

▲

│

skill one-hot：3维

d = 点积分数 + 通用自然性分数

│

▼

r\_imitation = max(0, 1 - 0.25(d-1)²)

│

▼

r\_total = 0.7 r\_imitation + 0.3 r\_task

│

▼

PPO 更新 Actor

```

最准确的一句话总结是：

> `F` 不是完整判别器，而是判别器的运动特征提取主干；`E` 是技能嵌入分支；`U` 是无条件线性判别头；三者合起来才是完整的条件判别器 `D`。`D` 输出原始分数 `d`，再由一个固定数学函数把 `d` 转换成 PPO 使用的模仿奖励。

## 7.

**一轮训练中的完整数据流**

入口是 [train.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/scripts/train.py:38>)：

1. 根据 Gym ID 创建 Isaac Lab 环境。

2. `LegacyAmpVecEnvWrapper` 把新环境适配到原来的 RSL-RL runner。

RSL-RL：一个开源的强化学习算法库（ETH苏黎世机器人实验室出品），里面实现了PPO等算法的训练循环（runner）。runner：就是"训练主循环"这个程序，负责反复执行"跟环境交互→收集数据→更新网络"这一整套流程。

VecEnv（Vectorized Environment，向量化环境）：不是只跑一个机器人仿真，而是同时并行跑成百上千个机器人仿真实例（比如你文档里提到"用500个环境训练"），这样能大幅加快数据收集速度。

Wrapper（适配器/包装器）：一个很常见的编程模式——当你有两套"接口约定"不一样的代码，想让它们能一起工作，就写一个"中间人"，把一边的输出格式转换成另一边能接受的格式,反之亦然。

总结LegacyAmpVecEnvWrapper 就是一个"翻译官"/"转接头"：它把Isaac Lab这个新环境包一层，对外表现得"看起来就像RSL-RL习惯的老式环境格式",这样老的PPO训练代码（RSL-RL runner）不需要改一行代码,就能直接拿新环境来跑。名字里的"Legacy"（遗留、旧式）就是在说"专门为了兼容老代码而写的适配层"

3. 环境生成 waypoint 和 one-hot skill。

waypoint确实是给导航用的，在强化学习框架里，"环境（environment）"是一个更宽泛的概念：它不只包含物理仿真（地面、重力、机器人身体），还包含整个任务的定义——包括"这一局任务的目标是什么"。而waypoint（目标点在哪）正是"这一局任务"的一部分设定。

具体来说，每次仿真重置（reset）、或者机器人到达上一个目标点之后，"环境"这段代码里会按预设规则生成一个新的目标点位置，作为这一段任务的waypoint。这跟"物理仿真怎么算重力、碰撞"是同一份代码里管的,只是环境代码除了管物理，还管"任务怎么出题"。

4. Actor 读取 45 维 observation，采样 12 维 action。

5. Isaac Lab 执行四个物理仿真步。

Actor（以及后续的Critic、Discriminator）只算一次，这一次的输出（动作）被固定不变地拿去连续执行4个物理仿真小步，之后才重新算一次。

这4小步里，机器人是靠PD控制器持续执行同一个目标角度，Actor全程不参与、不重新计算

6. 环境计算 task reward、终止状态和下一拍 observation。

7. 判别器根据根据"这一拍"和"上一拍"两帧 AMP 数据计算 imitation reward。

同时Critic对这一拍的状态打一次分

1. PPO rollout 存入普通训练缓存。

rollout 指的是：让当前的策略（Actor）在环境里实际跑一段时间，把这一路上发生的所有数据（状态、动作、奖励等）都记录下来，被存进一个缓存区（buffer）里，攒够一定数量之后，才拿去做PPO更新。

┌── rollout buffer ──→ PPO 更新 ──→ Actor + Critic

Rollout 阶段 ─────┤ （on-policy，用一次即丢）

└── AMP replay buffer ─┐

├──→ LSGAN 更新 ──→ Discriminator

专家数据集 ────────┘

PPO这类算法的训练是分成两大阶段反复循环的：

阶段一：rollout（数据收集阶段）——先不更新网络参数，就让当前版本的Actor拿着当前的脑子去跟环境互动很多步（比如几十步、几百步），每一步的observation、action、reward都记下来，攒成一批数据。

阶段二：update（更新阶段）——拿着刚攒下来的这批数据，去更新Actor、Critic（还有Discriminator）的网络参数,让它们变得更聪明一点。

主干（Actor/Critic 的 PPO）是严格 on-policy；带 replay buffer 的只有判别器，而判别器是分类器不是策略，不在 on/off-policy 这个话语体系里。用旧数据训判别器反而是好事——能防止它只针对最新一版策略的破绽过拟合，训练更稳。

为什么要叫"rollout"（把它"滚"出去）？ 想象你把一个策略像"滚雪球"一样，让它顺着时间一步步往前滚动执行下去（roll out），一步步展开出一整条轨迹（trajectory），这条轨迹上的每一步数据都被记录下来——这就是rollout这个词的直观来源，中文有时也直接翻成"轨迹采样"或"数据收集"。

1. 两帧动作状态和 skill 存入 AMP replay buffer。

存进AMP replay buffer的两帧，都是Actor（策略）自己在仿真里跑出来的动作,未掺杂专家数据

在每一个控制步（比如你之前问的那个 4-substep 之后重新决策的那一步）：

t 时刻: Actor 输出 action → 物理引擎跑 4 个 substep → 得到状态 s\_t

t+1 时刻: Actor 再次输出新 action → 物理引擎再跑 4 个 substep → 得到状态 s\_{t+1}

把 (s\_t, s\_{t+1}) 这一对儿存进 policy replay buffer

为什么必须是"相邻两帧"而不是单帧

因为单独一帧只有姿态信息（关节角度、位置等），看不出"运动的方式"，比如速度、加速度、动作的流畅度这些信息，必须靠两帧之差才能体现出来。所以 AMP 的判别器输入设计成 (st,st+1) 这种 pair，Discriminator 学的是"这一小段状态转移看起来像不像参考动作库里真实的运动片段"。

10. 更新阶段同时进行：

- Actor/Critic：PPO 更新。

- Discriminator：LSGAN 更新。

LSGAN（Least Squares GAN，最小二乘GAN）是训练判别器时用的一种具体损失函数设计，跟最原始版本的GAN（用交叉熵损失）相比，改成用均方误差（最小二乘）\*\*来衡量判别器输出得对不对：

判别器看到真实（专家）数据时，希望它输出的分数尽量接近某个"代表真"的目标值（比如1）

判别器看到Actor生成的数据时，希望它输出的分数尽量接近某个"代表假"的目标值（比如-1或0）

训练时用\*\*(判别器输出 - 目标值)² \*\*这种平方误差形式去算损失，然后反向传播更新判别器的参数,让它下次判断得更准。

为什么用LSGAN，而不是原始GAN那种交叉熵版本？

这是一个已经被机器学习社区验证过的经验性改进：原始GAN用交叉熵损失时，训练过程容易不稳定（判别器一下子太强、一下子梯度消失，生成器学不动），而LSGAN这种最小二乘的形式，梯度更平滑，训练过程通常更稳定,收敛效果更好,尤其在AMP这类"动作风格模仿"的场景里被广泛采用。

11. 周期性保存 `model\_<iteration>.pt`。

专家数据由 [secamp\_motion\_loader.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/datasets/secamp\_motion\_loader.py:214>)读取。它按策略当前 skill 选择同类专家动作，并以约 10% 概率混入左右转弯动作。

## 8.

**导出和部署数据流**

策略policy**导出链路**

由 [play.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/scripts/play.py:198>)完成：

Checkpoint

//

训练时每隔一段时间会保存一次 model\_<iteration>.pt，这就是 checkpoint（检查点/存档）。它里面装的是整个训练系统的全部家当：

Actor 的网络参数

Critic 的网络参数

Discriminator 的网络参数

优化器状态（Adam 的动量等，用于断点续训）

当前迭代次数等元信息

↓

只保留 Actor 的确定性均值输出

//

"只保留 Actor"——Critic 和 Discriminator 是训练用的脚手架，楼盖好就拆掉。真机上没有它们的用武之地（Critic 需要真机测不准的速度真值，Discriminator 需要专家数据来对比），所以直接扔掉，只把 Actor 挑出来。

"确定性均值输出"——训练时 Actor 输出的不是一个确定的动作，而是一个概率分布：网络输出一个均值 μ（12维）和一个标准差 σ，实际动作是从这个分布里随机采样出来的。

为什么训练时要随机采样？因为强化学习需要探索（exploration）——如果每次都输出完全一样的动作，网络永远发现不了"其实换个方式走可能更好"。随机性让它有机会试错、发现更优的动作。

但部署时不需要探索了，我们要的是最稳定、最可靠的表现。所以导出时只保留均值 μ，扔掉随机采样这一步——同样的输入，永远输出同样的动作。这就是"确定性（deterministic）"的意思。

↓

TorchScript：go2\_secamp.pt

JSON 契约：go2\_secamp.json

//

产出两个文件：

go2\_secamp.pt（TorchScript 格式）

TorchScript 是 PyTorch 的一种"脱离原始代码也能运行"的模型格式。普通的 PyTorch 模型必须有定义网络结构的 Python 代码才能加载；TorchScript 把**网络结构和参数一起打包**成一个自包含的文件，**部署端**不需要装 Isaac Lab、不需要 RSL-RL 训练框架、不需要你写的那些网络定义文件，**只要有 PyTorch 就能加载运行**。这对部署很关键——机载计算机上不可能装一整套训练环境。

go2\_secamp.json（契约文件）

.pt 文件只知道"45 个数字进来、12 个数字出去"，但它不知道这 45 个数字分别应该是什么、按什么顺序排。这些信息记在 JSON 里：

关节顺序（Isaac 的 FL/FR/RL/RR）

观测维度 45、动作维度 12

控制周期（50 Hz / 20 ms）

默认关节角、action scale（0.25）等

为什么叫"契约"？ 因为它是训练端和部署端之间的书面约定。训练端说"我是按这个格式训练的"，部署端照着这份文件去拼观测、去解释输出。只要双方都严格遵守这份契约，策略在真机上的行为就和仿真里一致。你文档第 9 节列的那些隐患（默认关节角对不上、Kp/Kd 不一致），本质上都是契约没被严格遵守导致的。

**部署链路（50Hz）**

MuJoCo 或 Go2 发布 LowState

//

LowState 是宇树定义的一种消息格式（"低层状态"），机器人（或 MuJoCo 仿真）会持续不断地把自己的传感器读数打包成这种消息广播出来，内容包括：

IMU：机身的姿态角、角速度、加速度

q：12 个关节当前的角度（来自关节编码器）

dq：12 个关节当前的角速度

关键点：这是唯一的信息来源。仿真里能直接读到的那些"上帝视角"数据（比如机身在世界坐标的精确位置和速度）在这里统统没有——这正是前面 Actor 刻意不用机身线速度的原因。

│

▼

deploy\_isaaclab.py（Python 推理节点）

- 订阅上一步的 LowState 消息，读取 IMU、q、dq

- Unitree 顺序 → Isaac 顺序（关节重排）：为的是把真机数据放回推理环境

- 构造 45 维 observation（把手上的数据按训练时完全相同的格式拼装）

- Actor 推理：加载 go2\_secamp.pt，把这 45 维喂进去，得到 12 维 action

- action → q\_target

- Isaac 顺序 → Unitree 顺序：推理完得到目标关节在输出到真机

│

▼

/rl/target\_pos

//

推理节点不直接控制电机，而是把算好的 12 个目标角度发布成一条 ROS2 消息，话题名是 /rl/target\_pos。

为什么要多这一道手续？ ROS2 在这里的作用就是各程序之间的"邮局"：

解耦：Python 推理节点和 C++ 控制节点是两个独立进程，可以分别开发、分别重启、分别调试

可观测：你可以在命令行直接监听这个话题，看策略到底输出了什么，不用改任何代码

可替换：想换个策略？只要新程序也往这个话题发消息，下游完全不用改

安全隔离：Python 那边即使卡住或崩溃，C++ 状态机还活着，可以进入安全模式

│

▼

low\_level\_ctrl.cpp

- 先看当前处于哪个状态：

躺下:这个状态下，状态机完全忽略 /rl/target\_pos 里的内容。哪怕 Python 那边已经在疯狂发送策略输出，状态机也不理会。

/站立q\_default

/RL 状态机：才转发 /rl/target\_pos 里的目标角度

- 设置 Kp/Kd：训练 Kp=50/Kd=1，部署 Kp=30/Kd=0.75是个隐患，和训练时不一致

- 状态机把最终的电机指令发布成 /mujoco/lowcmd（给 MuJoCo）或 /lowcmd（给真机）

│

▼

MuJoCo 或真实 Go2 电机

收到 lowcmd 后，电机驱动器内部按 torque = Kp × (q\_target − q) − Kd × dq 持续计算力矩并执行。

部署观测构造在 [deploy\_isaaclab.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_Deploy/src/deploy\_rl\_policy/scripts/deploy\_isaaclab.py:105>)，状态机在 [low\_level\_ctrl.cpp](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_Deploy/src/deploy\_rl\_policy/src/low\_level\_ctrl.cpp:124>)。

关节顺序必须牢记：

Isaac： FL, FR, RL, RR

Unitree：FR, FL, RR, RL

映射： [3,4,5, 0,1,2, 9,10,11, 6,7,8]

这是整个 sim-to-real 中最容易让机器人瞬间失控的地方。

**状态机（state machine）**是一种非常朴素的编程模式，规定三件事：

有哪几种状态（同一时刻只能处于其中一种）

每种状态下该做什么

什么条件下才允许从一种状态切换到另一种

## 9. 上真机前代码要检查的地方：

- `go2\_secamp/README.md` 写 `H=5`，当前配置实际为 `H=2`；到时候试试H=5效果

- 论文有 waypoint look-ahead；旧 Isaac Gym 实现有，当前 Isaac Lab 迁移版只在真正到达第一 waypoint 后切换，没有提前 look-ahead。

- 新的 `deploy\_isaaclab.py` 目前只使用 `default\_skill`，没有像旧 [deploy\_secamp.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_Deploy/src/deploy\_rl\_policy/scripts/deploy\_secamp.py:125>) 那样通过手柄切换 pace/trot/canter。

- YAML 中仍使用旧目录名 `Lab\_my\_AMP\_for\_hardware\_V1`，但你实际目录叫 `go2\_RL\_AMP`，对应路径目前不存在。

- 当前目录没有 `logs/` 和 `isaaclab/exports/`，所以还没有可直接部署的新策略。

- Isaac Lab 训练增益为 `Kp=50, Kd=1`，部署状态机运行策略时为 `Kp=30, Kd=0.75`，不在训练时 ±10% 随机化范围内。

- Isaac Lab 使用官方 `UNITREE\_GO2\_CFG` 的默认关节角，但部署 YAML 硬编码了旧 Isaac Gym 默认角；训练前应打印并严格核对。

- 手柄超时保护在 `xbox\_command.py` 中被注释，低层状态机也缺少策略消息 watchdog。

## 10. 最适合你的上手顺序

你现在已经有了项目的总地图。下一阶段最合适的是从 `go2\_env.py` 开始，我可以继续按“一次 50 Hz 控制周期”逐行带你读，让你最终能独立修改 observation、reward、skill 和 waypoint。

2. 阅读 `go2\_env\_cfg.py`，认识所有维度、频率和比例。

3. 阅读 `go2\_env.py` 的 `\_base\_policy\_obs()`、`\_get\_rewards()`、`\_pre\_physics\_step()`。

4. 阅读 `go2\_secamp\_runner.py` 的 `learn()`，理解 rollout。

5. 阅读 `secamp\_ppo.py` 的 `update()`，区分 PPO 和判别器更新。

6. 用 16 个环境运行 zero-agent。

7. 用很小的数据缓存和 64 个环境训练 2 次迭代。

8. 回放 checkpoint。

9. 导出 `.pt + .json`。

10. 修正部署路径后，只在 MuJoCo 测试。

11. 加完 watchdog、增益一致性和手柄技能切换后，再讨论真机。

**第19周：2026年8月17日-2026年8月23日**

问题（其他-2上）

**SECAMP和CAMP的不同**

根据论文原文，SECAMP 和 CAMP（Huang et al. 论文里引用的相关工作,论文标注为 [HXL25]）在\*\*同一大方向\*\*上（都是给 AMP 加"技能条件"，实现一个策略切换多个技能），但有几个关键的具体设计差异：

## 1. Conditioning 的具体实现方式不同（这是最核心的区别）

\*\*CAMP\*\*：论文原文提到CAMP 通过引入显式的技能条件变量（表示为潜在嵌入）来扩展 AMP 框架，这些条件变量同时提供给策略和判别器——但论文没有细说 CAMP 具体怎么把这个 embedding 喂进判别器（大概率是比较常规的做法，比如直接拼接/concatenation）。

\*\*SECAMP\*\*：明确采用了\*\*projection-based（投影式）条件判别器\*\*，而不是简单拼接：

- 判别器拆成三个模块：共享特征提取器 F、条件嵌入模块 E（把 skill code 投影到同一特征空间）、无条件真实性头 U

- 最终判别公式是 `d = F(τ)ᵀE(cS) + U(F(τ))`，用\*\*内积（inner product）\*\*来衡量"当前动作特征"和"目标技能嵌入"的匹配程度

- 这个设计借鉴自 \*\*Projection cGANs\*\*（Miyato & Koyama, 2018,是图像生成领域的经典条件 GAN 判别器设计）

- 论文特别强调了这么做的原因：直接把 one-hot skill code 拼接（concatenation）进网络，条件信号容易在网络浅层被稀释、被忽略；用投影+内积的方式可以避免这个问题，让条件信息在特征层面更直接地起作用（相当于一个 soft attention gate）

## 2. Skill 表示方式：用 one-hot，而不是连续 latent embedding

论文里明确指出他们\*\*没有用连续的 latent embedding\*\*来表示技能，而是用简单的离散 one-hot 向量（因为只需要区分 trot、pace、canter 三种固定步态，不需要技能空间里连续插值），CAMP 的描述里用的是"latent embeddings"（更偏向连续表示）。这是一个设计选择上的简化,论文里也在 Limitations 部分提到这是未来可以改进的方向（换成连续 embedding 可以实现步态间的平滑插值）。

## one-hot 是不是意味着技能在隐空间里相互独立、没法连续切换

\*\*你的直觉基本是对的，而且论文自己也承认了这个局限\*\*，但这里有个很容易混淆的地方需要分清楚，因为论文第 7.2 节偏偏又报告了"smooth and stable interpolation"的结果。

\*\*要区分两种完全不同的"平滑"：\*\*

\*\*(A) 隐空间插值（latent-space interpolation）——one-hot 做不到\*\*

连续 latent embedding 的真正价值在于：你可以喂一个\*\*训练时从没见过的中间点\*\*，比如 0.5·trot + 0.5·pace，得到一个介于两者之间的\*\*全新混合步态\*\*。one-hot 方案下，`cS ∈ {0,1}³` 只有三个合法取值，你喂 `[0.5, 0.5, 0]` 属于分布外输入（OOD），输出是什么完全没有保证——可能退化成某一种步态，也可能产生乱七八糟的动作。

严格说，`E(cS)` 确实把 one-hot 映射到了连续特征空间，得到 `e\_trot`、`e\_pace`、`e\_canter` 三个向量。理论上你可以去插值这三个向量……但训练过程中\*\*只有这三个点被约束过\*\*，它们之间的空间是完全自由、没有任何监督信号的，所以插值出来的东西没有任何保证。这正是论文在 Limitations 里写的：换成连续 embedding 才能实现步态间的平滑插值。

\*\*(B) 时间域上的过渡（temporal transition）——这个 SECAMP 是做到了的\*\*

论文 7.2 节实测的是：在 t=0 时刻把指令从 trot \*\*离散地\*\*切到 pace，然后观察机器人\*\*足端速度和加速度随时间的演化\*\*。结果是没有出现冲击性尖峰、没有不连续，机器人平滑地过渡过去了。

但这个平滑\*\*不是来自技能编码空间的连续性\*\*，而是来自两个别的因素：

- \*\*物理状态的连续性\*\*：机器人的关节角度、速度、足端位置在切换瞬间不可能瞬变，物理系统本身就带惯性和连续性

- \*\*策略学到了如何优雅地切换\*\*：策略网络是个动力学系统，它在训练中见过大量状态，学会了从任意状态平滑过渡到目标步态的模式

论文里那句 \*maintains a continuous embedding in the motion manifold across skill boundaries\* 说的是\*\*运动流形（motion manifold，即状态轨迹空间）\*\*的连续性，不是技能编码空间的连续性。措辞上确实容易让人误读成前者。

## 三、这个取舍值不值

论文给的理由是：目标行为集只有 trot、pace、canter 三种常见周期步态，不涉及跳跃、爬行、跌倒恢复这类高动态动作；既然目标不是在复杂技能之间插值、也不追求异质行为上的零样本泛化，那\*\*连续 latent embedding 就没有必要\*\*。

\*\*one-hot 的实际收益：\*\*

- \*\*可解释性和可控性拉满\*\*：你要 trot 它就一定给 trot，不会出现"我不知道现在这个 latent code 对应什么步态"的情况

- \*\*训练信号明确\*\*，不需要额外加互信息最大化（如 DIAYN、ASE 那套）之类的正则项来防止技能坍缩（mode collapse），大大简化了训练

- \*\*技能不会互相污染\*\*：正交编码天然保证了三种步态在条件层面不混淆

\*\*代价：\*\*

- 无法合成训练集之外的新步态

- 技能数量写死了，想加第四种步态得改输入维度、重新训练

- 编码之间的"距离"没有语义（trot 和 pace 的相似度关系没有被编码进去）

顺带一提，你注意到的这点其实是很好的批判性阅读——如果你之后要在这个工作基础上继续做，\*\*"把 one-hot 换成连续 latent embedding，实现步态之间真正的插值和新步态合成"就是论文自己点明的最直接的下一步\*\*（论文 Future Work 里写了）。你可以对照 ASE（Adversarial Skill Embeddings，论文引用列表里有）那类工作，看看连续技能空间是怎么用互信息约束来保证"中间点也有意义"的。

## 3. 任务目标不同：多了"目标导向的位置追踪"

CAMP（据论文描述）看起来聚焦点是\*\*多技能的动作模仿本身\*\*（让一个策略能表现出多种技能），而 SECAMP 除了多技能模仿，还额外加了一层\*\*waypoint-based 的目标导航能力\*\*——也就是说，SECAMP 不只是"能做出三种步态"，还要"在做出指定步态的同时，朝着指定的目标点导航"，这是论文贡献里单独列出的第三条：他们引入了一个基于\*\*限幅（clamped）、相对位置的路径点\*\*的解耦指令接口，来缓解目标导向运动里常见的"分布外指令"（out-of-distribution command）问题。这部分在 CAMP 原始工作里似乎并不是重点。

## 总结一下关系

| | CAMP | SECAMP |

|---|---|---|

| 基本思路 | AMP + 显式技能条件（分类），多技能共享一个判别器 | 同样思路，是在 CAMP 基础上做的进一步扩展 |

| 判别器条件方式 | 未详细说明（大概率非投影式） | 明确采用 projection-based + 内积，参考 Projection cGAN |

| 技能表示 | latent embedding（偏连续） | one-hot（离散，够用即可，简化设计） |

| 应用目标 | 多技能模仿本身 | 多技能 + 目标点导航（waypoint tracking）的结合 |

| 落地平台 | 未知（论文没细说） | Unitree Go2 四足机器人，仿真中的 trot/pace/canter 三种步态 |

\*\*所以回到"算不算复现"的问题\*\*：更准确的定位是——\*\*SECAMP 是在 CAMP 的思路基础上，做了判别器架构上的具体改进（projection-based conditioning）+ 应用场景上的扩展（加入目标导航），属于典型的"方法改进 + 新应用场景"型研究工作\*\*，不是照搬 CAMP 论文原样复现。如果你想更精确判断"改进程度算大还是小"，可以看看学长有没有做消融实验（ablation study）来对比"用拼接 vs 用投影"这两种方式的效果差异,如果做了对比实验证明投影方式更好，那这个贡献就更扎实。

接着问题：

这两个问题都问到了设计的关键点上。

**Discriminator输出奖励r\_i设计**

## 一、为什么是 `d = F(τ)ᵀE(cS) + U(F(τ))`，U 起什么作用

这个公式不是拍脑袋拼出来的，它来自 Projection cGAN（Miyato & Koyama, 2018）的一个理论推导。核心思想是：\*\*最优判别器的输出（logit）本身就可以被自然地分解成两项\*\*。

对条件 GAN 来说，理想判别器输出的是真假分布的对数似然比：

$$d^\*(x,c) = \log\frac{p(x,c)}{q(x,c)} = \underbrace{\log\frac{p(c|x)}{q(c|x)}}\_{\text{条件相关项}} + \underbrace{\log\frac{p(x)}{q(x)}}\_{\text{条件无关项}}$$

- 第一项：\*\*"这个动作属于哪个技能"\*\*——如果把类别后验建模成 log-linear（softmax）形式，这一项就正好写成特征和类别嵌入的内积，也就是 `F(τ)ᵀE(cS)`

- 第二项：\*\*"这个动作本身像不像真的"\*\*——跟技能标签完全无关，只看动作本身的真实性，这就是 `U(F(τ))`

所以 \*\*U 是无条件的真实性判别头（unconditional realism head）\*\*，论文里也是这么定义的：\*an unconditional realism head U : F → ℝ, which evaluates whether the motion resembles valid expert behavior independent of skill identity\*。

\*\*U 具体解决什么问题（为什么不能只留内积项）：\*\*

1. \*\*提供"绝对真实性"的基线判断\*\*。如果只有内积项，判别器学到的只是"这段动作和这个技能标签匹配度多高"，是一个\*\*相对\*\*的兼容性分数。一段抖动的、物理上不合理的假动作，只要它的特征方向恰好和 trot 的嵌入向量对齐，也可能拿到高分。U 独立地问"这动作本身像不像真狗跑出来的"，把这条底线补上了。

2. \*\*跨技能共享知识，提升数据效率\*\*。三种步态的数据全都在训练同一个 U，那些"什么叫物理上合理的四足运动"的通用知识（关节角度范围、足端轨迹形状、重心变化规律）被所有技能共享，而不是每个技能各学一遍。

3. \*\*实现论文所说的 reward disentanglement（奖励解耦）\*\*。两项各司其职：内积项管"风格对不对"，U 管"质量好不好"。训练信号分工明确，梯度不会互相打架，这也是论文强调这个设计能"减少异质运动基元之间的干扰"的原因。

4. \*\*对比拼接式（concatenation）的优势\*\*：如果把 one-hot 直接拼在输入上送进 MLP，条件信号只是几维输入里的一小部分，在前几层就容易被大量的运动特征淹没掉（论文原话是 diluted）。而投影式把条件放到\*\*特征空间\*\*里，用内积直接和高层特征作用，相当于一个 soft attention gate——技能标签在"动态地决定哪些特征维度应该被重视"，信号强得多。

- `go2\_secamp`：不是残差式 RL。Actor 直接输出 12 个关节动作。

-

**go2\_rough：残差式 RL**

冻结的 SECAMP 策略先产生基础动作，新的策略再输出小修正量。

## 1. SECAMP 是否属于残差控制？

SECAMP 的控制公式是：

```text

Actor 输出 a ∈ [-1,1]¹²

q\_target = q\_default + 0.25 × a

```

虽然这里相对于默认站姿 `q\_default` 做偏移，但通常叫“关节位置偏移控制”，不叫 residual RL。

真正的残差式 RL 在 `go2\_rough`：

```text

基础动作 = frozen\_SECAMP(obs, skill)

修正动作 = residual\_actor(obs\_history)

最终动作 = 基础动作 + 0.1 × 修正动作

```

对应代码在 [go2\_rough\_env.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/legged\_gym/envs/go2\_RoughTerrain/go2\_rough\_env.py:112>)。

```text

┌─ residual head ──► 12维修正动作 ───────┐

210维历史观测 ─► MLP ──┤ │

└─ skill head ─────► 3维skill logits │

│ │

softmax │

▼ │

42维当前观测 + skill ─► 冻结SECAMP先验 ─► 12维基础动作 │

│ │

└──── 相加 ◄─────┘

│

▼

12维最终关节动作

```

## 2. AAC 是什么？

如果你说的 AAC 是 Asymmetric Actor-Critic，那么它不是一种额外算法，而是 Actor-Critic 的非对称观测设计：

```text

Actor：只能看真机可获得的信息，45维

Critic：可以额外看仿真真值，51维

```

Actor 和 Critic 是两个独立 MLP：

- Actor 决定机器人做什么。

- Critic 只在训练时判断当前状态“未来大概能获得多少奖励”。

- 部署时只保留 Actor，Critic 被删除。

## 3. SECAMP 的神经网络架构

**AAC网络架构**

配置来自 [runner\_cfg.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/isaaclab/source/go2\_amp\_isaaclab/go2\_amp\_isaaclab/tasks/go2\_amp/runner\_cfg.py:52>)。

### Actor

```text

输入 45

↓

Linear 45 → 1024 + ELU

↓

Linear 1024 → 512 + ELU

↓

Linear 512 → 256 + ELU

↓

Linear 256 → 12

↓

12维关节动作

```

层数：

- 3 个隐藏层

- 1 个输出层

- 共 4 个 Linear 层

训练时，最后的 12 个数是高斯分布的均值：

```text

action ~ Normal(actor\_mean, learned\_std)

```

部署时不再随机采样，直接使用 `actor\_mean`。代码见 [actor\_critic.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/actor\_critic.py:58>)。

### Critic

```text

输入 51

↓

Linear 51 → 1024 + ELU

↓

Linear 1024 → 512 + ELU

↓

Linear 512 → 256 + ELU

↓

Linear 256 → 1

↓

状态价值 V(s)

```

也是：

- 3 个隐藏层

- 1 个输出层

- 共 4 个 Linear 层

Actor 和 Critic 不共享参数。

## 4. AMP 是不是神经网络？

AMP 本身不是一个网络，而是一套训练方法：

```text

Actor-Critic + 判别器 + 动作数据集 + PPO

```

其中真正属于 AMP 的主要网络是 Discriminator。

### 标准

**AMP 判别器网络架构**

标准 AMP 每帧模仿观测是 42 维，输入连续两帧：

```text

42 × 2 = 84维

```

网络：

```text

84

↓

Linear 84 → 1024 + ReLU

↓

Linear 1024 → 512 + ReLU

↓

Linear 512 → 1

↓

真实/非真实分数

```

层数：

- 2 个隐藏层

- 1 个输出层

- 共 3 个 Linear 层

实现见 [amp\_discriminator.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/algorithms/amp\_discriminator.py:18>)。

## 5. SECAMP 判别器

SECAMP 每帧使用 43 维模仿观测：

```text

关节位置 12

脚部局部位置 12

机身线速度 3

机身角速度 3

关节速度 12

机身高度 1

总计 43

```

连续两帧：

```text

43 × 2 = 86维

```

### 运动特征主干

```text

86

↓

Linear 86 → 1024 + ReLU

↓

Linear 1024 → 512 + ReLU

↓

运动特征 F(τ)，512维

```

### 你说的“AMP 前面的小型神经网络”

如果你指论文图中的 skill embedding `E(cˢ)`，它就是一个非常小的网络：

```text

3维 one-hot skill

↓

Linear 3 → 512，无 bias

↓

512维 skill embedding

```

只有一个 Linear 层，没有隐藏层。

然后计算：

```text

conditional\_score = F(τ) · E(skill)

unconditional\_score = Linear(512 → 1)

最终分数 =

conditional\_score + unconditional\_score

```

完整结构：

```text

两帧运动 86维 ─► 86→1024→512 ─► motion feature ───┐

├─ 内积 + 512→1 ─► score

skill one-hot 3维 ─► Linear 3→512 ─► skill feature ┘

```

对应 [secamp\_disc\_project.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/secamp\_disc\_project.py:37>)。

因此 SECAMP 判别器包含：

- 运动主干：2 个隐藏 Linear 层

- skill embedding：1 个并行 Linear 层

- 普通输出 head：1 个 Linear 层

沿运动主路径看是 3 个 Linear 层；整个模块总共有 4 个 Linear 模块。

## 6. PPO 是什么架构？

PPO 没有自己的神经网络层。

它是一套更新 Actor 和 Critic 参数的方法：

```text

仿真采样

↓

得到 observation、action、reward

↓

Critic 计算 value

↓

计算 return 和 advantage

↓

PPO clipped loss 更新 Actor

↓

value loss 更新 Critic

```

这里 PPO 管理的是：

```text

Actor loss

+ Critic value loss

- entropy bonus

```

SECAMP 同时还执行另一套更新：

```text

PPO optimizer → 更新 Actor + Critic

Discriminator optimizer → 更新 AMP 判别器

```

也就是说，PPO不训练判别器。判别器由 LSGAN loss 单独训练。代码见 [secamp\_ppo.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/algorithms/secamp\_ppo.py:215>)。

## 7. 真正的残差式 rough 网络

### Residual Actor

输入是最近 5 帧，每帧 42 维：

```text

5 × 42 = 210维

```

共享主干：

```text

210

↓

Linear 210 → 512 + ELU

↓

Linear 512 → 256 + ELU

```

之后分为两个 head：

```text

256 → Linear → 12维 residual

256 → Linear → 3维 skill logits

```

层数：

- 共享主干：2 个隐藏层

- 每条输出路径：1 个输出层

- 从输入到任一输出共经过 3 个 Linear 层

- 整个 Actor 一共有4个 Linear 模块，因为有两个输出 head

代码见 [actor\_critic\_two\_head.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/modules/actor\_critic\_two\_head.py:60>)。

### Rough Critic

```text

48

↓

Linear 48 → 512 + ELU

↓

Linear 512 → 256 + ELU

↓

Linear 256 → 1

```

即：

- 2 个隐藏层

- 1 个输出层

- 共3个 Linear 层

### Frozen Motion Prior

rough 里的“小型/基础策略”其实不是一个新网络，而是之前训练好的 SECAMP Actor：

```text

45 → 1024 → 512 → 256 → 12

```

它在 rough 训练期间被冻结：

- 不计算梯度

- 不被 PPO 更新

- 专门产生基础步态

- rough Actor 只学习地形修正量和 skill 选择

## 8. 一张总表

| 模块 | 输入输出 | 隐藏层 | Linear 层总数 | 激活 |

|---|---|---:|---:|---|

| 标准 AMP Actor | `42→12` | `512,256,128` | 4 | ELU |

| 标准 AMP Critic | `48→1` | `512,256,128` | 4 | ELU |

| 标准 AMP 判别器 | `84→1` | `1024,512` | 3 | ReLU |

| SECAMP Actor | `45→12` | `1024,512,256` | 4 | ELU |

| SECAMP Critic | `51→1` | `1024,512,256` | 4 | ELU |

| SECAMP 判别器主干 | `86→512` | `1024,512` | 2 | ReLU |

| Skill embedding | `3→512` | 无 | 1 | 无 |

| 判别器输出 head | `512→1` | 无 | 1 | 无 |

| Rough residual Actor | `210→12+3` | `512,256` | 4个模块 | ELU |

| Rough Critic | `48→1` | `512,256` | 3 | ELU |

| Rough frozen prior | `45→12` | `1024,512,256` | 4 | ELU |

最核心的关系是：

```text

AMP/SECAMP：训练“基础运动先验”

PPO：优化 Actor-Critic

AAC：Actor 和 Critic 使用不同观测

go2\_rough：在冻结的 SECAMP 运动先验上学习 residual

```

所以建议你的学习顺序是：先完全理解 SECAMP Actor-Critic 和判别器，再学习 rough residual；否则会把“关节偏移动作”和“策略残差”混为一谈。

**PPO的更新原理**

首先什么是return 和 advantage，怎么从奖励算到损失去更新

核心区别是：

> Reward 是环境给出的“成绩”；loss 是为了让神经网络参数可以通过梯度下降更新而构造的“训练误差”。

> Reward 本身通常不能直接反向传播，必须先变成 return 和 advantage，再构造 Actor/Critic 的 loss。

## 1. PPO 采样时真正的顺序

你的流程可以更准确地写成：

```text

当前状态 sₜ

│

├─ Actor 输出动作分布 π(a|sₜ)

│ └─ 采样动作 aₜ，并记录 log π\_old(aₜ|sₜ)

│

└─ Critic 预测 V(sₜ)

│

▼

环境执行 aₜ

│

├─ 得到即时奖励 rₜ

├─ 得到下一状态 sₜ₊₁

└─ 得到是否结束 done

连续采样24步

│

▼

根据 rewards、values、dones

计算 return 和 advantage

│

├─ advantage → Actor loss

└─ return → Critic loss

```

在这个项目中，每个环境一次收集24步，4096个环境并行，因此一轮大约收集：

```text

4096 × 24 = 98,304 条 transition

```

## 2. Reward 是什么？

Reward 是执行一拍动作后，环境立即给出的分数：

\[

r\_t=r\_\text{total}(s\_t,a\_t,s\_{t+1})

\]

SECAMP 中：

\[

r\_t=0.7r\_\text{imitation}+0.3r\_\text{task}

\]

例如：

```text

机器人朝 waypoint 前进 → task reward 增大

动作像指定的 trot → imitation reward 增大

摔倒、走反方向、动作不自然 → reward 较小

```

但当前这一拍奖励高，不代表这个动作长期是好的。

例如：

```text

动作A：这一拍冲得很快，reward=2，但下一拍摔倒

动作B：这一拍reward=1，但之后连续稳定行走100步

```

显然动作B长期更好。因此 PPO 不只看当前 `reward`，而要计算未来累计奖励，也就是 return。

## 3. Return 是什么？

Return 中文一般叫“回报”，表示从时间 `t` 开始，未来能拿到的折扣累计奖励：

\[

G\_t=r\_t+\gamma r\_{t+1}+\gamma^2r\_{t+2}+\cdots

\]

其中 `γ` 是 discount factor。当前 Isaac Lab 配置中：

```text

γ = 0.99

```

越远的奖励，权重稍微越低。

### 简单例子

假设只有3步：

```text

r₀ = 1

r₁ = 2

r₂ = 3

γ = 0.9

```

从最后向前计算：

\[

G\_2=3

\]

\[

G\_1=2+0.9\times3=4.7

\]

\[

G\_0=1+0.9\times4.7=5.23

\]

所以：

| 时刻 | 当前奖励 | 从这里开始的 return |

|---|---:|---:|

| `t=0` | 1 | 5.23 |

| `t=1` | 2 | 4.70 |

| `t=2` | 3 | 3.00 |

Reward 是“一拍成绩”，return 是“从这一拍开始的长期总成绩”。

## 4. Critic 的 Value 是什么？

Critic 输出：

\[

V(s\_t)

\]

它表示：

> 根据过去训练经验，从状态 \(s\_t\) 出发，按照当前 Actor 继续行动，预计未来总共能拿到多少 return？

注意它只是预测，不一定正确。

例如：

```text

Critic预测：V(sₜ) = 4.5

实际算出的return：Gₜ = 5.23

```

说明实际结果比 Critic 预期更好。

Critic 相当于一个“预期基准”：

```text

return：这一次实际上表现如何

value：正常情况下预计表现如何

```

## 5. Advantage 是什么？

Advantage 表示这次选择的动作，比当前平均预期好多少：

\[

A\_t=G\_t-V(s\_t)

\]

例如：

```text

实际return = 5.23

Critic预测value = 4.50

advantage = 5.23 - 4.50 = +0.73

```

正 advantage：

> 这个动作带来的结果比预期好，以后应该更常选择。

负 advantage：

```text

实际return = 3.0

Critic预测value = 4.5

advantage = -1.5

```

表示：

> 这个动作的结果比预期差，以后应该减少它的概率。

这就是 advantage 最重要的作用：

```text

A > 0：提高该动作的概率

A < 0：降低该动作的概率

A ≈ 0：这个动作和正常水平差不多

```

为什么不直接用 return？

因为不同状态的天然难度不同。例如：

```text

平稳站立状态，return=5 可能很差

即将摔倒状态，return=2 可能已经很好

```

用 `return - value` 后，比较的是“相对当前状态预期好不好”，Actor 的学习会稳定很多。

## 6. 实际代码使用 GAE

实际 PPO 不直接使用完整的 Monte Carlo return，而使用 GAE：

\[

\delta\_t

=

r\_t+\gamma V(s\_{t+1})-V(s\_t)

\]

这个 `δ` 叫 TD error。

它表示：

> 当前奖励加下一状态价值，与 Critic 原来对当前状态的预测相差多少？

然后从后向前累积：

\[

A\_t

=

\delta\_t+\gamma\lambda A\_{t+1}

\]

当前配置：

```text

γ = 0.99

λ = 0.95

```

最后构造 Critic 的训练目标：

\[

R\_t=A\_t+V(s\_t)

\]

代码完全对应：

```python

delta = reward + gamma \* next\_value - value

advantage = delta + gamma \* lam \* next\_advantage

returns = advantage + value

```

见 [rollout\_storage.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/storage/rollout\_storage.py:124>)。

当 episode 结束时，下一状态价值不再计算：

```text

done=True → next\_value 按 0 处理

```

## 7. Advantage 怎样更新 Actor？

Actor 并不直接输出一个确定动作，而是输出高斯动作分布：

```text

Actor(obs) → mean

可学习参数 → std

a ~ Normal(mean, std)

```

采样时保存旧策略对动作的概率：

\[

\pi\_\text{old}(a\_t|s\_t)

\]

更新时，用当前的新参数重新计算：

\[

\pi\_\text{new}(a\_t|s\_t)

\]

然后计算概率比：

\[

\rho\_t=

\frac{\pi\_\text{new}(a\_t|s\_t)}

{\pi\_\text{old}(a\_t|s\_t)}

\]

代码为了数值稳定，使用 log probability：

```python

ratio = exp(new\_log\_prob - old\_log\_prob)

```

### 当 advantage 为正

假设：

```text

A = +2

```

这个动作比预期好，因此希望：

```text

π\_new(a|s) > π\_old(a|s)

ratio > 1

```

也就是提高未来再次选择该动作的概率。

### 当 advantage 为负

```text

A = -2

```

这个动作比预期差，因此希望：

```text

π\_new(a|s) < π\_old(a|s)

ratio < 1

```

也就是降低未来选择它的概率。

因此最基础的 policy objective 是：

\[

\rho\_tA\_t

\]

因为优化器默认最小化 loss，所以写成负号：

\[

L\_\text{actor}=-\rho\_tA\_t

\]

## 8. 为什么需要 PPO clipping？

如果一次更新让动作概率变化太大，策略可能突然崩溃。

例如：

```text

旧概率：0.10

新概率：0.30

ratio：3.0

```

只用普通 policy gradient 可能认为：

> 这是好动作，继续疯狂提高概率。

PPO 将 ratio 限制在：

\[

[1-\epsilon,1+\epsilon]

\]

当前：

```text

ε = 0.2

```

即主要限制在：

```text

[0.8, 1.2]

```

PPO objective：

\[

L\_\text{clip}

=

-\min

\left(

\rho\_tA\_t,

\operatorname{clip}(\rho\_t,0.8,1.2)A\_t

\right)

\]

直观上：

```text

好动作：可以提高概率，但一次不要提高太多

坏动作：可以降低概率，但一次不要降低太多

```

例如 `A=+2`：

```text

ratio = 1.1 → 允许继续学习

ratio = 1.3 → 按1.2计算，不再鼓励继续大幅改变

```

项目代码因为已经加了负号，所以用 `max` 实现同一件事，见 [secamp\_ppo.py](<C:/Users/ASUS/Desktop/Semester Arbeit/Projects/go2\_RL\_AMP/rsl\_rl/rsl\_rl/algorithms/secamp\_ppo.py:215>)：

```python

surrogate = -advantage \* ratio

surrogate\_clipped = -advantage \* clip(ratio, 0.8, 1.2)

actor\_loss = max(surrogate, surrogate\_clipped).mean()

```

## 9. Return 怎样更新 Critic？

Critic 预测：

\[

V\_\psi(s\_t)

\]

GAE 得到它应该接近的目标：

\[

R\_t

\]

所以 Critic 使用均方误差：

\[

L\_\text{critic}

=

(V\_\psi(s\_t)-R\_t)^2

\]

例如：

```text

Critic预测 value = 4.5

训练目标 return = 5.23

value loss = (4.5 - 5.23)²

= 0.5329

```

梯度下降会调整 Critic 参数，让下次预测更接近5.23。

Critic 越准确：

```text

advantage = return - value

```

就越能准确判断某个动作究竟比正常水平好还是差。

## 10. Actor 和 Critic 的总 loss

项目中的 PPO loss 大致为：

\[

L\_\text{PPO}

=

L\_\text{actor}

+

c\_vL\_\text{critic}

-

c\_eH(\pi)

\]

其中：

- `actor loss`：让好动作概率增加、坏动作概率降低。

- `critic loss`：让 value 接近 return。

- `entropy H`：防止动作分布过早变得确定，保留探索。

- `c\_v`：value loss 权重。

- `c\_e`：entropy 权重。

代码为：

```python

policy\_loss\_total = (

surrogate\_loss

+ value\_loss\_coef \* value\_loss

- entropy\_coef \* entropy

)

```

然后：

```python

policy\_loss\_total.backward()

optimizer.step()

```

## 11. Reward 到参数更新的完整链路

**PPO更新AC完整路线**

```text

Actor参数 θ

│

▼

动作分布 πθ(a|s)

│

▼

采样动作 a

│

▼

物理仿真

│

▼

task reward + imitation reward

│

▼

每一步 reward rₜ

│

▼

GAE计算

├─ return Rₜ ─────► Critic正确答案

│ │

│ ▼

│ (V(sₜ)-Rₜ)²

│ │

│ ▼

│ 更新Critic

│

└─ advantage Aₜ ─► 判断动作好坏

│

▼

PPO clipped actor loss

│

▼

更新Actor

```

最简洁地记：

```text

Reward：环境说这一拍得多少分

Return：从这一拍开始，未来总共能得多少分

Value：Critic预测未来能得多少分

Advantage：实际结果比预期好多少

Actor loss：让高advantage动作更常出现

Critic loss：让value预测更接近return

```

因此奖励决定“希望机器人学会什么”，损失函数决定“怎样把这个目标转换成神经网络参数的梯度”。

**URDF / MJCF / USD文件配置**

这些都是\*\*配置文件 / 描述文件\*\*——不是程序代码，而是"数据"或"说明书"，程序读取它们来知道该怎么做。我按你项目里会遇到的分几类讲。

---

# 一、配置类文件（存参数）

## YAML（`.yaml` / `.yml`）

\*\*是什么\*\*：一种给人读的配置文件格式，靠\*\*缩进\*\*表示层级关系，语法极简。名字是 "YAML Ain't Markup Language" 的递归缩写。

长这样：

```yaml

policy:

path: "models/go2\_secamp.pt"

control\_freq: 50

action\_scale: 0.25

pd\_gains:

kp: 30.0

kd: 0.75

default\_joint\_angles:

FL\_hip: 0.1

FL\_thigh: 0.8

FL\_calf: -1.5

```

\*\*为什么要用它\*\*：把参数从代码里抽出来单独放。想改 Kp 从 30 变成 50，只需要改这个文本文件，\*\*不用重新编译、不用碰一行代码\*\*。ROS2 生态里几乎所有节点参数都用 YAML 配置。

\*\*你项目里的角色\*\*：`go2\_Deploy` 的部署参数就在 YAML 里——策略文件路径、控制频率、Kp/Kd、默认关节角。你文档第 9 节说的"YAML 里硬编码了旧 Isaac Gym 的默认关节角""YAML 仍指向旧目录名"，指的就是这类文件里的值没跟着更新。

\*\*要注意的坑\*\*：YAML 用缩进表示结构，\*\*多一个空格少一个空格意义就变了\*\*，而且\*\*不能用 Tab 只能用空格\*\*。这是新手最常踩的雷。

## JSON（`.json`）

跟 YAML 干同一件事，但语法更严格（大括号、引号、逗号一个都不能少），对机器友好、对人稍微不友好。

```json

{

"obs\_dim": 45,

"action\_dim": 12,

"joint\_order": ["FL\_hip", "FL\_thigh", "FL\_calf", "..."],

"control\_dt": 0.02

}

```

\*\*你项目里的角色\*\*：`go2\_secamp.json` 那份"契约"就是 JSON。选 JSON 而不是 YAML，是因为它主要给程序读，且 PyTorch/Python 生态里 JSON 是最通用的交换格式。

## TOML（`.toml`）

第三种配置格式，Python 项目的 `pyproject.toml` 常见。你可能会在安装依赖时见到，但一般不用自己改。

---

# 二、机器人模型描述文件（存"机器人长什么样"）

这一类回答的问题是：\*\*机器人有几个关节、每段连杆多长多重、关节能转多少度、外观是什么样\*\*。仿真器靠它们把机器人"造"出来。

## URDF（`.urdf`）

\*\*Unified Robot Description Format\*\*，统一机器人描述格式。ROS 生态的事实标准，用 XML 写。

```xml

<link name="FL\_thigh">

<inertial>

<mass value="1.152"/>

<origin xyz="-0.003 -0.022 -0.032"/>

</inertial>

<visual>

<geometry><mesh filename="FL\_thigh.dae"/></geometry>

</visual>

</link>

<joint name="FL\_thigh\_joint" type="revolute">

<parent link="FL\_hip"/>

<child link="FL\_thigh"/>

<axis xyz="0 1 0"/>

<limit lower="-1.05" upper="4.19" effort="23.7" velocity="30.1"/>

</joint>

```

核心是两个概念：

- \*\*link（连杆）\*\*：刚体部件，比如大腿、小腿。记录质量、惯量、碰撞形状、外观网格。

- \*\*joint（关节）\*\*：连接两个 link 的活动关节。记录转轴方向、角度上下限、最大力矩、最大转速。

\*\*URDF 的局限\*\*：只能描述\*\*树形结构\*\*（每个 link 只有一个父节点），没法直接表达闭环机构（比如平行四边形连杆）。另外它不能描述传感器、材质摩擦等仿真细节，所以各家仿真器都在它基础上做了扩展。

\*\*宇树 Go2 官方就提供 URDF\*\*，这是最原始的模型来源。

## MJCF / XML（`.xml`）

\*\*MuJoCo 自己的格式\*\*（MuJoCo XML Configuration Format）。功能比 URDF 强——能描述接触摩擦参数、执行器模型、传感器、约束、求解器设置等仿真专有的东西。

MuJoCo 可以导入 URDF，但通常大家会把 URDF 转成 MJCF 再手工调优摩擦、阻尼这些参数，因为这些直接影响仿真的真实度。

\*\*你项目里的角色\*\*：`go2\_Deploy` 里跑 MuJoCo 仿真时用的就是这类文件。

## USD（`.usd` / `.usda` / `.usdc`）

\*\*Universal Scene Description\*\*，通用场景描述。\*\*Pixar 开发\*\*的格式（原本用于电影动画制作），后来被 NVIDIA 选为 Omniverse / Isaac Sim 的核心格式。

- `.usda` 是\*\*文本\*\*格式，可以用文本编辑器打开看

- `.usdc` 是\*\*二进制\*\*格式，体积小、加载快

- `.usd` 是通用后缀，两种都可能

\*\*跟 URDF 的区别\*\*：URDF 只描述"一个机器人"；USD 描述的是\*\*整个场景\*\*——机器人、地形、光照、材质、物理属性、相机，全都能装进去，而且支持\*\*分层引用\*\*（一个 USD 文件可以引用另一个，改底层文件所有引用处都跟着变）。

\*\*你项目里的角色\*\*：Isaac Lab / Isaac Sim 训练时，Go2 机器人是以 USD 形式加载的。通常流程是：官方 URDF → 用 Isaac Sim 的转换工具导入 → 生成 USD → 训练时引用这个 USD。

## 其他你可能撞见的

| 后缀 | 是什么 |

|---|---|

| `.xacro` | 带宏和变量的 URDF，用来避免重复（四条腿只写一遍，用参数生成四份），最终会被展开成 URDF |

| `.sdf` | Simulation Description Format，Gazebo 仿真器的格式，比 URDF 强但生态不如 URDF 广 |

| `.dae` / `.stl` / `.obj` | \*\*纯几何网格文件\*\*，只有形状没有物理属性。被 URDF/USD 引用，用来渲染外观或计算碰撞。`.stl` 常用于碰撞体（简单），`.dae`/`.obj` 常用于视觉外观（精细，带材质） |

---

# 三、你项目里其他会遇到的后缀

| 后缀 | 内容 | 出现位置 |

|---|---|---|

| `.pt` / `.pth` | PyTorch 模型权重或 TorchScript 模型 | `go2\_secamp.pt`、`model\_<iter>.pt` |

| `.npy` / `.npz` | NumPy 数组的二进制存储，动捕数据常用这个存 | 专家动作数据集 |

| `.csv` | 逗号分隔表格，也常用于存动捕轨迹 | 动作数据 |

| `.launch.py` | ROS2 启动文件，规定一次要启动哪几个节点、各自加载哪个 YAML | `go2\_Deploy` |

| `.msg` | ROS2 消息类型定义，规定 `LowState`、`target\_pos` 这类消息里有哪些字段 | ROS2 接口 |

| `CMakeLists.txt` / `package.xml` | ROS2 功能包的构建配置和依赖声明 | 每个 ROS2 包必备 |

| `.pkl` | Python 的 pickle 序列化文件，存任意 Python 对象 | 有时用于存训练配置或数据 |

---

# 串起来看你的项目

```

宇树官方 URDF （机器人的原始"图纸"）

│

├─→ 转成 USD ──→ Isaac Lab 训练 ──→ 产出 model\_<iter>.pt（checkpoint）

│ │

│ └─→ play.py 导出

│ ├─ go2\_secamp.pt （TorchScript）

│ └─ go2\_secamp.json（契约）

│ │

└─→ 转成 MJCF/XML ──→ MuJoCo 仿真 ←──────────────────┤

│

go2\_Deploy 的 YAML 读取 ────┘

（路径、Kp/Kd、默认关节角、控制频率）

│

ROS2 节点运行

```

\*\*一句话概括\*\*：

- \*\*URDF / MJCF / USD\*\* = 机器人和场景"长什么样"，给仿真器造模型用

- \*\*YAML / JSON\*\* = 程序运行的"参数怎么设"，给代码读配置用

- \*\*.pt / .npy\*\* = 训练产出的"数据和权重"，是二进制不可读的

这三类的共同点是：\*\*都不是可执行代码\*\*，而是被代码读取的数据。改它们不需要重新编译，但改错了程序照样会出问题——你文档第 9 节里那些"路径失效""默认关节角对不上"的隐患，全都出在 YAML 这一层。