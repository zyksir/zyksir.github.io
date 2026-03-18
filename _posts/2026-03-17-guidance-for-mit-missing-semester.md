---
layout: post
title: "MIT Missing Semester 课程笔记"
subtitle: ""
date: 2026-03-17
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: ["class note"]
---

# MIT Missing Semester 课程导引与实战笔记

这篇文章是 [MIT Missing Semester](https://missing.csail.mit.edu/) 的课程导引与笔记。作为一个工作两年的程序员，我会在课程内容的基础上，融入自己在实际工作中高频使用的技巧和经验。

MIT Missing Semester 是一门专门教"程序员日常工具"的课程。传统的 CS 课程教你写算法、造系统，却很少教你如何高效地使用终端、配置开发环境、用 Git 协作——这门课就是来填补这个空缺的。课上会介绍大量命令行工具，其中不少在日常中并不常用（尤其是 AI agent 时代，很多琐碎操作可以直接交给 agent），但理解它们能让你对 shell 建立更深的认知。让我印象深刻的是，这门课的内容一直在与时俱进：三年前我看的时候还有不少篇幅在讲数据处理（Data Wrangling），如今这部分已经被 [Agentic Coding](https://missing.csail.mit.edu/2026/agentic-coding/) 等更前沿的话题取代了。

**2026 课程大纲**

以下是 2026 年最新的课程目录，每节课都有独立的讲义和录像。本文覆盖了前五讲的内容，后四讲的笔记后续补充。

| 日期 | 主题 | 链接 |
|------|------|------|
| 1/12 | Course Overview + Introduction to the Shell | [讲义](https://missing.csail.mit.edu/2026/course-shell/) |
| 1/13 | Command-line Environment | [讲义](https://missing.csail.mit.edu/2026/command-line-environment/) |
| 1/14 | Development Environment and Tools | [讲义](https://missing.csail.mit.edu/2026/development-environment/) |
| 1/15 | Debugging and Profiling | [讲义](https://missing.csail.mit.edu/2026/debugging-profiling/) |
| 1/16 | Version Control and Git | [讲义](https://missing.csail.mit.edu/2026/version-control/) |
| 1/20 | Packaging and Shipping Code | [讲义](https://missing.csail.mit.edu/2026/shipping-code/) |
| 1/21 | Agentic Coding | [讲义](https://missing.csail.mit.edu/2026/agentic-coding/) |
| 1/22 | Beyond the Code | [讲义](https://missing.csail.mit.edu/2026/beyond-code/) |
| 1/23 | Code Quality | [讲义](https://missing.csail.mit.edu/2026/code-quality/) |

> 课程视频可在 [YouTube @MissingSemester](https://www.youtube.com/@MissingSemester) 频道免费观看。中文翻译版讲义见 [missing-semester-cn.github.io](https://missing-semester-cn.github.io/)（基于 2020 版本，部分内容已过时，但基础概念依然适用）。

**阅读建议**：建议在观看每节课视频之前，先浏览本文对应章节的内容概要，建立大致印象；看完视频后，再回来查阅"实用知识"部分，重点关注我在工作中高频使用的内容。

## 第一讲：The Shell

**课程概要**：[第一讲](https://missing.csail.mit.edu/2026/course-shell/)介绍 Shell 的基本概念和常见命令，[第二讲](https://missing.csail.mit.edu/2026/command-line-environment/)则深入讲解命令行环境，包括参数、流（stream）、环境变量、信号（signal）以及 Shell 脚本编写。在我看来，Shell 代表着计算机最极致、最朴素的美学——现代软件的 UI 越做越精美，但追根溯源，一个程序只需要一个启动命令，再加上与用户的文字交互，就足够了。

**打造属于自己的 Shell 环境**

我选择了 [zsh](https://www.zsh.org/) 作为默认 shell，配合 [Oh My Zsh](https://ohmyz.sh/) 框架管理插件和主题。我的配置脚本在[这里](https://github.com/zyksir/dotfiles/blob/main/dotfiles/zshrc)。本地 Mac 上我用的是 [powerlevel10k](https://github.com/romkatv/powerlevel10k) 主题，虽然比较重，但信息丰富、颜值在线；在远程开发机上则用更轻量的 `philips` 主题，启动更快。

![img](../../../../img/mit-missing-semester/1.mac_shell_demo.png)
*本地 Mac：powerlevel10k 主题，显示 Git 分支、conda 环境等信息*

![img](../../../../img/mit-missing-semester/1.dev_shell_demo.png)
*远程开发机：philips 主题，简洁轻量*

**历史命令搜索：`Ctrl + R`**

这是我每天使用频率最高的快捷键之一。安装 [fzf](https://github.com/junegunn/fzf) 后，`Ctrl + R` 会升级为模糊搜索界面，体验质的飞跃：

![img](../../../../img/mit-missing-semester/1.fzf_command_search.png)
*fzf 加持下的 Ctrl + R：模糊匹配历史命令，告别逐条翻找*

小技巧：如果不小心在命令行输入了敏感信息（密码、token 等），可以在 `.zshrc` 中设置 `setopt HIST_IGNORE_SPACE`，之后在命令前加一个空格就不会被记录到历史中。（说实话我基本没用过，安全意识有待加强。）

**文件查找：`Ctrl + T`**

fzf 的 `Ctrl + T` 可以在当前目录下模糊搜索文件路径并粘贴到命令行中。此外，macOS 上的 `Command + Space`（Spotlight）也能根据文件名片段快速定位文件。

![img](../../../../img/mit-missing-semester/1.fzf_file_search.png)
*Ctrl + T：在当前目录下模糊搜索文件*

![img](../../../../img/mit-missing-semester/1.%20mac_spot_search.png)
*macOS Spotlight：根据关键词快速找到文件*

**目录跳转：`z` 命令**

fzf 的 `Alt + C` 可以模糊搜索子目录并直接跳转，但我更常用的是 `z` 命令（[zsh-z](https://github.com/agkozak/zsh-z) 插件）。它会学习你的访问习惯，只需输入目录名的一部分就能跳转到最常访问的匹配目录。比如 `z sgl` 就能直接跳到 `~/github/sglang`：

![img](../../../../img/mit-missing-semester/1.z_jump_cd.png)
*z sgl → 直接跳转到 sglang 目录，无需输入完整路径*

**常用命令速查**

- `cd`, `ls`, `echo`, `pwd`, `kill`, `pkill`——这些是最基本的命令。其中 `kill`、`pkill` 和信号（signal）的部分与操作系统课程有所重叠，上过 OS 课的同学应该不陌生，这些知识都很实用。
- `find . -name "*.py" -exec grep -l "TODO" {} \;`——`find ... -exec` 是我用得最多的组合，可以按文件名、修改时间、大小等条件批量对文件执行指定操作。
- `df` 查看文件系统整体使用情况；`du -sh *` 快速查看当前目录下各项的磁盘占用——多人共用开发机时，磁盘告急是常有的事。
- 这些复杂命令其实可以让 LLM 帮你写。不过在开发机上配置 API Key 不太方便（安全性存疑），我一般在本地 Claude App 里生成好命令再复制过去。课程也推荐安装 [tldr](https://tldr.sh/)，可以直接在终端里查看常用命令的简洁示例。

**Shell 脚本编写**

批量调参、起实验时，我通常会写一个 `run.sh` 脚本。以下是几个值得记住的要点（详见[第二讲讲义](https://missing.csail.mit.edu/2026/command-line-environment/)）：

- **特殊变量**：`$0`（脚本名）、`$1 ~ $9`（位置参数）、`$?`（上一条命令的返回值）、`$@`（所有参数列表）、`$#`（参数个数）
- **命令连接符**：`&&`（前一个成功才执行后一个）、`||`（前一个失败才执行后一个）、`;`（无论成败都执行下一个）
- **获取命令输出**：`$(CMD)` 将输出赋值给变量，如 `for file in $(ls)`；`<(CMD)` 将输出作为临时文件传递，适用于需要文件路径的场景，如 `diff <(ls foo) <(ls bar)` 比较两个目录的差异
- **条件判断**：尽量使用 `[[ ]]` 而非 `[ ]`，可以避免变量为空时报错等常见问题

## 第二讲（上）：tmux

**课程概要**：你可能见过"边骑自行车边盯着笔记本"的程序员——其实很可能只是因为笔记本一合上，SSH 连接断了程序就没了。[tmux](https://github.com/tmux/tmux) 就是为了解决这个问题而生的。相比简单地把程序放到后台运行（`nohup` 或 `&`），tmux 最大的优势在于能完整保留"工作现场"：环境变量、命令历史、输出结果全都还在，随时可以 detach/reattach 恢复。

tmux 的核心概念是 **session → window → pane**：一个 session 包含多个 window（标签页），每个 window 又可以分割成多个 pane（窗格）。可以把它理解为一个运行在终端里的小型窗口管理器。

![img](../../../../img/mit-missing-semester/2.tmux.png)
*我的 tmux 界面：左右分屏 + 底部状态栏显示 window 列表和时间*

tmux 的界面和快捷键都可以自定义。默认的前缀键是 `Ctrl + B`（例如 `Ctrl + B, D` detach 当前 session）。我的配置文件在这里：[tmux.conf](https://github.com/zyksir/dotfiles/blob/main/dotfiles/tmux.conf.ini)。更多进阶用法（如增大历史缓冲区、配置复制粘贴等）可以自行查阅或直接问 Claude。

当然，如果你在本地用 Mac 开发，[iTerm2](https://iterm2.com/) 本身的多标签和分屏功能也非常强大：

![img](../../../../img/mit-missing-semester/2.iterm_example.png)
*iTerm2：左边是本地的两个分屏，右边 SSH 到了远程开发机*

顺便提一个我非常喜欢的 iTerm2 功能：**密码管理器**（`Option + Command + F`）。经常要在远程机器上输入 `sudo` 密码，但开发机密码往往又长又难记。用这个功能可以提前保存好，需要时一键填入：

![img](../../../../img/mit-missing-semester/2.iterm_pass.png)
*iTerm2 密码管理器：保存常用密码，一键输入*

> **拓展阅读**：tmux 在 2026 课程的[第二讲（Command-line Environment）](https://missing.csail.mit.edu/2026/command-line-environment/)中仍有涉及，只是不再作为单独的章节。更系统的学习可以参考 [tmux 官方 Wiki](https://github.com/tmux/tmux/wiki)、[A beginner's guide to tmux](https://towardsdatascience.com/a-beginners-guide-to-tmux-a-multitasking-superpower-for-your-terminal/) 或 [Getting Started with tmux](https://tmux.info/docs/getting-started)。

## 第二讲（下）：SSH

SSH 同样在[第二讲](https://missing.csail.mit.edu/2026/command-line-environment/)中讲解。对于大部分开发者来说，SSH 的使用场景比较固定。我的日常流程：生成本地密钥对 → 把公钥交给运维 → 运维在开发机上配置权限 → 愉快登录。

核心命令：

```bash
# 生成密钥对（推荐 ed25519 算法）
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519

# 将公钥复制到远程机器
ssh-copy-id -i ~/.ssh/id_ed25519 <username>@<remote-ip>
```

建议配置 `~/.ssh/config` 来管理多台机器的连接信息，免去每次输入完整地址的麻烦。课程还推荐了 [Mosh](https://mosh.org/) 作为 SSH 的替代方案，它能更好地处理断网、休眠、切换网络等场景。

## 第三讲：Vim 编辑器

**课程概要**：2026 版本的[第三讲](https://missing.csail.mit.edu/2026/development-environment/)已经将 Vim 与 IDE、AI 辅助开发整合在一起，统称为"Development Environment and Tools"。课程指出 Vim 的核心理念——其界面本身就是一种编程语言——即使你不直接使用 Vim，也可以通过各种编辑器的 Vim 模式（如 [VSCodeVim](https://marketplace.visualstudio.com/items?itemName=vscodevim.vim)）来受益。

我曾经觉得不会用 Vim 的程序员不算合格，但工作两年后发现自己大部分时间还是在用 IDE——尤其有了 [Cursor](https://cursor.com/) 之后，AI 辅助编程的效率实在太高了。话虽如此，熟练掌握 Vim 依然很有价值：**当你 SSH 到远程服务器上 debug 时，Vim 往往是唯一可用的编辑器。** 能用 Vim 快速改配置、查日志、定位问题，效率远超其他方式。不必用 Vim 写整个项目，但掌握基本的移动、编辑和搜索替换就够了。

**移动（Normal 模式）**

| 层级 | 按键 | 说明 |
|------|------|------|
| 基础 | `h` `j` `k` `l` | 左、下、上、右 |
| 词级 | `w` / `b` / `e` | 下一个词首 / 当前词首 / 当前词尾 |
| 行级 | `0` / `^` / `$` | 行首 / 第一个非空字符 / 行尾 |
| 屏幕级 | `H` / `M` / `L` | 屏幕顶 / 中 / 底 |
| 翻页 | `Ctrl-u` / `Ctrl-d` | 上翻半页 / 下翻半页 |
| 文件级 | `gg` / `G` / `:{行号}` | 文件头 / 文件尾 / 跳转到指定行 |
| 查找 | `f{字符}` / `/{正则}` | 行内查找 / 全文搜索，`n`/`N` 切换匹配 |

**编辑**

- `d{移动}` — 删除：`dw` 删一个词，`d$` 删到行尾，`dd` 删整行
- `c{移动}` — 修改（删除并进入插入模式）：`cw` 改一个词，`ci(` 改括号内的内容
- `y{移动}` — 复制：`yy` 复制整行，`p` 粘贴
- `u` — 撤销，`Ctrl-r` — 重做
- `O` / `o` — 在当前行上方 / 下方新开一行

**计数前缀**

数字可以与任何操作组合：`3w` 前进 3 个词，`5j` 下移 5 行，`7dw` 删除 7 个词。

**配置**

编辑 `~/.vimrc` 即可自定义 Vim。推荐从课程提供的[基础配置](https://missing-semester-cn.github.io/2020/files/vimrc)起步，它修复了不少默认行为的坑。我自己的配置在[这里](https://github.com/zyksir/dotfiles/blob/main/dotfiles/vimrc)。

> **拓展阅读**：课程推荐了以下 Vim 学习资源——[Practical Vim](https://pragprog.com/titles/dnvim2/)（书）、[Vim Screencasts](http://vimcasts.org/)、[VimGolf](https://www.vimgolf.com/)（用最少按键完成编辑挑战）、[Vim Adventures](https://vim-adventures.com/)（游戏化学习）。此外，安装了 Vim 后可以直接在终端运行 `vimtutor`，这是官方自带的交互式入门教程。2020 版课程有一个[更深入的 Vim 讲义](https://missing.csail.mit.edu/2020/editors/)也值得参考。

## 第四讲：Debugging & Profiling

**课程概要**：[第四讲](https://missing.csail.mit.edu/2026/debugging-profiling/)覆盖面非常广——从最基础的 printf debugging、日志（logging），到 `gdb`/`lldb` 调试器、record-replay debugging（[rr](https://rr-project.org/)）、系统调用追踪（`strace`/`bpftrace`）、内存调试（AddressSanitizer、[Valgrind](https://valgrind.org/)），再到性能分析工具（`time`、[perf](https://www.man7.org/linux/man-pages/man1/perf.1.html)、[flamegraph](https://www.brendangregg.com/flamegraphs.html)、[hyperfine](https://github.com/sharkdp/hyperfine)）。2026 版本还新增了 AI 辅助 debugging 和数据可视化（`gnuplot`、`matplotlib`）的内容。

**我的看法**

课程中的 profiling 内容偏向 CPU 层面，而在我目前的工作中，**GPU profiling 往往更为关键**——[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)、[PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 等工具是课程尚未覆盖的。

不过，**profiling 的核心思想是通用的：先测量，再优化。** 不要凭直觉猜瓶颈在哪里，要用数据说话。CPU 层面用 `perf` / flamegraph 定位热点，GPU 层面用 Nsight / PyTorch Profiler 分析 kernel 耗时。课程中提到的 [Speedscope](https://www.speedscope.app/)（交互式 flamegraph 查看器）和 [Perfetto](https://perfetto.dev/)（系统级分析）也是很好的可视化工具。

## 第五讲：版本控制（Git）

**课程概要**：[第五讲](https://missing.csail.mit.edu/2026/version-control/)从 Git 的数据模型讲起——用 blob（文件）、tree（目录）、commit（快照）三个概念解释 Git 的底层设计，然后介绍 branch、merge、staging area 等核心机制。课程的方法论是"自底向上"：先理解数据模型，再学命令，就不会觉得 Git 是在念咒语了。

Git 要深入挖掘的话内容非常多（感兴趣推荐 [Pro Git](https://git-scm.com/book/zh/v2)，读完前 5 章即可），但**日常开发只需要掌握一套基础工作流就够了**，遇到复杂场景再现学也来得及。

> 掌握 `add → commit → push` 和分支管理就能应对 90% 的场景。`rebase` 冲突、`cherry-pick`、`bisect` 这些高级命令，需要的时候再查也不迟。

**日常开发够用的 Git 命令**

```bash
# 基础操作
git status / git diff / git log --oneline --graph
git add <file> / git commit -m "message"
git push / git pull

# 分支操作
git branch <name> / git checkout -b <name>
git merge <branch> / git rebase <branch>

# 撤销与修复
git stash / git stash pop
git reset HEAD~1          # 撤销最近一次 commit（保留修改）
git checkout -- <file>    # 丢弃工作区修改
git restore <file>        # 更现代的写法，等价于 git checkout -- <file>
```

> **拓展阅读**：课程推荐的 Git 学习资源，放进收藏夹吃灰去吧——[Learn Git Branching](https://learngitbranching.js.org/)（浏览器里的交互式 Git 学习游戏）、[Oh Shit, Git!?!](https://ohshitgit.com/)（常见 Git 错误的快速修复指南）、[Git from the Bottom Up](https://jwiegley.github.io/git-from-the-bottom-up/)（深入理解 Git 内部实现）、[How to explain git in simple words](https://smusamashah.github.io/blog/2017/10/14/explain-git-in-simple-words)。写好 commit message 也很重要，推荐阅读 [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)。

## 后续课程预览（待整理）

2026 版本新增了几门非常有价值的讲座，以下是简要介绍：

**[Packaging and Shipping Code](https://missing.csail.mit.edu/2026/shipping-code/)**（第六讲）：讲解依赖管理、虚拟环境、容器化（Docker）、CI/CD 等发布流程。对于需要把代码交付给他人运行的场景非常实用。

**[Agentic Coding](https://missing.csail.mit.edu/2026/agentic-coding/)**（第七讲）：这是 2026 版本最大的亮点之一。讲解 AI coding agent 的工作原理（LLM + 工具调用）、使用场景（实现功能、修 bug、重构、代码审查、vibe coding），以及进阶技巧（`AGENTS.md`、subagent、context 管理）。课程使用的工具包括 [Claude Code](https://www.claude.com/product/claude-code)、[Codex](https://openai.com/codex/) 等。一个值得记住的观点：AI 工具可以犯错，不要把它当作万能替代品，计算思维仍然很有价值。

**[Beyond the Code](https://missing.csail.mit.edu/2026/beyond-code/)**（第八讲）和 **[Code Quality](https://missing.csail.mit.edu/2026/code-quality/)**（第九讲）：分别涵盖软件工程中代码之外的话题（安全、协作等）以及代码质量保障（linting、formatting、type checking、CI）。

后面这四讲的详细笔记还在整理中。课程内容和三年前相比已经发生了巨大变化——Data Wrangling、Security and Cryptography 等旧专题被新增的 Agentic Coding、Code Quality 等取代，旧内容淘汰了近一半，课程更新速度令人感慨。

MIT Missing Semester 确实是一门值得花时间看的课程。正如它的名字所说，这些是大学里"缺失的一学期"，越早补上越好。当然也不必在上面花太多时间——了解一下这些工具的存在，在需要时知道去哪里找答案，就已经能显著提升你的开发效率了。

--- 
**参考链接汇总**

- **课程主页**：[missing.csail.mit.edu](https://missing.csail.mit.edu/) ｜ [YouTube 频道](https://www.youtube.com/@MissingSemester) ｜ [中文翻译（2020 版）](https://missing-semester-cn.github.io/)
- **Shell & 工具**：[fzf](https://github.com/junegunn/fzf) ｜ [Oh My Zsh](https://ohmyz.sh/) ｜ [powerlevel10k](https://github.com/romkatv/powerlevel10k) ｜ [zsh-z](https://github.com/agkozak/zsh-z) ｜ [tldr](https://tldr.sh/)
- **tmux**：[tmux Wiki](https://github.com/tmux/tmux/wiki) ｜ [Beginner's Guide](https://towardsdatascience.com/a-beginners-guide-to-tmux-a-multitasking-superpower-for-your-terminal/) ｜ [iTerm2](https://iterm2.com/)
- **Vim**：[vimtutor（内置）](https://vimhelp.org/usr_01.txt.html) ｜ [Practical Vim](https://pragprog.com/titles/dnvim2/) ｜ [VimGolf](https://www.vimgolf.com/) ｜ [Vim Adventures](https://vim-adventures.com/) ｜ [2020 版深入讲义](https://missing.csail.mit.edu/2020/editors/)
- **Debugging & Profiling**：[Brendan Gregg's Flamegraphs](https://www.brendangregg.com/flamegraphs.html) ｜ [Speedscope](https://www.speedscope.app/) ｜ [Perfetto](https://perfetto.dev/) ｜ [rr（record-replay debugger）](https://rr-project.org/) ｜ [hyperfine](https://github.com/sharkdp/hyperfine)
- **Git**：[Pro Git（中文版）](https://git-scm.com/book/zh/v2) ｜ [Learn Git Branching](https://learngitbranching.js.org/) ｜ [Oh Shit, Git!?!](https://ohshitgit.com/) ｜ [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- **我的 dotfiles**：[github.com/zyksir/dotfiles](https://github.com/zyksir/dotfiles)（zshrc、vimrc、tmux.conf、gitconfig）
