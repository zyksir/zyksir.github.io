---
layout: post
title: "CMake Tutorial"
subtitle: "CMake 学习笔记"
date: 2023-11-18
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: ["CMake", "tool"]
---

## Introduction

文本为 zyk 学习 CMake 的学习笔记。记录了我一步一步学习 CMake 的一些过程，目的是当我之后遇到 CMake 的问题的时候，我可以通过检索这个笔记，快速回忆起之前的内容。主要是参考了 [cmake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) 和 [上交大学的分享](https://www.bilibili.com/video/BV14h41187FZ/?vd_source=cd1453e3dd41871bffd9b41e9617d6cc), 其代码仓库地址在 [这里](https://github.com/stdrc/modern-cmake-by-example)。

## 基本语法

- `cmake_minimum_required(VERSION 3.9)`: 指定最小 CMake 版本要求为 3.9。
- `project(PROJ_NAME)`: 指定 project 名称为 PROJ_NAME
- `add_executable(<name> <source1> <source2>)`: 可执行文件 `<name>` 依赖哪些源文件(source file, e.g 以`.cpp`结尾的文件)。这里不需要指定头文件(`.h`或者`.hpp`结尾)，因为 CMake 会自动找到依赖的头文件。
- `aux_source_directory(<dir> <var>)`: 获取`dir`目录下的所有源文件，将其存储在`var`里。这样在 `add_executable`的时候可以用过`${var}`获取所有的源文件。
- `message(STATUS MSG_STRING)`: message 可用于打印调试信息或错误信息，除了 STATUS 外还有 DEBUG WARNING SEND_ERROR FATAL_ERROR 等。
- `set(<var> <val>)`: 设定变量以及其对应值。
  - `set(CMAKE_CXX_STANDARD 11)`: 指定C++版本为 C++11。
- `set(<var> <default_val> CACHE <msg>)`: 为`var`指定`default_val`的值，可以通过`-Dvar=val`的方式来修改这个值。`<msg>`是描述`<var>`的说明。
- `target_compile_definitions(<target> PUBLIC <val>)`: 对应编译时的`-D`的选项，会传入宏定义。这里的`<val>`既可以是`FOO`这s(<target> INTERFACE <val>)`: 设定编译时选项.
  - 这个示例代表在编译`target`时使用 cxx_std_20: `target_compile_features(target INTERFACE cxx_std_20)`
  - 还能指定`cxx_auto_type、cxx_lambda`等 feature。
- 构建 cmake 项目的命令如下

```cmake
cmake -B build          # 生成构件目录
cmake --build build     # 执行构建

# 亦或是手动构件 build 命令
# 然后用 cmake .. 来生成 makefile 文件，最后执行 makefile
mkdir build && cd build && cmake .. && make
```

## 构建 library

- `add_library(<target> STATIC [<source>...])`: (在 library 对应的 CMakelist 里)添加 `<target>` 库目标，STATIC 指定为静态库, 依赖`<source>`对应的源文件。
  - `add_library(<target> INTERFACE)`: INTERFACE 类型的 target 一般用于没有源文件的情况，比如header-only 库，或者只是为了抽象地提供一组 target_xxx的配置。INTERFACE target 的后续所有 target_xxx 都必须也使用
INTERFACE，效果将会直接应用到链接此库的 target 上。
- `target_include_directories(<target> PUBLIC <dir>)`: 编译`<target>`目标时所需要用到的目录。与之对应的，`include_directories(<dir>)`中`dir`会全局生效，而此处的`dir` 仅在生成`target`时生效。这里的`dir`会被加在`-I`这个选项里。
- `target_link_libraries(<target> <libitem>)`: 为 `<target>` 可执行目标链接 `<libitem>`。其中`<target>`之前通过`add_executable`或者`add_library`创建；`<libitem>`为库的名字。
- `add_subdirectory(source_dir)`: 表明`source_dir`也是一个需要 build 的目录。会立即处理`source_dir`下的`CMakelist`。这里的`source_dir`不会被加到`-I`里面去。

## language && CUDA

如果使用了 CMake，涉及到的语言主要就是C, Cpp, CUDA。下面会以 CUDA 为例子，讲 CUDA 项目所需要用到的 CMake 语法。

- `project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)`: 
- `find_package(CUDA REQUIRED)`
- `check_language(CUDA)`: 在此之前需要先`include(CheckLanguage)`。这个会检查 CUDA 相关的配置是否都齐。
- `set(CUDA_COMPUTE_CAPABILITY 80)`: 表示只生成 80(A100的 computer capability 就是80) 的代码。参考[这里](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
- `target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})`: 为了使用相关库，我们需要 link，和 CUDA 相关的环境变量在[这里](https://cmake.org/cmake/help/v3.0/module/FindCUDA.html)。

## 关于依赖

- `find_package(<name> REQUIRED)`: 用于在系统中寻找已经安装的第三方库的头文件和库文件的位置，并创建一个名为 CURL::libcurl 的库目标，以供链接。
- 下面是一个使用 `FetchContent` 的示例: Catch2是一个用于测试的例子。关于测试，可以参考[上交的示例](https://github.com/stdrc/modern-cmake-by-example/tree/12_testing)，目前没有用到，因此不做深挖。

```cmake
# 导入 FetchContent 相关命令
include(FetchContent)

# 描述如何获取 Catch2
FetchContent_Declare(
    catch2 # 建议使用全小写
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.0-preview3)

# 一条龙地下载、构建 Catch2
FetchContent_MakeAvailable(catch2)
```