---
layout: post
title: "Makefile Tutorial"
subtitle: "This is a subtitle"
date: 2023-11-18
author: "Yikai"
header-img: "img/background/post-default-bg.jpg"
tags: ["tools"]
---

## Introduction

本文为 zyk 学习 makefile 的学习笔记。希望以后想写 makefile 的时候，可以用这篇文档作为一个索引。

## 符号及其含义

- $@

## 奇技淫巧

### .o 文件的自动推断

Makefile 可以自动推断 .o 目标文件需要依赖同名的 .cpp 文件，所以其实不需要在依赖中指定 main.cpp 和 answer.cpp，也不需要写编译 commands，它知道要用 CXX 变量制定的命令作为 C++ 编译器。这里只需要指定目标文件所依赖的头文件，使头文件变动时可以重新编译对应目标文件。

```Makefile
main.o: answer.hpp
answer.o: answer.hpp
```