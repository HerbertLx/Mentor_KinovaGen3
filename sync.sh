#!/bin/bash

# 1. 检查是否有文件变动
if [ -z "$(git status --porcelain)" ]; then
    echo "✨ 没有发现任何改动，无需同步。"
    exit 0
fi

# 2. 提示输入本次修改的内容描述
echo "🚀 发现文件改动，请输入本次提交的描述 (Description):"
read desc

# 如果用户直接回车，给一个默认描述
if [ -z "$desc" ]; then
  desc="Update: $(date +'%Y-%m-%d %H:%M:%S')"
fi

# 3. 执行 Git 三部曲
echo "正在暂存文件..."
git add .

echo "正在提交: $desc"
git commit -m "$desc"

echo "正在推送到 GitHub..."
git push origin main

echo "✅ 同步完成！"