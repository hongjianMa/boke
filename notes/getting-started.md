# 使用说明

## 新增笔记

1. 在 `notes` 文件夹中新建一个 `.md` 文件，例如 `my-note.md`。
2. 使用 Markdown 编写内容。
3. 在根目录的 `_sidebar.md` 中添加链接：

```markdown
- [我的新笔记](notes/my-note.md)
```

提交并推送到 GitHub 后，网站会自动显示最新内容。

## 插入图片

把图片放进 `assets` 文件夹，然后在笔记中写：

```markdown
![图片说明](assets/example.png)
```

## 常用 Markdown

```markdown
# 一级标题
## 二级标题

**加粗**、*斜体*、`行内代码`

- 无序列表
- 第二项

[链接文字](https://example.com)
```
