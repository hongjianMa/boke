const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const notesDir = path.join(rootDir, 'notes');
const sidebarPath = path.join(rootDir, '_sidebar.md');
const allNotesPath = path.join(rootDir, 'all-notes.md');

const ignoredFiles = new Set(['getting-started.md']);
const markdownExt = '.md';

function normalizePath(filePath) {
  return filePath.split(path.sep).join('/');
}

function cleanTitle(name) {
  return name
    .replace(/\.md$/i, '')
    .replace(/^\d+[-_、.\s]+/, '')
    .trim();
}

function getMarkdownTitle(filePath) {
  const fileNameTitle = cleanTitle(path.basename(filePath));

  if (fileNameTitle.length <= 12) {
    return fileNameTitle;
  }

  const content = fs.readFileSync(filePath, 'utf8');
  const heading = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => /^#\s+/.test(line));

  return heading ? heading.replace(/^#\s+/, '').trim() : fileNameTitle;
}

function sortByNumberThenName(a, b) {
  const getNumber = (name) => {
    const match = name.match(/^(\d+)/);
    return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
  };

  const numberA = getNumber(a.name);
  const numberB = getNumber(b.name);

  if (numberA !== numberB) return numberA - numberB;
  return a.name.localeCompare(b.name, 'zh-Hans-CN', { numeric: true });
}

function hasMarkdownFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).some((entry) => {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      return hasMarkdownFiles(fullPath);
    }

    return entry.isFile()
      && entry.name.endsWith(markdownExt)
      && !ignoredFiles.has(entry.name);
  });
}

function buildSidebarLines(dir, depth = 1) {
  const entries = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => {
      if (entry.isDirectory()) return hasMarkdownFiles(path.join(dir, entry.name));
      return entry.isFile()
        && entry.name.endsWith(markdownExt)
        && !ignoredFiles.has(entry.name);
    })
    .sort(sortByNumberThenName);

  const lines = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const indent = '  '.repeat(depth);

    if (entry.isDirectory()) {
      lines.push(`${indent}- ${cleanTitle(entry.name)}`);
      lines.push(...buildSidebarLines(fullPath, depth + 1));
      continue;
    }

    const relativePath = normalizePath(path.relative(rootDir, fullPath));
    const title = getMarkdownTitle(fullPath);
    lines.push(`${indent}- [${title}](<${relativePath}>)`);
  }

  return lines;
}

const sidebar = [
  '- [HongjianMa的学习笔记](/)',
  '',
  '- [全部笔记](all-notes.md)',
  ...buildSidebarLines(notesDir, 1),
  '',
].join('\n');

fs.writeFileSync(sidebarPath, sidebar, 'utf8');

const allNotes = [
  '# HongjianMa的学习笔记',
  '',
  '> 欢迎来到马神的知识仓库：这里不保证每一篇都像论文一样严肃，但保证每一篇都在认真变强。',
  '',
  '这里收着我的科研阅读、推荐系统、深度学习、算法与保研面试笔记。平时想到哪、学到哪、踩到坑了也记到哪——主打一个“脑子负责探索，博客负责别让我忘”。',
  '',
  '## 怎么逛',
  '',
  '- 想按主题看：点左侧目录，文件夹都可以展开/收起。',
  '- 想直接找东西：用左上角搜索框，关键词一敲，知识自己出来。',
  '- 想看最新整理：通常新笔记会自动出现在下面的目录里，不用手动翻 GitHub。',
  '',
  '## 全部笔记',
  '',
  ...buildSidebarLines(notesDir, 0),
  '',
  '---',
  '',
  '慢慢看，不着急。知识这东西，今天多懂一点，明天就少慌一点。马神继续施工中。🚧',
  '',
].join('\n');

fs.writeFileSync(allNotesPath, allNotes, 'utf8');

console.log(`Generated ${path.relative(rootDir, sidebarPath)}`);
console.log(`Generated ${path.relative(rootDir, allNotesPath)}`);
