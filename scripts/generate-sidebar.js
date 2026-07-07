const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const notesDir = path.join(rootDir, 'notes');
const sidebarPath = path.join(rootDir, '_sidebar.md');

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
  const content = fs.readFileSync(filePath, 'utf8');
  const heading = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => /^#\s+/.test(line));

  return heading ? heading.replace(/^#\s+/, '').trim() : cleanTitle(path.basename(filePath));
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
    lines.push(`${indent}- [${getMarkdownTitle(fullPath)}](<${relativePath}>)`);
  }

  return lines;
}

const sidebar = [
  '- [HongjianMa的学习笔记](/)',
  '',
  '- 使用指南',
  '  - [博客使用说明](notes/getting-started.md)',
  '',
  '- 全部笔记',
  ...buildSidebarLines(notesDir, 1),
  '',
].join('\n');

fs.writeFileSync(sidebarPath, sidebar, 'utf8');

console.log(`Generated ${path.relative(rootDir, sidebarPath)}`);
