import { cp, readFile, readdir, realpath, rm, rmdir } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';

async function publicInputs(directory) {
  const files = [];
  const directories = [directory];
  for (const entry of await readdir(directory, { recursive: true, withFileTypes: true })) {
    const path = join(entry.parentPath, entry.name);
    (entry.isDirectory() ? directories : files).push(path);
  }
  return { files, directories };
}

export function staticAssetsPlugin({ root, output, publish }) {
  const publicDirectory = join(root, 'public');
  const html = join(root, 'index.html');
  const manifest = join(publicDirectory, 'manifest.webmanifest');
  let publishedFiles = new Set();
  let publishedDirectories = new Set();

  return {
    name: 'dashboard-assets',
    async setup(build) {
      const entry = await realpath(join(root, 'src/index.tsx'));
      build.onLoad({ filter: /[/\\]src[/\\]index\.tsx$/, namespace: 'file' }, async ({ path }) => {
        if (path !== entry) return;
        const inputs = await publicInputs(publicDirectory);
        return {
          contents: await readFile(path, 'utf8'),
          loader: 'tsx',
          resolveDir: dirname(path),
          watchFiles: [path, html, ...inputs.files],
          watchDirs: inputs.directories,
        };
      });

      build.onEnd(async (result) => {
        if (result.errors.length || !result.metafile) return;
        const settings = JSON.parse(await readFile(manifest, 'utf8'));
        if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
          throw new Error('Dashboard manifest must contain an object.');
        }
        const { files, directories } = await publicInputs(publicDirectory);
        const currentFiles = new Set(files.map((path) => relative(publicDirectory, path)));
        const currentDirectories = new Set(directories.map((path) => relative(publicDirectory, path)).filter(Boolean));
        for (const path of publishedFiles) {
          if (!currentFiles.has(path)) await rm(join(output, path), { force: true });
        }
        for (const path of [...publishedDirectories].sort((a, b) => b.length - a.length)) {
          if (currentDirectories.has(path)) continue;
          try {
            await rmdir(join(output, path));
          } catch (error) {
            if (!['ENOENT', 'ENOTEMPTY'].includes(error.code)) throw error;
          }
        }
        publishedFiles = currentFiles;
        publishedDirectories = currentDirectories;
        await cp(publicDirectory, output, { recursive: true });
        await publish(result);
      });
    },
  };
}
