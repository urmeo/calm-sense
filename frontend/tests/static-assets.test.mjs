import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout } from 'node:timers/promises';
import { test } from 'node:test';
import { context } from 'esbuild';
import { staticAssetsPlugin } from '../static-assets.mjs';

async function fixture(t, publish) {
  const directory = await mkdtemp(join(tmpdir(), 'calmsense-assets-'));
  const root = join(directory, 'frontend');
  const output = join(directory, 'site');
  let builder;
  t.after(async () => {
    await builder?.dispose();
    await rm(directory, { recursive: true, force: true });
  });
  for (const name of ['src', 'public/media']) {
    await mkdir(join(root, name), { recursive: true });
  }
  await writeFile(join(root, 'src/index.tsx'), 'export const value = 1;\n');
  await writeFile(join(root, 'index.html'), '<title>Original</title>');
  await writeFile(join(root, 'public/manifest.webmanifest'), '{"name":"Original"}\n');
  await writeFile(join(root, 'public/media/clip.txt'), 'original');
  builder = await context({
    absWorkingDir: root, entryPoints: ['src/index.tsx'],
    outdir: join(output, 'assets'), bundle: true, format: 'esm',
    metafile: true, logLevel: 'silent',
    plugins: [staticAssetsPlugin({ root, output, publish })],
  });
  return { root, output, builder };
}

async function waitFor(check, description) {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    try {
      if (await check()) return;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    await setTimeout(40);
  }
  assert.fail('Timed out waiting for ' + description);
}

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

test('development rebuilds HTML, manifest and nested public edits without a source change', { timeout: 30000 }, async (t) => {
  let html;
  let publications = 0;
  const { root, output, builder } = await fixture(t, async () => {
    html = await readFile(join(root, 'index.html'), 'utf8');
    publications++;
  });
  await builder.watch();
  await waitFor(() => publications > 0, 'initial publication');

  await writeFile(join(root, 'index.html'), '<title>Changed</title>');
  await waitFor(() => html === '<title>Changed</title>', 'HTML edit');

  await writeFile(join(root, 'public/media/clip.txt'), 'changed');
  await waitFor(async () => await readFile(join(output, 'media/clip.txt'), 'utf8') === 'changed', 'public file edit');

  await mkdir(join(root, 'public/media/nested'));
  await writeFile(join(root, 'public/media/nested/new.txt'), 'new');
  await waitFor(() => exists(join(output, 'media/nested/new.txt')), 'new nested asset');

  await rename(join(root, 'public/media/clip.txt'), join(root, 'public/media/renamed.txt'));
  await waitFor(async () => await exists(join(output, 'media/renamed.txt'))
    && !await exists(join(output, 'media/clip.txt')), 'renamed asset');

  await rm(join(root, 'public/media/nested'), { recursive: true });
  await waitFor(async () => !await exists(join(output, 'media/nested')), 'deleted nested asset');

  await writeFile(join(root, 'public/manifest.webmanifest'), '{"name":"Changed"}\n');
  await waitFor(async () => JSON.parse(await readFile(join(output, 'manifest.webmanifest'), 'utf8')).name === 'Changed', 'manifest edit');
  assert.ok(await exists(join(output, 'assets/index.js')), 'static cleanup preserves the bundle');
  assert.equal(await readFile(join(root, 'src/index.tsx'), 'utf8'), 'export const value = 1;\n');
});

test('a malformed manifest reports a build failure and a corrected edit recovers', async (t) => {
  let publications = 0;
  const { root, output, builder } = await fixture(t, async () => { publications++; });
  await builder.rebuild();
  for (const source of ['{', '{"name":', 'null', '[]']) {
    await writeFile(join(root, 'public/manifest.webmanifest'), source);
    await assert.rejects(builder.rebuild(), /Build failed/);
    assert.equal(publications, 1);
  }
  await writeFile(join(root, 'public/manifest.webmanifest'), '{"name":"Recovered"}\n');
  await builder.rebuild();
  assert.equal(JSON.parse(await readFile(join(output, 'manifest.webmanifest'), 'utf8')).name, 'Recovered');
  assert.equal(publications, 2);
});

test('public paths can change between files and directories without retaining stale content', async (t) => {
  const { root, output, builder } = await fixture(t, async () => {});
  const source = join(root, 'public/change');
  const target = join(output, 'change');
  await writeFile(source, 'file');
  await builder.rebuild();
  assert.equal(await readFile(target, 'utf8'), 'file');

  await rm(source);
  await mkdir(source);
  await writeFile(join(source, 'nested.txt'), 'nested');
  await builder.rebuild();
  assert.equal(await readFile(join(target, 'nested.txt'), 'utf8'), 'nested');

  await rm(source, { recursive: true });
  await writeFile(source, 'file again');
  await builder.rebuild();
  assert.equal(await readFile(target, 'utf8'), 'file again');
  assert.ok(await exists(join(output, 'assets/index.js')));
});
