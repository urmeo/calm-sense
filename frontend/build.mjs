import { build, context } from 'esbuild';
import { createServer } from 'node:http';
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { staticAssetsPlugin } from './static-assets.mjs';
import { dashboardRoutes } from './src/lib/routes.ts';

const root = dirname(fileURLToPath(import.meta.url));
const output = resolve(root, '../outputs/generated/site');
const base = '/calm-sense/';
const mode = process.argv[2] ?? 'build';
const args = process.argv.slice(3);
const clients = new Set();
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json',
  '.webmanifest': 'application/manifest+json', '.png': 'image/png',
  '.gif': 'image/gif', '.mp4': 'video/mp4',
  '.ico': 'image/x-icon', '.svg': 'image/svg+xml', '.txt': 'text/plain; charset=utf-8',
};

function flag(name, fallback) {
  const index = args.indexOf(name);
  const inline = args.find((value) => value.startsWith(`${name}=`));
  return inline?.slice(name.length + 1) ?? (index < 0 ? fallback : args[index + 1]);
}

async function notices(inputs) {
  const packages = new Set();
  for (const input of Object.keys(inputs)) {
    if (!input.includes('node_modules/')) continue;
    let directory = dirname(resolve(root, input));
    while (directory.startsWith(root + sep)) {
      try {
        await stat(join(directory, 'package.json'));
        packages.add(directory);
        break;
      } catch { directory = dirname(directory); }
    }
  }
  const entries = [];
  for (const directory of [...packages].sort()) {
    const info = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8'));
    const files = (await readdir(directory)).filter((name) => /^(licen[sc]e|notice|copying)([.-]|$)/i.test(name));
    const content = await Promise.all(files.map((name) => readFile(join(directory, name), 'utf8')));
    entries.push(`${info.name}@${info.version} (${info.license ?? 'See notice'})\n${content.join('\n')}`);
  }
  return entries.join('\n\n' + '='.repeat(72) + '\n\n') + '\n';
}

async function publish(result) {
  const entry = Object.entries(result.metafile.outputs).find(([, value]) => value.entryPoint === 'src/index.tsx');
  if (!entry) throw new Error('Dashboard entry is missing from the bundle.');
  const asset = (path) => base + relative(output, resolve(root, path)).split(sep).join('/');
  let html = (await readFile(join(root, 'index.html'), 'utf8')).replaceAll('%BASE_URL%', base);
  html = html.replace('src="/src/index.tsx"', `src="${asset(entry[0])}"`);
  if (entry[1].cssBundle) html = html.replace('</head>', `  <link rel="stylesheet" href="${asset(entry[1].cssBundle)}" />\n  </head>`);
  if (mode === 'dev') {
    html = html.replace('</body>', `<script type="module">new EventSource('${base}__reload').onmessage = () => location.reload();</script>\n  </body>`);
  }
  await writeFile(join(output, 'index.html'), html);
  for (const { path } of dashboardRoutes) {
    if (path === '/') continue;
    const directory = join(output, path.slice(1));
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, 'index.html'), html);
  }
  await writeFile(join(output, 'licenses.txt'), await notices(result.metafile.inputs));
  for (const response of clients) response.write('data: reload\n\n');
}

async function serve() {
  const port = Number(flag('--port', mode === 'dev' ? '5173' : '4173'));
  const host = flag('--host', '127.0.0.1');
  if (!Number.isInteger(port) || port < 0 || port > 65535 || !host) throw new Error('Invalid host or port.');
  const server = createServer(async (request, response) => {
    try {
      if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
      const url = new URL(request.url ?? '/', 'http://localhost');
      const path = decodeURIComponent(url.pathname);
      if (path === '/') { response.writeHead(302, { Location: base }).end(); return; }
      if (!path.startsWith(base)) { response.writeHead(404).end('Not found'); return; }
      if (path === `${base}__reload` && mode === 'dev') {
        response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
        response.write(': connected\n\n');
        clients.add(response);
        request.on('close', () => clients.delete(response));
        return;
      }
      let file = resolve(output, path.slice(base.length) || 'index.html');
      if (!file.startsWith(output + sep)) { response.writeHead(403).end(); return; }
      try {
        if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
        if (!(await stat(file)).isFile()) throw new Error('Not a file');
      } catch {
        if (extname(file)) { response.writeHead(404).end('Not found'); return; }
        file = join(output, 'index.html');
      }
      const content = await readFile(file);
      response.writeHead(200, {
        'Content-Type': types[extname(file)] ?? 'application/octet-stream',
        'Content-Length': content.length, 'Cache-Control': mode === 'dev' ? 'no-store' : 'no-cache',
        'X-Content-Type-Options': 'nosniff',
      });
      response.end(request.method === 'HEAD' ? undefined : content);
    } catch {
      if (!response.headersSent) response.writeHead(400);
      response.end('Invalid request');
    }
  });
  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(port, host, accept);
  });
  console.log(`Dashboard: http://${host}:${server.address().port}${base}`);
  return server;
}

try {
  if (!['build', 'dev', 'preview'].includes(mode)) throw new Error('Usage: node build.mjs build|dev|preview [--host HOST] [--port PORT]');
  if (mode === 'preview') {
    await stat(join(output, 'index.html'));
    await serve();
  } else {
    await rm(output, { recursive: true, force: true });
    await mkdir(output, { recursive: true });
    const options = {
      absWorkingDir: root, entryPoints: ['src/index.tsx'], outdir: join(output, 'assets'),
      entryNames: '[name]-[hash]', chunkNames: 'chunks/[name]-[hash]', assetNames: '[name]-[hash]',
      bundle: true, splitting: true, format: 'esm', platform: 'browser', target: 'es2022',
      jsx: 'automatic', minify: mode !== 'dev', legalComments: 'eof', metafile: true,
      loader: { '.svg': 'file' },
      publicPath: `${base}assets`,
      define: { 'import.meta.env.BASE_URL': JSON.stringify(base), 'process.env.NODE_ENV': JSON.stringify(mode === 'dev' ? 'development' : 'production') },
      logLevel: 'info', plugins: [staticAssetsPlugin({ root, output, publish })],
    };
    if (mode === 'build') {
      await build(options);
    } else {
      const builder = await context(options);
      await builder.rebuild();
      await builder.watch();
      const server = await serve();
      for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, async () => {
        for (const client of clients) client.end();
        server.close();
        await builder.dispose();
        process.exit(0);
      });
    }
  }
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
