import { pathToFileURL } from 'node:url';

const viteModule = await import(pathToFileURL('C:/workspace/RL_Errorsite_fullstack_run4/site100_SaaS구독요금제관리_사용량요금제불일치/frontend/node_modules/vite/dist/node/index.js'));
const reactModule = await import(pathToFileURL('C:/workspace/RL_Errorsite_fullstack_run4/site100_SaaS구독요금제관리_사용량요금제불일치/frontend/node_modules/@vitejs/plugin-react/dist/index.js'));
const { createServer } = viteModule;
const react = reactModule.default;

const args = process.argv.slice(2);
const value = (name) => {
  const i = args.indexOf(name);
  if (i < 0 || i + 1 >= args.length) throw new Error(`missing ${name}`);
  return args[i + 1];
};

const root = value('--root');
const port = Number(value('--port'));
const target = value('--target');
const cacheDir = `C:/workspace/browsergym/.run6-vite-cache-${port}`;

const server = await createServer({
  root,
  configFile: false,
  cacheDir,
  optimizeDeps: { disabled: true },
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port,
    strictPort: true,
    proxy: {
      '/api': { target, changeOrigin: true },
    },
  },
});

await server.listen();
server.printUrls();
console.log(JSON.stringify({ root, port, target, pid: process.pid }));

const stop = async () => {
  await server.close();
  process.exit(0);
};
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
