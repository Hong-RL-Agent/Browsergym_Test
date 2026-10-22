
const path = require('path');

// Resolve RL_Errorsite root from environment or sibling directory
const root = path.resolve(
  process.env.RL_ERRORSITE_ROOT ||
  path.resolve(__dirname, '../../../RL_Errorsite')
);

// Load dependencies from RL_Errorsite
const express = require(
  require.resolve('express', { paths: [root] })
);

const manifest = require(
  path.join(root, 'manifest.json')
);

const servers = [];

for (const site of manifest) {
  const app = express();

  app.use(
    '/vendor',
    express.static(path.join(root, 'node_modules/react/umd'))
  );

  app.use(
    '/vendor',
    express.static(path.join(root, 'node_modules/react-dom/umd'))
  );

  app.use(
    '/vendor',
    express.static(path.join(root, 'node_modules/@babel/standalone'))
  );

  app.use(express.static(path.join(root, site.path)));

  app.get('/__meta', (_req, res) => res.json(site));

  const server = app.listen(site.port, () => {
    console.log(`STARTED ${site.id} ${site.port}`);
  });

  server.on('error', (err) => {
    console.error(`PORT_ERROR ${site.id} ${site.port} ${err.code}`);
  });

  servers.push(server);
}

// Graceful shutdown
process.on('SIGINT', () => {
  for (const server of servers) server.close();
  process.exit(0);
});

process.on('SIGTERM', () => {
  for (const server of servers) server.close();
  process.exit(0);
});

setInterval(() => {}, 1000);
