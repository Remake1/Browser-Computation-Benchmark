import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';

// Cross-origin isolation: required for SharedArrayBuffer (WASM threads) and for
// the finer-grained performance.now(). Production is served by Nginx with the
// same headers (deploy/nginx/); `vite preview` mirrors them for convenience.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function gitCommit(): string {
  try {
    const hash = execSync('git rev-parse HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
    const dirty = execSync('git status --porcelain', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
    return dirty ? `${hash}-dirty` : hash;
  } catch {
    return 'unknown'; // no git, or no commits yet
  }
}

export default defineConfig({
  define: {
    __GIT_COMMIT__: JSON.stringify(gitCommit()),
  },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
