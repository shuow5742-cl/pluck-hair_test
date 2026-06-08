# Pluck Frontend (Vite)

React 18 + Vite + React Router + Tailwind dashboard for the 燕窝智能挑毛监控系统。

## 运行方式（纯 Deno）

- 依赖通过 Deno 的 npm 兼容层缓存，无需 Node / pnpm / node_modules。
- 命令均在 `frontend` 目录执行：

```bash
deno task dev       # 开发
deno task build     # 产物构建
deno task preview   # 本地预览构建
deno task lint      # Deno 内置 lint
deno task fmt       # Deno 内置 fmt
```

环境变量示例：

```bash
VITE_API_BASE=http://localhost:8000/api deno task dev
```

- 如需代理 `/api`，设置 `VITE_API_PROXY_TARGET=http://localhost:8000`，Vite dev server 会代理到该目标。

## 说明

- `deno.jsonc` 已设置 `nodeModulesDir: false`，不会生成或使用 `node_modules`。
- 若需锁定依赖版本，可运行 `deno lock` 生成 `deno.lock` 并提交。
