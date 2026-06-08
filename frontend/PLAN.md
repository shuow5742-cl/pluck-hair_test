# Pluck Robot Frontend Implementation Plan

## 技术栈与基础
- 框架：Next.js 14.2.x（App Router）+ TypeScript，包管理使用 pnpm。
- 样式：TailwindCSS + shadcn/ui（定制主题），图标：Lucide React。
- 状态：Zustand 负责全局控制/状态/设置；数据获取与轮询：SWR。
- 字体与排版：`Noto Sans SC, "PingFang SC", "Microsoft YaHei", sans-serif`；数字使用千分位格式化。
- 设计风格：深色工业风；背景 #0f172a/#111827，卡片 #1f2937，描边 #243047，绿 #16a34a，红 #dc2626，文字 #e5e7eb，圆角 12-14px，低阴影。

## 信息架构
- 页面
  - `/` 主控面板：左视频 + 右控制/统计/日志，16:9 优化，窄屏右侧可纵向堆叠或折叠。
  - `/settings` 设置页：摄像头、算法、系统参数表单，本地保存。
- 全局布局：Topbar/标题区可复用；主题变量集中在 `globals.css` + shadcn 主题配置。

## 数据与状态流
- Store（Zustand `useRobotStore`）
  - `status`: `running | stopped`
  - `stats`: `{ fps, totalImpurities, currentTargets, durationSec, confidence }`
  - `settings`: `{ camera: { exposureUs, gainDb, whiteBalanceMode }, detection: { confidenceThreshold, minSizePx, maxSizePx }, system: { logLevel, imageSavePath } }`
  - `logs`: 最近 N 条 `{ timestamp, level, message }`
  - actions: `startAlgo`, `stopAlgo`, `setSettings`, `pushLog`, `setStatus`, `setStats`
- 数据获取（SWR 轮询）
  - `/api/control/status` 每 2s 获取运行状态；POST 切换状态并回写 store。
  - `/api/stream/stats` 每 1s 获取统计；轻微随机波动。
  - 轮询失败在 toast+日志提示。
- 设置持久化
  - 客户端加载后从 `localStorage` 读取/写入 `settings`；SSR 时不访问存储。

## Mock API 设计
- `/api/stream/video`
  - 返回 `multipart/x-mixed-replace` 模拟 MJPEG；若实现复杂，首版可返回单张静态图，前端 overlay 叠加假框。
  - 可轮询几张占位图或生成随机噪声。
- `/api/control/status`
  - GET: `{ running: boolean }`
  - POST: 切换状态并返回最新 `{ running }`
- `/api/stream/stats`
  - `{ fps, totalImpurities, currentTargets, durationSec, confidence }`，totalImpurities 单调递增，小幅随机浮动其它值。

## 组件与布局
- `app/page.tsx`：主控布局，左视频右侧面板。
- `components/VideoFeed.tsx`：`<img src="/api/stream/video">`；前端用绝对定位 overlay 假框/标签；加载失败显示占位。
- `components/StatusIndicator.tsx`：绿/红 pill，显示“AI算法正在运行中.../已停止”。
- `components/StatCard.tsx`：用于“累计挑出杂质”“当前视野目标”，数字大号、千分位、单位“个”。
- `components/ControlButtons.tsx`：开始/停止按钮，含图标、loading、禁用逻辑，点击更新状态+日志。
- `components/LogPanel.tsx`：滚动区域，时间戳 + 文本，自动滚动到底，最多 100 条。
- `components/FpsFooter.tsx`：左下角显示 FPS 文案。
- `app/settings/page.tsx`：表单分组
  - 摄像头：曝光（μs）、增益（dB）、白平衡（自动/手动）。
  - 算法：置信度阈值 0-1，最小/最大目标尺寸（px）。
  - 系统：日志级别（Info/Debug）、图片保存路径。
  - 操作：保存（写 store + localStorage）、重置为默认。

## 交互与反馈
- Start/Stop：点击后立即更新本地状态；成功 toast；失败回滚状态并提示。
- 统计区：SWR 轮询更新；异常时在日志中追加“统计拉取失败”。
- 视频：加载中 skeleton，失败可重试按钮。
- 日志：新事件自动滚动；显示运行/停止/设置保存/请求失败等关键事件。

## 样式规范与细节
- 间距：页面 padding 16-20px；右侧列宽 ~360px；卡片内 padding 16px。
- 边框/分割：细描边 #243047；卡片阴影低（如 `shadow-lg/10`）。
- 字号：标题 18-20px，卡片标题 14px，数字 32-40px，辅助 12px。
- 圆角：主容器和卡片 12-14px；按钮 10px。
- Overlay 边框：红色 2px，标签深色背景 + 白字，轻微圆角。

## 可访问性与容错
- 按钮和状态有清晰颜色/文字双重反馈。
- 表单带基础校验（范围/必填），错误提示靠近控件。
- 轮询失败退避（如指数退避上限 10s），避免刷日志。

## 验证清单
- 打开 `/`：布局在 1920x1080、1366x768 下正常；窄屏右侧堆叠/折叠不溢出。
- Start/Stop：状态、指示灯、按钮禁用、日志同步更新；接口失败有提示。
- Stats：数字更新、千分位、单位正确；FPS 底部展示同步刷新。
- 视频：能看到流/占位图，overlay 可见；失败时提示。
- Settings：修改保存后刷新仍保留；无效输入被阻止；日志记录“设置已保存”。
- Dark 主题一致：背景/卡片/描边/按钮颜色符合规范，字体为中文友好字体。
