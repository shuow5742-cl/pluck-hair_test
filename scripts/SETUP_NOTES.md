# 部署踩坑记录（SETUP_NOTES）

## 1) SSH 免密登录（部署前必须）
先在本机配置到目标机器的 SSH 免密，否则脚本无法非交互执行远程命令。

```bash
ssh-copy-id user@host
```

## 2) 固定 IP（建议）
目标机器建议用 netplan 配置静态 IP，避免 DHCP 变更导致后续连不上。

## 3) GitHub 凭证持久化（关键坑）

- 现象：`backend_dev.sh deploy` 通过非交互式 SSH 执行 `git fetch` 时报错：
  `fatal: could not read Username for 'https://github.com': No such device or address`
- 原因：非交互式 shell 无 tty，HTTPS 认证无法弹出用户名/密码输入。
- 解决：在目标机器手动执行一次：

```bash
git config --global credential.helper store
git fetch origin
```

首次会要求输入 GitHub 用户名和 Personal Access Token（不是密码）。
凭证会保存到 `~/.git-credentials`，后续非交互环境可自动认证。

- 私有仓库注意：必须使用 fine-grained personal access token（GitHub Settings > Developer settings > Personal access tokens）。

## 4) 大恒相机 SDK 容器挂载（关键坑）

容器是隔离环境，宿主机上装的 SDK 库文件容器里看不到。需要挂载以下文件：

- `/usr/lib/libgxiapi.so` — 主 API 库
- `/usr/lib/liblog4cplus_gx.so` — 日志依赖（libgxiapi.so 依赖它）
- `/usr/lib/GxU3VTL.cti` — USB3 Transport Layer
- `/usr/lib/GxGVTL.cti` — GigE Transport Layer
- `/etc/Galaxy/` — SDK 配置目录

这些已经在 `docker-compose.runtime.yml` 里配好了。但前提是 phase2 脚本把 cti 文件拷贝到了 `/usr/lib/`。
如果是手动装的 SDK，cti 文件可能在 SDK 安装目录下（如 `Galaxy_camera/lib/x86_64/`），需要手动拷贝：

```bash
sudo cp /path/to/Galaxy_camera/lib/x86_64/*.cti /usr/lib/
sudo ldconfig
```

## 5) 项目路径

建议将项目代码放在 `/opt/pluck-hair`，而不是桌面等非标准路径。

```bash
sudo mkdir -p /opt/pluck-hair
sudo chown $(whoami):$(whoami) /opt/pluck-hair
git clone https://github.com/Einstellung/pluck-hair.git /opt/pluck-hair
```

## 6) 推荐部署顺序

1. Phase 1：`sudo bash scripts/bootstrap_host_phase1.sh`（Docker + NVIDIA 驱动）
2. 如有提示，重启机器（reboot）
3. Phase 2：`sudo bash scripts/bootstrap_host_phase2.sh`（NVIDIA Container Toolkit + 大恒相机 SDK + cti 拷贝）
4. 配好 SSH 免密 + GitHub 凭证持久化
5. 克隆代码到 `/opt/pluck-hair`
6. Phase 3：`bash scripts/backend_dev.sh deploy test`（拉代码 + 构建 + 启动）
