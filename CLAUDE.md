# Project Notes

## Python 环境

后端使用 `uv` 管理虚拟环境。运行 Python 命令（测试、import 验证等）时，必须通过 `uv run` 在 `backend/` 目录下执行：

```bash
cd backend && uv run python -c "..."
cd backend && uv run python -m pytest tests/ -v
```
