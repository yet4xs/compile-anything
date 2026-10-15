# SSH 凭据泄露 — 处置记录(Phase 9B-2.3 P0)

**日期**: 2026-10-16 | **级别**: P0 | **本文件不含任何实际凭据**

## 事实

- `scripts/ssh_fetch.py` 与 `scripts/ssh_push_run.py` 曾以明文硬编码
  服务器地址、端口、root 用户名与密码,并使用
  `paramiko.AutoAddPolicy()` 无条件信任未知主机密钥。
- 首次进入 Git 历史: 5818b1b(2026-10-03, Phase 5C);端口更新后再次
  进入: ccce655(2026-10-11, Phase 7C-0.1)。
- 仓库为公开 GitHub 仓库(origin/main)。截至处置时,自 5818b1b 起
  共 94 个提交携带该凭据,公开暴露约 13 天。

## 已执行的处置(仓库侧)

1. 当前树中所有硬编码凭据已删除(`git grep` 复核为零匹配)。
2. 新增 `scripts/ssh_config.py`: 连接参数全部来自环境变量
   (`BITAHUB_HOST/PORT/USER/PW/KEY_FILE/KNOWN_HOSTS`),支持密钥认证。
3. 禁用 AutoAddPolicy,改为**首次 pinned 的主机指纹策略**
   (`.ssh_known_fingerprints` 本地缓存,已加入 .gitignore;
   指纹变化即拒绝连接)。
4. 两个 helper 脚本改为从 ssh_config 导入 connect()。

## 所有人决定(2026-10-16, 服务器所有者)

- 服务器侧凭据轮换与登录审计由所有者自行决定执行与否
  (服务器不需要时保持关机,已缩小暴露面)。
- **不执行 Git 历史改写**。如后续决定清理: 需 `git filter-repo` 或
  BFG 替换历史中的密码串 + 强推 origin/main + 通知所有克隆方,
  且必须以**先轮换凭据**为前提——历史清理不能使已曝光密码重新安全。

## 后续约束

- 任何新代码禁止写入凭据;提交前 `git diff --cached` 检查。
- GPU/服务器端任务在安全处置(所有者决定)完成前保持 NO-GO,
  与 Phase 9B-2.2 的研究性 NO-GO 相互独立。
