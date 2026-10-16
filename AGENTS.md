# Agent 接手入口

开始工作先阅读 [NEXT_STEPS.md](NEXT_STEPS.md)，再查看 `git status --short`。

- 当前实现、已验证内容和执行顺序：`NEXT_STEPS.md`。
- 实际实验数值及证据限制：`EXPERIMENT_RESULTS.md`。
- 当前可运行命令与统一实验协议：`TRAINING_COMMANDS.md`。
- 文献依据、尚未实现的候选方案：`ML_METHOD_SURVEY.md`。

当前只维护 A0 / attention_fix / A1 和 r1_event / r2_balanced / r3_term。用户已要求移除旧路线；除非用户提出新要求，不恢复废弃模块、参数或文档。

优先完成已实现方案的服务器验证，不重复实现 TERM。保留工作区未提交修改；迁移时包括新文件和删除项。环境、数据路径和 GPU 可用性需重新核验。

探索使用验证集；固定 50 轮协议需显式关闭早停及自动测试。CPU 检查不代表 CUDA 训练或精度收益。每次完成工作，将新证据、已改文件及下一步更新到 NEXT_STEPS.md；避免新增重复交接文档。
