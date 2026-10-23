# 可选的预训练文件

默认没有下载模型。① 微型随机教学模型与 ② 浏览器真实训练不依赖本目录。

在项目根目录执行：

```bash
python -m pip install torch transformers huggingface_hub
python scripts/export_weights.py --model sshleifer/tiny-gpt2 --out public/models
```

导出器提供 `config.json`、`weights.json`、可整体导入的 `model.json` 和独立参考结果 `reference.json`。
前端只从同源 `models/` 读取文件；缺失、格式错误、维度不匹配、非有限数字与超时均会显示错误并保留当前模型。

当前导出器精确支持 **GPT-2 架构的小型检查点**。示例是非常小的公开测试检查点，不代表高质量语言能力。Pythia / Qwen / GPT-Neo 的架构不同，不能直接导入。
为避免逐数检查耗尽浏览器内存，当前上限为 d ≤ 128、FFN ≤ 512、12 层、上下文 ≤ 128、词表 ≤ 65,536、存储参数 ≤ 16,000,000。大矩阵分页显示。
`sshleifer/tiny-gpt2` 每头只有一维，默认使用其原始 learned position；切到 RoPE 时没有可旋转的维度对，旋转退化为恒等变换。

权重与词表默认不提交 Git。请在下载前确认原始模型仓库的许可，按许可分发。
