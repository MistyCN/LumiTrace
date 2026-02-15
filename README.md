# 微光心事LumiTrace（Flask + Python + Gemini/OpenAI-Compatible）

这个项目是一个“可记忆恋爱情节”的 AI 辅助工具：
- 结构化保存关系档案（偏好、边界、目标）
- 按时间线记录恋爱事件（情绪、重要度、细节）
- 基于历史上下文向 AI 提问，获得更具体的行动建议
- 支持多模型切换：`gemini` 与任意 OpenAI 兼容模型

## 1. 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 配置环境变量

复制 `.env.example` 到 `.env`（或直接设置系统环境变量）。

Gemini：
```bash
GOOGLE_API_KEY=你的key
GEMINI_MODEL=gemini-2.0-flash
```

OpenAI 兼容：
```bash
OPENAI_API_KEY= 你的key
OPENAI_MODEL= 模型名
OPENAI_BASE_URL= 模型URL
```

默认 provider（可选）：
```bash
LLM_PROVIDER=gemini
```
可选值：`gemini` / `openai`。

## 3. 运行

```bash
python app.py
```

打开 `http://127.0.0.1:5000`

## 4. 项目结构

```text
app.py
templates/index.html
static/style.css
requirements.txt
data.db  # 启动后自动生成
example_data.db #包含测试用例
```

## 5. 多模型切换说明

在页面“向军师提问”区域可直接设置：
- `provider`：`gemini` 或 `openai`
- `model`：模型名
- `base_url`：仅 OpenAI 兼容接口需要，官方 OpenAI 可留默认

## 6.测试用例
- 将example_data.db改名为data.db即可使用

```bash
███╗   ███╗██╗███████╗████████╗██╗   ██╗
████╗ ████║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██╔████╔██║██║███████╗   ██║    ╚████╔╝ 
██║╚██╔╝██║██║╚════██║   ██║     ╚██╔╝  
██║ ╚═╝ ██║██║███████║   ██║      ██║   
╚═╝     ╚═╝╚═╝╚══════╝   ╚═╝      ╚═╝ 
```  
QQ:2104146829

微信:chxmistcn

有问题可随时联系，有功能建议或bug反馈请使用github issues
