"""Flask app for a memory-driven relationship coach.

This file intentionally keeps data access, AI orchestration, and page routes in one
place for easy local running, but it is organized by section to stay readable.
"""

import os
import re
import sqlite3
import time
import json
from datetime import datetime
from html import escape, unescape
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

from dotenv import load_dotenv
from flask import Flask, Response, redirect, render_template, request, stream_with_context, url_for
from markupsafe import Markup

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data.db"
DEFAULT_PAGE_EVENTS_LIMIT = 30
DEFAULT_CONTEXT_EVENTS_LIMIT = 40
SUPPORTED_PROVIDERS = ("gemini", "openai")

PROFILE_FIELDS = (
    "your_name",
    "partner_name",
    "stage",
    "preferences",
    "boundaries",
    "goals",
)

AI_PROMPT_TEMPLATE = """
你是“微光心事LumiTrace”的资深关系教练。
目标：给出稳健、尊重边界、可立即执行的建议，避免空泛安慰。

你的建议必须基于用户提供的恋爱记忆上下文，优先引用具体事件（日期/标题/细节）。
如果上下文信息不足，请先指出缺口，再给低风险的默认策略。

输出格式:
1) 现状分析
2) 操作建议
3) 注意事项

写作要求:
- 每条尽量短句，避免大段抽象说教
- 尽量采用“原因 -> 动作 -> 预期结果”
- 关键结论用 **加粗**
- 用 [focus]...[/focus] 标出关键判断关键词
- 用 [action]...[/action] 标出可执行动作关键词
- 用 [risk]...[/risk] 标出风险提醒关键词
- 用 [script]...[/script] 标出可直接发送的话术关键词
- 标签里只能放 2-8 字词汇，禁止给整句加标签，每句最多 1-2 个标签
- 不要输出任何其他HTML
- 不要使用“重要程度/优先级高低”这类分级概念

恋爱记忆上下文:
{context}

用户问题:
{question}
""".strip()

DEFAULT_PROVIDER = str(os.getenv("LLM_PROVIDER") or "").strip() or "gemini"
DEFAULT_GEMINI_MODEL = str(os.getenv("GEMINI_MODEL") or "").strip() or "gemini-3-flash-preview"
DEFAULT_OPENAI_MODEL = str(os.getenv("OPENAI_MODEL") or "").strip() or "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = str(os.getenv("OPENAI_BASE_URL") or "").strip()
GEMINI_MODELS = tuple(
    dict.fromkeys(
        [DEFAULT_GEMINI_MODEL, "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"]
    )
)
OPENAI_MODELS = tuple(
    dict.fromkeys([DEFAULT_OPENAI_MODEL, "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"])
)
MODEL_CATALOG = {"gemini": GEMINI_MODELS, "openai": OPENAI_MODELS}

app = Flask(__name__)
load_dotenv()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_iso() -> str:
    return datetime.now().date().isoformat()


def clean(value: Any) -> str:
    # 把 None/空值统一成 ""，再去掉首尾空白，避免后续 .lower() / .get() 报错或出现脏输入
    return str(value or "").strip()


def normalize_events_for_timeline(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render timeline in chronological order (old -> new)."""
    # sorted(..., key=...) 会按 key 返回的元组排序：
    # 先比 event_date，再比 id；两者都是升序，所以结果是“从旧到新”的时间线
    # e.get("event_date", ""): 没日期时用空字符串兜底
    # int(e.get("id", 0)): 没 id 时用 0，且确保按数字而不是字符串比较
    return sorted(events, key=lambda e: (e.get("event_date", ""), int(e.get("id", 0))))


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create required tables if they do not exist yet."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            your_name TEXT,
            partner_name TEXT,
            stage TEXT,
            preferences TEXT,
            boundaries TEXT,
            goals TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT,
            mood TEXT,
            importance INTEGER DEFAULT 3,
            details TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_profile() -> Dict[str, Any]:
    conn = get_db()
    row = conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else {}


def upsert_profile(payload: Mapping[str, Any]) -> None:
    """Persist single-row profile (id=1)."""
    conn = get_db()
    conn.execute(
        """
        INSERT INTO profile (id, your_name, partner_name, stage, preferences, boundaries, goals, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          your_name = excluded.your_name,
          partner_name = excluded.partner_name,
          stage = excluded.stage,
          preferences = excluded.preferences,
          boundaries = excluded.boundaries,
          goals = excluded.goals,
          updated_at = excluded.updated_at
        """,
        (
            clean(payload.get("your_name")),
            clean(payload.get("partner_name")),
            clean(payload.get("stage")),
            clean(payload.get("preferences")),
            clean(payload.get("boundaries")),
            clean(payload.get("goals")),
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def add_event(payload: Mapping[str, Any]) -> int:
    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO events (event_date, title, category, mood, importance, details, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean(payload.get("event_date")),
            clean(payload.get("title")),
            clean(payload.get("category")),
            clean(payload.get("mood")),
            int(payload.get("importance", 3)),
            clean(payload.get("details")),
            now_iso(),
        ),
    )
    conn.commit()
    event_id = cur.lastrowid
    conn.close()
    return int(event_id)


def delete_event(event_id: int) -> None:
    conn = get_db()
    conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()


def list_events(limit: int = DEFAULT_PAGE_EVENTS_LIMIT) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT * FROM events
        ORDER BY event_date DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_ai_note(question: str, answer: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO ai_notes (question, answer, created_at) VALUES (?, ?, ?)",
        (question, answer, now_iso()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# AI-related helpers
# ---------------------------------------------------------------------------
def build_context_text(profile: Mapping[str, Any], events: List[Dict[str, Any]]) -> str:
    """Convert profile + recent events into a compact prompt context block."""
    profile_text = (
        f"你的名字: {profile.get('your_name', '未填写')}\n"
        f"对方名字: {profile.get('partner_name', '未填写')}\n"
        f"关系阶段: {profile.get('stage', '未填写')}\n"
        f"偏好: {profile.get('preferences', '未填写')}\n"
        f"边界/雷区: {profile.get('boundaries', '未填写')}\n"
        f"关系目标: {profile.get('goals', '未填写')}\n"
    )

    event_lines = []
    for event in events:
        event_lines.append(
            f"- [{event['event_date']}] {event['title']} | 分类:{event.get('category') or '未分类'} "
            f"| 情绪:{event.get('mood') or '未知'} | 细节:{event.get('details') or '无'}"
        )

    events_text = "\n".join(event_lines) if event_lines else "- 暂无事件记录"
    return f"{profile_text}\n最近恋爱情节:\n{events_text}"


def _is_retryable_error(message: str) -> bool:
    upper = message.upper()
    return "RESOURCE_EXHAUSTED" in upper or "RATE_LIMIT" in upper or "429" in upper


def ask_with_gemini(prompt: str, model: str) -> str:
    api_key = clean(os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        return (
            "尚未配置 GOOGLE_API_KEY。请在系统环境变量或 .env 中设置后重试。\n"
            "当前建议：先持续记录关键事件（时间、情绪、对话原话、结果），"
            "再让AI给出更个性化策略。"
        )
    if genai is None:
        return "未安装 google genai SDK。请执行: pip install google-genai"

    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = getattr(response, "text", None)
            return text.strip() if text else "模型返回为空，请稍后再试。"
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            if _is_retryable_error(message):
                time.sleep(1.5 * (attempt + 1))
                continue
            return f"AI调用失败: {exc}"
    return (
        "AI调用失败: 资源配额不足（429 RESOURCE_EXHAUSTED）。\n"
        "建议：\n"
        "1) 确认使用可用的模型（设置 GEMINI_MODEL）。\n"
        "2) 检查项目配额与计费，或切换到付费配额。\n"
        "3) 稍后重试。"
    )


def _extract_openai_text(content: Any) -> str:
    # Compatible with providers returning string or multi-part content blocks.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


def ask_with_openai(prompt: str, model: str, base_url: str) -> str:
    api_key = clean(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return "尚未配置 OPENAI_API_KEY。请在系统环境变量或 .env 中设置后重试。"
    if OpenAI is None:
        return "未安装 openai SDK。请执行: pip install openai"

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = (response.choices or [None])[0]
            message = getattr(choice, "message", None)
            text = _extract_openai_text(getattr(message, "content", ""))
            return text.strip() if text else "模型返回为空，请稍后再试。"
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            if _is_retryable_error(message):
                time.sleep(1.5 * (attempt + 1))
                continue
            return f"AI调用失败: {exc}"
    return "AI调用失败: 请求频率或配额受限（429）。请稍后重试。"


def ask_love_coach(question: str, context: str, provider: str, model: str, base_url: str) -> str:
    """Route request to selected provider (gemini/openai-compatible)."""
    # provider 统一清洗+小写；如果为空就默认 gemini
    selected_provider = clean(provider).lower() or "gemini"
    if selected_provider not in SUPPORTED_PROVIDERS:
        return f"不支持的 provider: {selected_provider}"

    prompt = AI_PROMPT_TEMPLATE.format(context=context, question=question)
    # 用户没填 model 时，根据 provider 选择对应默认模型
    selected_model = clean(model) or (
        DEFAULT_GEMINI_MODEL if selected_provider == "gemini" else DEFAULT_OPENAI_MODEL
    )

    if selected_provider == "gemini":
        return ask_with_gemini(prompt, selected_model)
    return ask_with_openai(prompt, selected_model, clean(base_url) or DEFAULT_OPENAI_BASE_URL)


def _extract_openai_delta(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return ""


def stream_with_gemini(prompt: str, model: str) -> Iterator[str]:
    api_key = clean(os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        yield "尚未配置 GOOGLE_API_KEY。请在系统环境变量或 .env 中设置后重试。"
        return
    if genai is None:
        yield "未安装 google genai SDK。请执行: pip install google-genai"
        return

    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            any_chunk = False
            for chunk in client.models.generate_content_stream(model=model, contents=prompt):
                text = clean(getattr(chunk, "text", ""))
                if text:
                    any_chunk = True
                    yield text
            if not any_chunk:
                yield "模型返回为空，请稍后再试。"
            return
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            if _is_retryable_error(message):
                time.sleep(1.5 * (attempt + 1))
                continue
            yield f"AI调用失败: {exc}"
            return
    yield "AI调用失败: 请求频率或配额受限（429）。请稍后重试。"


def stream_with_openai(prompt: str, model: str, base_url: str) -> Iterator[str]:
    api_key = clean(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        yield "尚未配置 OPENAI_API_KEY。请在系统环境变量或 .env 中设置后重试。"
        return
    if OpenAI is None:
        yield "未安装 openai SDK。请执行: pip install openai"
        return

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            any_chunk = False
            for event in stream:
                choice = (event.choices or [None])[0]
                delta = getattr(choice, "delta", None)
                text = _extract_openai_delta(getattr(delta, "content", ""))
                if text:
                    any_chunk = True
                    yield text
            if not any_chunk:
                yield "模型返回为空，请稍后再试。"
            return
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            if _is_retryable_error(message):
                time.sleep(1.5 * (attempt + 1))
                continue
            yield f"AI调用失败: {exc}"
            return
    yield "AI调用失败: 请求频率或配额受限（429）。请稍后重试。"


def stream_love_coach(question: str, context: str, provider: str, model: str, base_url: str) -> Iterator[str]:
    selected_provider = clean(provider).lower() or "gemini"
    if selected_provider not in SUPPORTED_PROVIDERS:
        yield f"不支持的 provider: {selected_provider}"
        return

    selected_model = clean(model) or (
        DEFAULT_GEMINI_MODEL if selected_provider == "gemini" else DEFAULT_OPENAI_MODEL
    )
    prompt = AI_PROMPT_TEMPLATE.format(context=context, question=question)
    if selected_provider == "gemini":
        yield from stream_with_gemini(prompt, selected_model)
        return
    yield from stream_with_openai(prompt, selected_model, clean(base_url) or DEFAULT_OPENAI_BASE_URL)


def render_answer_html(raw: str) -> str:
    """Convert safe subset of markup from model text into styled HTML spans."""
    def _pick_keyword(content: str) -> str:
        text = re.sub(r"<[^>]+>", "", unescape(clean(content)))
        text = re.sub(r"^[\-*0-9\.\)\(]+", "", text)
        text = text.strip("[]【】()（）\"'“”‘’ ")
        parts = [p for p in re.split(r"[，。；：、,.\s!！？?/\\|]+", text) if p]
        keyword = parts[0] if parts else text
        return keyword[:8]

    def _render_tag_keyword(safe_text: str, tag_name: str, css_class: str) -> str:
        pattern = rf"\[{tag_name}\]([\s\S]+?)\[/{tag_name}\]"

        def _repl(match: re.Match[str]) -> str:
            keyword = _pick_keyword(match.group(1))
            if not keyword:
                return ""
            return f'<span class="{css_class}">{keyword}</span>'

        return re.sub(pattern, _repl, safe_text, flags=re.I)

    safe = escape(raw or "")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = _render_tag_keyword(safe, "focus", "tag-focus")
    safe = _render_tag_keyword(safe, "action", "tag-action")
    safe = _render_tag_keyword(safe, "risk", "tag-risk")
    safe = _render_tag_keyword(safe, "script", "tag-script")
    # Backward compatibility for older model outputs.
    safe = _render_tag_keyword(safe, "high", "tag-focus")
    safe = _render_tag_keyword(safe, "mid", "tag-action")
    safe = _render_tag_keyword(safe, "low", "tag-risk")
    return safe.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# Page rendering helpers
# ---------------------------------------------------------------------------
def render_home_page(
    *,
    answer_html: str = "",
    profile_hint: str = "",
    timeline_hint: str = "",
    coach_hint: str = "",
    error_text: str = "",
    selected_provider: str = DEFAULT_PROVIDER,
    selected_model: str = "",
    selected_base_url: str = DEFAULT_OPENAI_BASE_URL,
    scroll_target: str = "",
):
    """Single place for homepage context assembly to keep routes short."""
    # 所有首页展示需要的数据，都在这里一次性准备好，路由函数只负责分支判断
    profile = get_profile()
    events = normalize_events_for_timeline(list_events(limit=DEFAULT_PAGE_EVENTS_LIMIT))
    # 规范化 provider：空值用默认值；非法值回退到 gemini，保证模板渲染稳定
    normalized_provider = clean(selected_provider).lower() or DEFAULT_PROVIDER
    if normalized_provider not in SUPPORTED_PROVIDERS:
        normalized_provider = "gemini"
    # model 为空时跟随 provider 给默认值，避免提交空模型名
    default_model = DEFAULT_GEMINI_MODEL if normalized_provider == "gemini" else DEFAULT_OPENAI_MODEL
    normalized_model = clean(selected_model) or default_model
    return render_template(
        "index.html",
        profile=profile,
        events=events,
        answer_html=Markup(answer_html) if answer_html else "",
        profile_hint=profile_hint,
        timeline_hint=timeline_hint,
        coach_hint=coach_hint,
        error_text=error_text,
        today=today_iso(),
        selected_provider=normalized_provider,
        selected_model=normalized_model,
        selected_base_url=clean(selected_base_url),
        provider_options=SUPPORTED_PROVIDERS,
        model_catalog_json=json.dumps(MODEL_CATALOG, ensure_ascii=False),
        scroll_target=clean(scroll_target),
    )


def required_fields_error(form_data: Mapping[str, Any], fields: List[str]) -> str:
    for field in fields:
        if not clean(form_data.get(field)):
            return f"{field} 不能为空"
    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_home_page()


@app.route("/profile/save", methods=["POST"])
def profile_save_page():
    upsert_profile(request.form)
    return redirect(url_for("index", profile_hint="档案已保存"))


@app.route("/events/add", methods=["POST"])
def events_add_page():
    error_text = required_fields_error(request.form, ["event_date", "title"])
    if error_text:
        return redirect(url_for("index", timeline_hint="保存失败", error_text=error_text))

    add_event(request.form)
    return redirect(url_for("index", timeline_hint="已加入时间轴"))


@app.route("/events/delete/<int:event_id>", methods=["POST"])
def events_delete_page(event_id: int):
    delete_event(event_id)
    return redirect(url_for("index", timeline_hint="已删除"))


@app.route("/coach/ask", methods=["POST"])
def coach_page():
    question = clean(request.form.get("question"))
    provider = clean(request.form.get("provider")).lower() or DEFAULT_PROVIDER
    model = clean(request.form.get("model"))
    raw_base_url = clean(request.form.get("base_url"))
    base_url = (raw_base_url or DEFAULT_OPENAI_BASE_URL) if provider == "openai" else ""

    if provider not in SUPPORTED_PROVIDERS:
        return render_home_page(
            coach_hint="生成失败",
            error_text=f"不支持的 provider: {provider}",
            selected_provider=provider,
            selected_model=model,
            selected_base_url=base_url,
            scroll_target="coach",
        )

    if not question:
        return render_home_page(
            coach_hint="生成失败",
            error_text="question 不能为空",
            selected_provider=provider,
            selected_model=model,
            selected_base_url=base_url,
            scroll_target="coach",
        )

    profile = get_profile()
    events = list_events(limit=DEFAULT_CONTEXT_EVENTS_LIMIT)
    context = build_context_text(profile, events)
    answer = ask_love_coach(question, context, provider=provider, model=model, base_url=base_url)
    save_ai_note(question, answer)

    return render_home_page(
        answer_html=render_answer_html(answer),
        coach_hint="已生成",
        selected_provider=provider,
        selected_model=model,
        selected_base_url=base_url,
        scroll_target="coach",
    )


@app.route("/coach/ask/stream", methods=["POST"])
def coach_stream_page():
    question = clean(request.form.get("question"))
    provider = clean(request.form.get("provider")).lower() or DEFAULT_PROVIDER
    model = clean(request.form.get("model"))
    raw_base_url = clean(request.form.get("base_url"))
    base_url = (raw_base_url or DEFAULT_OPENAI_BASE_URL) if provider == "openai" else ""

    if provider not in SUPPORTED_PROVIDERS:
        return Response(f"不支持的 provider: {provider}", status=400, mimetype="text/plain")
    if not question:
        return Response("question 不能为空", status=400, mimetype="text/plain")

    profile = get_profile()
    events = list_events(limit=DEFAULT_CONTEXT_EVENTS_LIMIT)
    context = build_context_text(profile, events)

    def _generate() -> Iterator[str]:
        full_parts: List[str] = []
        for chunk in stream_love_coach(question, context, provider=provider, model=model, base_url=base_url):
            full_parts.append(chunk)
            yield chunk
        save_ai_note(question, "".join(full_parts).strip())

    return Response(
        stream_with_context(_generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
