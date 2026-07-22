"""
Odoo 18 SFT Dataset Generator — forge_v3.py  "Forge Elite"
==============================================================
Generates ChatML-format SFT training data from real Odoo 18 modules.

Tiers:
  MICRO  — individual methods, field blocks, XML records, JS components
  MESO   — 2-4 related files grouped by Odoo model graph (features)
  MACRO  — entire module (full module creation brief + all files)
  DOC_GROUNDED - high-quality README grounded generation
"""

import ast
import asyncio
import argparse
import hashlib
import json
import logging
import os
import re
import signal
import time
import urllib.request
import base64
from dataclasses import dataclass
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import urllib.error

import aiofiles
import aiosqlite
import ollama
import tiktoken
from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

class Settings(BaseSettings):
    source_directory: Path = PydanticField(default=Path("./resource"))
    output_dataset_file: Path = PydanticField(default=Path("./odoo18_sft_v3.jsonl"))
    db_file: Path = PydanticField(default=Path("./forge_v3_state.db"))

    generator_model: str = PydanticField(default="qwen2.5-coder:7b-instruct")
    ollama_host: str = PydanticField(default="http://localhost:11434")

    max_concurrent_micro: int = PydanticField(default=1)
    max_concurrent_meso: int = PydanticField(default=1)
    max_concurrent_macro: int = PydanticField(default=1)
    max_concurrent_doc_grounded: int = PydanticField(default=1)

    micro_max_tokens: int = PydanticField(default=3000)
    meso_max_tokens: int = PydanticField(default=8000)
    macro_max_tokens: int = PydanticField(default=12000)

    temperature_micro: float = PydanticField(default=0.75)
    temperature_meso: float = PydanticField(default=0.65)
    temperature_macro: float = PydanticField(default=0.55)
    temperature_doc_grounded: float = PydanticField(default=0.4)

    enable_vision: bool = PydanticField(default=False)
    vision_api_url: str = PydanticField(default="http://localhost:8000/v1/chat/completions")
    vision_model: str = PydanticField(default="qwen2.5-vl-7b-instruct")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

class TaskTier(str, Enum):
    MICRO = "MICRO"
    MESO = "MESO"
    MACRO = "MACRO"
    DOC_GROUNDED = "DOC_GROUNDED"

@dataclass
class ModuleMeta:
    name: str
    technical_name: str
    description: str
    depends: List[str]
    path: Path

@dataclass
class WorkItem:
    tier: TaskTier
    hash: str
    payload: str
    source_files: List[str]
    module_meta: Optional[ModuleMeta]
    extra_data: Optional[Dict] = None

_SYS_MICRO = """\
You are an expert who creates high-quality AI training data for an Odoo 18 coding assistant.

You will receive a small code snippet from an Odoo 18 module (a method, field block,
XML record, or JS component).

Your job:
1. Write a REALISTIC developer instruction — exactly how a developer would ask an AI
   assistant to write or modify this specific piece of code. Keep it natural and direct.
   Examples: "Add a computed field...", "Write an onchange for...", "I need a constraint that..."
   
   CRITICAL RULE FOR INSTRUCTION CONTEXT:
   If the snippet is a Python class that inherits an Odoo model (has `_inherit = ...`), your instruction MUST explicitly mention keywords like "in Python", "inherit the model", "backend model", or "Python class". 
   DO NOT write ambiguous instructions like "add a field for job_no" — instead write "inherit the sale.order model in Python and add a field for job_no". This teaches the AI to associate those keywords with Python `_inherit` blocks!

2. Write the IDEAL code response — the exact, clean code a senior Odoo developer would
   produce (use ```python / ```xml / ```javascript fenced blocks).

Return ONLY valid JSON (no surrounding markdown) with exactly two keys:
  "instruction": "<developer request — be specific, technical, concise>"
  "response": "<code only — properly fenced>"
"""

_SYS_MESO = """\
You are an expert who creates high-quality AI training data for an Odoo 18 coding assistant.

You will receive a cluster of 2–4 related files from an Odoo 18 module (e.g., a Python
model + its XML view + security CSV).

Your job:
1. Write a REALISTIC multi-file feature request — as a senior developer briefing an AI
   assistant. Describe the FEATURE to implement, not the files themselves.
   Example: "Extend the sale order with a custom approval workflow. Add the new state
   field, an Approve button in the form view, and update the access rules."
2. Write the IDEAL multi-file response. Use:
      ## FILE: <relative/path>
      ```lang
      <full file content>
      ```
   for EVERY file that needs to be created or changed.

Return ONLY valid JSON (no surrounding markdown) with exactly two keys:
  "instruction": "<multi-file feature request>"
  "response": "OK"
"""

_SYS_MACRO = """\
You are an expert who creates high-quality AI training data for an Odoo 18 coding assistant.

You will receive the COMPLETE source code of an Odoo 18 module.

Your job:
1. Write a REALISTIC module creation brief — as a developer asking an AI to build this
   entire module from scratch. Be specific: mention the domain, the models, the UI
   expectations, and key business logic. Do NOT mention filenames — describe behaviour.
2. Write the IDEAL complete module response. Use:
      ## FILE: <relative/path>
      ```lang
      <full file content>
      ```
   for EVERY file (__manifest__.py, models, views, security, data, etc.).

Return ONLY valid JSON (no surrounding markdown) with exactly two keys:
  "instruction": "<complete module brief — requirement-focused, not file-focused>"
  "response": "OK"
"""

_SYS_DOC_GROUNDED = """\
You are an expert who creates high-quality AI training data for an Odoo 18 coding assistant.

You will receive documentation (README) extracted from a real Odoo 18 module, and possibly visual descriptions of its UI.

Your job is to write a REALISTIC module creation brief — as a developer asking an AI to build this module from scratch. Base your request ENTIRELY on the provided documentation.
- Describe the functional requirements, configuration steps, and UI behaviors listed.
- Make it sound like a natural, detailed requirement document from a client/developer.
- DO NOT mention files or technical filenames. Focus on the business logic and UI.

Return ONLY valid JSON (no surrounding markdown) with exactly two keys:
  "instruction": "<the developer's requirement prompt>"
  "response": "OK"
"""

_ASSISTANT_SYSTEM = (
    "You are an expert Odoo 18 developer. You write clean, idiomatic, production-ready "
    "Odoo 18 code following community best practices. "
    "For single changes, provide the relevant code block. "
    "For multi-file changes, prefix each file with ## FILE: <relative/path> followed by "
    "a fenced code block."
)

class QualityFilter:
    MIN_TOKENS = {TaskTier.MICRO: 45, TaskTier.MESO: 160, TaskTier.MACRO: 450}

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    def token_count(self, text: str) -> int:
        return len(self._tok.encode(text))

    @staticmethod
    def is_trivial_python(code: str) -> bool:
        meaningful = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        non_trivial = [ln for ln in meaningful if ln not in ("pass", "...", "return", "return None", "super()")]
        return len(non_trivial) <= 2

    def passes(self, tier: TaskTier, payload: str) -> bool:
        return self.token_count(payload) >= self.MIN_TOKENS.get(tier, 0)

    def within_budget(self, tier: TaskTier, settings: Settings, payload: str) -> bool:
        budgets = {
            TaskTier.MICRO: settings.micro_max_tokens,
            TaskTier.MESO:  settings.meso_max_tokens,
            TaskTier.MACRO: settings.macro_max_tokens,
        }
        if tier not in budgets: return True
        return self.token_count(payload) <= budgets[tier]

_SKIP_DIRS = frozenset({"i18n", "node_modules", "__pycache__", ".git", "static/lib", "tests"})
_SKIP_FILE_PATTERNS = ("__manifest__", "__init__", "test_", "_test")
_EXT_TO_LANG = {".py": "python", ".xml": "xml", ".js": "javascript", ".css": "css", ".scss": "scss", ".csv": "csv"}

class OdooModuleParser:
    def __init__(self, module_path: Path, qf: QualityFilter, settings: Settings) -> None:
        self.path = module_path
        self.qf = qf
        self.settings = settings
        self.meta: Optional[ModuleMeta] = None
        self.registry: Dict[str, str] = {}
        self.model_graph: Dict[str, Set[str]] = {}

    def parse_manifest(self) -> Optional[ModuleMeta]:
        mf = self.path / "__manifest__.py"
        if not mf.exists(): return None
        try:
            with open(mf, "r", encoding="utf-8", errors="ignore") as fh:
                data = eval(fh.read(), {"__builtins__": {}})
            self.meta = ModuleMeta(
                name=data.get("name", self.path.name),
                technical_name=self.path.name,
                description=data.get("summary", data.get("description", data.get("name", ""))),
                depends=data.get("depends", []),
                path=self.path,
            )
            return self.meta
        except Exception:
            return None

    def index_workspace(self) -> None:
        for root, dirs, files in os.walk(self.path):
            rp = Path(root)
            dirs[:] = sorted([d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")])
            for fname in sorted(files):
                if not fname.endswith((".py", ".xml", ".js", ".csv", ".scss", ".css")): continue
                if any(pat in fname for pat in _SKIP_FILE_PATTERNS): continue
                full = rp / fname
                rel = str(full.relative_to(self.path)).replace("\\", "/")
                try:
                    content = full.read_text(encoding="utf-8", errors="ignore").strip()
                    if content:
                        self.registry[rel] = content
                        self._update_model_graph(rel, content)
                except Exception: pass

    def extract_micro_tasks(self) -> List[Tuple[str, List[str]]]:
        results: List[Tuple[str, List[str]]] = []
        for rel, content in self.registry.items():
            if rel.endswith(".py"): results.extend(self._micro_python(rel, content))
            elif rel.endswith(".xml"): results.extend(self._micro_xml(rel, content))
            elif rel.endswith(".js"): results.extend(self._micro_js(rel, content))
        return results

    def extract_meso_clusters(self) -> List[Tuple[str, List[str]]]:
        results: List[Tuple[str, List[str]]] = []
        for model_name, files in self.model_graph.items():
            file_list = sorted(files)
            if len(file_list) < 2: continue
            mod_label = self.meta.technical_name if self.meta else self.path.name
            parts = [f"### MESO CLUSTER: model `{model_name}` | module `{mod_label}` ###\n"]
            for fp in file_list: parts.append(f"## FILE: {fp}\n```{self._lang(fp)}\n{self.registry[fp]}\n```")
            payload = "\n\n".join(parts)
            if self.qf.passes(TaskTier.MESO, payload) and self.qf.within_budget(TaskTier.MESO, self.settings, payload):
                results.append((payload, file_list))
        return results

    def extract_macro_bundle(self) -> Optional[Tuple[str, List[str]]]:
        if not self.meta: return None
        def sort_key(p: str) -> Tuple[int, str]:
            if "models" in p: return (1, p)
            if "views" in p: return (2, p)
            if "templates" in p: return (2, p)
            if "security" in p: return (3, p)
            if "wizard" in p: return (4, p)
            if "data" in p: return (5, p)
            if "static" in p: return (9, p)
            return (6, p)
        sorted_files = sorted(self.registry.keys(), key=sort_key)
        mod = self.meta
        header = f"### COMPLETE MODULE: `{mod.technical_name}` ###\n### Display Name: {mod.name} | Depends: {', '.join(mod.depends)} ###\n"
        mf_path = self.path / "__manifest__.py"
        mf_block = ""
        if mf_path.exists():
            mf_content = mf_path.read_text(encoding="utf-8", errors="ignore").strip()
            mf_block = f"## FILE: __manifest__.py\n```python\n{mf_content}\n```"
        parts = [header, mf_block] if mf_block else [header]
        for fp in sorted_files: parts.append(f"## FILE: {fp}\n```{self._lang(fp)}\n{self.registry[fp]}\n```")
        payload = "\n\n".join(parts)
        if not self.qf.passes(TaskTier.MACRO, payload): return None
        if not self.qf.within_budget(TaskTier.MACRO, self.settings, payload):
            trimmed = [header, mf_block] if mf_block else [header]
            used = self.qf.token_count("\n\n".join(trimmed))
            budget = self.settings.macro_max_tokens
            for fp in sorted_files:
                block = f"## FILE: {fp}\n```{self._lang(fp)}\n{self.registry[fp]}\n```"
                cost = self.qf.token_count(block)
                if used + cost > budget: break
                trimmed.append(block)
                used += cost
            payload = "\n\n".join(trimmed)
        return payload, list(self.registry.keys())

    def extract_doc_grounded_bundle(self) -> Optional[Tuple[str, List[str], Dict]]:
        macro = self.extract_macro_bundle()
        if not macro: return None
        macro_code, files = macro
        readme_path = self.path / "README.rst"
        if not readme_path.exists(): return None
        text = readme_path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r'^\.\. \|[^|]+\| image::.*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s+:(?:target|alt):.*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\.\. image::.*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\|badge\d+\|\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\.\.\s*$.*?(?=^[^\s])', '', text, flags=re.MULTILINE|re.DOTALL)
        text = re.sub(r'`([^`<]+)\s*<[^>]+>`[_]*', lambda m: m.group(1).strip(), text)
        text = text.replace('**', '').replace('__', '')
        text = re.sub(r'\n{3,}', '\n\n', text)
        title_m = re.search(r'^=+\n(.+?)\n=+', text, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else ''
        desc_m = re.search(r'^=+\n.+?\n=+\n\n(.+?)(?=\nTable of|\nConfiguration\n=|\nUsage\n=|\nInstallation\n=)', text, re.DOTALL)
        desc = desc_m.group(1).strip() if desc_m else ''
        def get_section(name):
            m = re.search(rf'^{name}\n=+\n(.+?)(?=\n[A-Z][^\n]*\n[=\-]+|\Z)', text, re.MULTILINE|re.DOTALL)
            return m.group(1).strip() if m else ''
        config = get_section('Configuration')
        usage = get_section('Usage')
        parts = []
        if title: parts.append(f"Title: {title}")
        if desc: parts.append(f"Description:\n{desc}")
        if config: parts.append(f"Configuration:\n{config}")
        if usage: parts.append(f"Usage:\n{usage}")
        cleaned = "\n\n".join(parts)
        if len(cleaned) < 100: return None
        images = []
        desc_dir = self.path / "static" / "description"
        if desc_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"):
                for img in desc_dir.glob(ext):
                    if "icon" not in img.name.lower() and img.stat().st_size > 10000: images.append(str(img))
        return cleaned, files, {"macro_code": macro_code, "images": images[:3]}


    def _update_model_graph(self, rel: str, content: str) -> None:
        if rel.endswith(".py"):
            for m in re.findall(r"_(?:name|inherit)\s*=\s*['\"]([^'\"]+)['\"]", content): self.model_graph.setdefault(m, set()).add(rel)
        elif rel.endswith(".xml"):
            for m in re.findall(r"<field name=[\"']model[\"']>([^<]+)</field>", content): self.model_graph.setdefault(m, set()).add(rel)

    @staticmethod
    def _lang(rel: str) -> str:
        for ext, lang in _EXT_TO_LANG.items():
            if rel.endswith(ext): return lang
        return "text"

    def _header(self, rel: str, extra: str = "") -> str:
        mod = self.meta.technical_name if self.meta else self.path.name
        return f"# module: {mod} | file: {rel}{f' | {extra}' if extra else ''}"

    def _micro_python(self, rel: str, content: str) -> List[Tuple[str, List[str]]]:
        results = []
        try:
            tree = ast.parse(content)
            lines = content.splitlines(keepends=True)
            func_to_class: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)): func_to_class.setdefault(child.name, node.name)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
                if (node.end_lineno - node.lineno) < 4: continue
                snippet = "".join(lines[node.lineno - 1 : node.end_lineno]).strip()
                if self.qf.is_trivial_python(snippet): continue
                cls = func_to_class.get(node.name, "")
                extra = f"class: {cls} | method: {node.name}" if cls else f"method: {node.name}"
                payload = f"{self._header(rel, extra)}\n\n```python\n{snippet}\n```"
                if self.qf.passes(TaskTier.MICRO, payload): results.append((payload, [rel]))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef): continue
                field_lines_acc: List[str] = []
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        raw = "".join(lines[item.lineno - 1 : item.end_lineno]).strip()
                        if "fields." in raw and len(raw) > 15: field_lines_acc.append(raw)
                if len(field_lines_acc) >= 2:
                    block = "\n".join(field_lines_acc[:8])
                    payload = f"{self._header(rel, f'class: {node.name} | field definitions')}\n\n```python\n{block}\n```"
                    if self.qf.passes(TaskTier.MICRO, payload): results.append((payload, [rel]))
        except Exception: pass
        return results

    def _micro_xml(self, rel: str, content: str) -> List[Tuple[str, List[str]]]:
        results = []
        for tag in ("record", "template", "menuitem", "act_window"):
            for m in re.finditer(rf"(<{tag}\b[^>]*>.*?</{tag}>)", content, re.DOTALL | re.IGNORECASE):
                snippet = m.group(1).strip()
                if len(snippet) < 60: continue
                payload = f"{self._header(rel, f'<{tag}>')}\n\n```xml\n{snippet}\n```"
                if self.qf.passes(TaskTier.MICRO, payload): results.append((payload, [rel]))
        return results

    def _micro_js(self, rel: str, content: str) -> List[Tuple[str, List[str]]]:
        results = []
        pattern = re.compile(r"((?:export\s+)?class\s+\w+|patch\s*\(|registry\.category\s*\()")
        idx = 0
        while idx < len(content):
            m = pattern.search(content, idx)
            if not m: break
            start = m.start()
            brace = content.find("{", start)
            if brace == -1:
                idx = m.end()
                continue
            depth, end = 0, -1
            for i in range(brace, min(brace + 20_000, len(content))):
                if content[i] == "{": depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                idx = m.end()
                continue
            snippet = content[start:end].strip()
            if snippet.count("\n") >= 3:
                payload = f"{self._header(rel, 'JS component')}\n\n```javascript\n{snippet}\n```"
                if self.qf.passes(TaskTier.MICRO, payload): results.append((payload, [rel]))
            idx = end
        return results

class _SFTRaw(BaseModel):
    instruction: str
    response: str

class InferenceEngine:
    _SYS_MAP = {TaskTier.MICRO: _SYS_MICRO, TaskTier.MESO: _SYS_MESO, TaskTier.MACRO: _SYS_MACRO, TaskTier.DOC_GROUNDED: _SYS_DOC_GROUNDED}
    _TEMP_ATTR = {TaskTier.MICRO: "temperature_micro", TaskTier.MESO: "temperature_meso", TaskTier.MACRO: "temperature_macro", TaskTier.DOC_GROUNDED: "temperature_doc_grounded"}
    _TIMEOUT = {TaskTier.MICRO: 240, TaskTier.MESO: 420, TaskTier.MACRO: 660, TaskTier.DOC_GROUNDED: 420}
    _USER_PREFIX = {
        TaskTier.MICRO: "Analyze this Odoo 18 code snippet and generate the training pair:\n\n",
        TaskTier.MESO:  "Analyze this cluster of related Odoo 18 files and generate the training pair:\n\n",
        TaskTier.MACRO: "Analyze this complete Odoo 18 module and generate the training pair:\n\n",
        TaskTier.DOC_GROUNDED: "Analyze this Odoo 18 module documentation and generate the developer requirement:\n\n",
    }

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.client = ollama.AsyncClient(host=settings.ollama_host)

    async def run_vision_precompute(self, item: WorkItem) -> Optional[str]:
        images = item.extra_data.get("images", []) if item.extra_data else []
        if not images or not self.settings.enable_vision: return None
        descriptions = []
        for path in images:
            try:
                def _req():
                    import json, urllib.request
                    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
                    url = "http://localhost:11434/api/generate"
                    data = {
                        "model": self.settings.vision_model,
                        "prompt": "Describe the key UI elements, fields, and functionality shown in this Odoo screenshot. Be concise but specific about what this module adds to the UI.",
                        "images": [b64],
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={"Content-Type": "application/json"}, method='POST')
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        return json.loads(resp.read().decode('utf-8')).get("response", "").strip()
                
                desc = await asyncio.to_thread(_req)
                if desc: 
                    self.logger.info(f"✅ Vision Processed: {os.path.basename(path)}")
                    descriptions.append(f"- Image {os.path.basename(path)}:\n  {desc}")
            except Exception as e:
                self.logger.warning(f"Vision error on {path}: {e}")
        if descriptions: return "\n\n### UI Screenshots Descriptions ###\n" + "\n\n".join(descriptions)
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
    async def generate(self, item: WorkItem, precomputed_vision: Optional[str] = None) -> Optional[Dict]:
        tier = item.tier
        sys_prompt = self._SYS_MAP[tier]
        temp = getattr(self.settings, self._TEMP_ATTR[tier])
        timeout = float(self._TIMEOUT[tier])
        
        extra_txt = ""
        if precomputed_vision:
            extra_txt += precomputed_vision
            
        user_msg = self._USER_PREFIX[tier] + item.payload + extra_txt
        
        try:
            raw_response = await asyncio.wait_for(
                self.client.chat(
                    model=self.settings.generator_model,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                    format=_SFTRaw.model_json_schema(),
                    options={"temperature": temp, "num_ctx": 16384, "top_p": 0.9},
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout [{tier.value}] hash={item.hash[:8]}")
            raise
            
        raw_text = raw_response.message.content
        raw_text = re.sub(r"```(?:json)?(.*?)```", r"\1", raw_text, flags=re.DOTALL).strip()
        try:
            record = _SFTRaw.model_validate_json(raw_text)
        except Exception as e:
            self.logger.warning(f"JSON parse failed [{tier.value}]: {e}")
            raise ValueError("Bad JSON") from e

        if tier in (TaskTier.DOC_GROUNDED, TaskTier.MACRO, TaskTier.MESO):
            if tier == TaskTier.DOC_GROUNDED and item.extra_data:
                record.response = item.extra_data["macro_code"]
            else:
                clean_response = re.sub(r"^###.*?###\n+", "", item.payload, flags=re.MULTILINE|re.DOTALL)
                record.response = clean_response.strip() if clean_response.strip() else item.payload

        if len(record.instruction.strip()) < 25 or len(record.response.strip()) < 40:
            self.logger.debug(f"Low-quality output, skipping [{tier.value}]")
            return None

        return {
            "task_tier": tier.value,
            "source_module": item.module_meta.technical_name if item.module_meta else "unknown",
            "messages": [
                {"role": "system", "content": _ASSISTANT_SYSTEM},
                {"role": "user", "content": record.instruction},
                {"role": "assistant", "content": record.response},
            ],
            "metadata": {
                "file_count": len(item.source_files),
                "source_files": item.source_files,
                "response_tokens": len(record.response.split()),
            },
        }


class AsyncDataManager:
    def __init__(self, db_path: Path, jsonl_path: Path) -> None:
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._db_lock = asyncio.Lock()
        self._file_lock = asyncio.Lock()

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS completed_tasks (task_hash TEXT PRIMARY KEY, tier TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS tier_stats (tier TEXT PRIMARY KEY, count INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS precomputed_data (task_hash TEXT PRIMARY KEY, vision_text TEXT);
            """)
            await db.commit()

    async def is_processed(self, task_hash: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT 1 FROM completed_tasks WHERE task_hash = ?", (task_hash,)) as cur: return bool(await cur.fetchone())

    async def get_precomputed(self, task_hash: str) -> Dict[str, Optional[str]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT vision_text FROM precomputed_data WHERE task_hash = ?", (task_hash,)) as cur:
                row = await cur.fetchone()
                if row: return {"vision_text": row[0]}
                return {"vision_text": None}

    async def set_precomputed(self, task_hash: str, vision_text: Optional[str] = None) -> None:
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO precomputed_data (task_hash, vision_text) 
                    VALUES (?, ?) 
                    ON CONFLICT(task_hash) DO UPDATE SET 
                        vision_text = COALESCE(excluded.vision_text, vision_text)
                """, (task_hash, vision_text))
                await db.commit()

    async def commit(self, task_hash: str, tier: str, record: Dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._file_lock:
            async with aiofiles.open(self.jsonl_path, mode="a", encoding="utf-8") as fh: await fh.write(line)
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("INSERT OR IGNORE INTO completed_tasks (task_hash, tier) VALUES (?, ?)", (task_hash, tier))
                await db.execute("INSERT INTO tier_stats (tier, count) VALUES (?, 1) ON CONFLICT(tier) DO UPDATE SET count = count + 1", (tier,))
                await db.commit()

    async def get_stats(self) -> Dict[str, int]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT tier, count FROM tier_stats") as cur: return {row[0]: row[1] for row in await cur.fetchall()}

_shutdown = False
def _install_signal_handler() -> None:
    def _handler(sig, frame):
        global _shutdown
        if not _shutdown:
            print("\n⚠️  Graceful shutdown — finishing current tasks …")
            _shutdown = True
    signal.signal(signal.SIGINT, _handler)

async def _process(item: WorkItem, engine: InferenceEngine, dm: AsyncDataManager, progress: Progress, task_id) -> None:
    if _shutdown: return
    if await dm.is_processed(item.hash):
        progress.advance(task_id)
        return
    try:
        pre = await dm.get_precomputed(item.hash)
        result = await engine.generate(item, pre["vision_text"])
        if result: await dm.commit(item.hash, item.tier.value, result)
    except Exception as e:
        engine.logger.error(f"[{item.tier.value}] {item.hash[:8]}: {e}")
    finally:
        progress.advance(task_id)

async def _run_tier(queue: List[WorkItem], concurrency: int, label: str, engine: InferenceEngine, dm: AsyncDataManager, console: Console) -> None:
    if not queue or _shutdown: return
    sem = asyncio.Semaphore(concurrency)
    with Progress(SpinnerColumn(), TextColumn(f"[bold]{label}[/bold]"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
        tid = prog.add_task(f"{len(queue)} tasks", total=len(queue))
        async def _bounded(item: WorkItem) -> None:
            async with sem: await _process(item, engine, dm, prog, tid)
        await asyncio.gather(*[asyncio.create_task(_bounded(i)) for i in queue])

async def _unload_ollama(model_name: str, host: str):
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"), headers={"Content-Type": "application/json"})
        await asyncio.to_thread(urllib.request.urlopen, req, timeout=5)
    except Exception: pass

async def main(run_tiers: List[TaskTier]) -> None:
    _install_signal_handler()
    settings = Settings()
    console = Console()
    log_handlers = [
        RichHandler(console=console, rich_tracebacks=True),
        RotatingFileHandler("forge_v3.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=log_handlers)
    for h in log_handlers:
        if isinstance(h, RotatingFileHandler): h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger = logging.getLogger("forge_v3")

    console.print(Panel.fit("[bold bright_cyan]  Odoo 18 Forge Elite - SFT Dataset Generator v3  [/bold bright_cyan]\n[dim]Direction: REQUIREMENT -> CODE  |  Architecture: MULTI-PASS (8GB VRAM Optimized)[/dim]", border_style="bright_cyan"))

    if not settings.source_directory.exists():
        console.print(f"[bold red]ERROR:[/bold red] Source directory not found: {settings.source_directory}")
        return

    tokenizer = tiktoken.get_encoding("cl100k_base")
    qf = QualityFilter(tokenizer)
    engine = InferenceEngine(settings, logger)
    dm = AsyncDataManager(settings.db_file, settings.output_dataset_file)
    await dm.init_db()

    console.rule("[yellow]Phase 1 — Module Discovery & Parsing[/yellow]")
    module_paths = sorted([Path(root) for root, _, files in os.walk(settings.source_directory) if "__manifest__.py" in files])
    console.print(f"[green]OK[/green] Found [bold]{len(module_paths)}[/bold] Odoo modules.")

    micro_q, meso_q, macro_q, doc_q = [], [], [], []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as prog:
        t = prog.add_task("[cyan]Parsing source code & READMEs …", total=len(module_paths))
        for mod_path in module_paths:
            parser = OdooModuleParser(mod_path, qf, settings)
            meta = parser.parse_manifest()
            if not meta:
                prog.advance(t)
                continue
            parser.index_workspace()

            if TaskTier.MICRO in run_tiers:
                for payload, files in parser.extract_micro_tasks():
                    micro_q.append(WorkItem(TaskTier.MICRO, hashlib.sha256(payload.encode()).hexdigest(), payload, files, meta))
            if TaskTier.MESO in run_tiers:
                for payload, files in parser.extract_meso_clusters():
                    meso_q.append(WorkItem(TaskTier.MESO, hashlib.sha256(payload.encode()).hexdigest(), payload, files, meta))
            if TaskTier.MACRO in run_tiers:
                bundle = parser.extract_macro_bundle()
                if bundle:
                    macro_q.append(WorkItem(TaskTier.MACRO, hashlib.sha256(bundle[0].encode()).hexdigest(), bundle[0], bundle[1], meta))
            if TaskTier.DOC_GROUNDED in run_tiers:
                bundle = parser.extract_doc_grounded_bundle()
                if bundle:
                    doc_q.append(WorkItem(TaskTier.DOC_GROUNDED, hashlib.sha256(bundle[0].encode()).hexdigest(), bundle[0], bundle[1], meta, extra_data=bundle[2]))
            prog.advance(t)

    console.rule("[yellow]Phase 2 — Heavy Model Pre-computation[/yellow]")
    
    if doc_q and settings.enable_vision:
        console.print("[cyan]Running Phase 2.1: Vision Pre-computation (vLLM / Qwen VL)[/cyan]")
        for item in doc_q:
            if _shutdown: break
            pre = await dm.get_precomputed(item.hash)
            if not pre["vision_text"]:
                vision_txt = await engine.run_vision_precompute(item)
                if vision_txt: await dm.set_precomputed(item.hash, vision_text=vision_txt)
        await _unload_ollama(settings.vision_model, settings.ollama_host)
        console.print("[dim]Vision model gracefully evicted.[/dim]")

    console.rule("[yellow]Phase 3 — Text Generation (Qwen Coder)[/yellow]")
    t0 = time.time()
    await _run_tier(micro_q, settings.max_concurrent_micro, "MICRO", engine, dm, console)
    await _run_tier(meso_q, settings.max_concurrent_meso, "MESO ", engine, dm, console)
    await _run_tier(doc_q, settings.max_concurrent_doc_grounded, "DOC_GROUNDED", engine, dm, console)
    await _run_tier(macro_q, settings.max_concurrent_macro, "MACRO", engine, dm, console)
    elapsed = time.time() - t0

    console.rule("[yellow]Phase 4 — Report[/yellow]")
    stats = await dm.get_stats()
    report = Table(title="📊 Final Dataset Report", border_style="bright_green")
    report.add_column("Tier", style="bold")
    report.add_column("Records", justify="right", style="bright_green")
    for tier in [TaskTier.MICRO, TaskTier.MESO, TaskTier.DOC_GROUNDED, TaskTier.MACRO]:
        report.add_row(tier.value, str(stats.get(tier.value, 0)))
    report.add_section()
    report.add_row("[bold]TOTAL[/bold]", f"[bold bright_green]{sum(stats.values())}[/bold bright_green]")
    console.print(report)
    console.print(f"\n[bold green]Done![/bold green]  Dataset -> [cyan]{settings.output_dataset_file.resolve()}[/cyan]  |  Elapsed: [dim]{elapsed:.1f}s[/dim]")
    if _shutdown: console.print("[bold yellow]⚠️  Graceful exit — partial data safely written.[/bold yellow]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Odoo 18 SFT Dataset Generator v3 — Forge Elite")
    ap.add_argument("--tier", choices=["MICRO", "MESO", "MACRO", "DOC_GROUNDED", "ALL"], default="ALL")
    args = ap.parse_args()
    t_map = {
        "MICRO": [TaskTier.MICRO],
        "MESO": [TaskTier.MESO],
        "MACRO": [TaskTier.MACRO],
        "DOC_GROUNDED": [TaskTier.DOC_GROUNDED],
        "ALL": [TaskTier.MICRO, TaskTier.MESO, TaskTier.DOC_GROUNDED, TaskTier.MACRO],
    }
    asyncio.run(main(t_map[args.tier]))
