import os
import asyncio
import json
import re
import sqlite3
import signal
import hashlib
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Set, Optional

import aiosqlite
import aiofiles
import tiktoken
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.logging import RichHandler
from tenacity import retry, stop_after_attempt, wait_exponential
import ollama

# ==========================================
# ENTERPRISE CONFIGURATION
# ==========================================
class Settings(BaseSettings):
    source_directory: Path = Field(..., description="Path to your Odoo 18 addons")
    output_dataset_file: Path = Field(default=Path("odoo18_expert_sft_dataset.jsonl"))
    db_file: Path = Field(default=Path("forge_state_machine.db"))
    model_name: str = Field(default="qwen2.5-coder:7b-instruct")
    max_context_tokens: int = Field(default=6500)
    max_concurrent_requests: int = Field(default=2)
    ollama_host: str = Field(default="http://localhost:11434")

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

# Initialize console and logger
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        RotatingFileHandler("forge.log", maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
    ]
)
logger = logging.getLogger("odoo_forge")

# File logger needs a standard formatter so the text file is readable
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.root.handlers:
    if isinstance(handler, RotatingFileHandler):
        handler.setFormatter(file_formatter)

# Initialize global subsystems
settings = Settings()
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

shutdown_flag = False

# ==========================================
# GRACEFUL SHUTDOWN INTERCEPTOR
# ==========================================
def signal_handler(sig, frame):
    global shutdown_flag
    if not shutdown_flag:
        logger.warning("Interrupt received! Finishing current inference and saving state safely...")
        shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# CORE SCHEMAS & PROMPTS
# ==========================================
SYSTEM_PROMPT = """You are an elite Odoo 18 Solutions Architect. 
Analyze the provided cross-functional module bundle (Python, XML, JS, CSV, CSS/SCSS).
Determine the unified business requirement that ties these files together.

Rules:
1. 'instruction': A detailed Business Requirement Document (BRD) user story.
2. 'input': Technical blueprint explaining how the logic interacts, explicitly detailing the directory structure and *why* files are placed in their specific paths (e.g., models/, views/, security/).
3. 'output': EXACT copy of the prompt's source code bundle. NO modifications."""

class SFTOutputSchema(BaseModel):
    instruction: str = Field(description="The functional business requirement driving this code.")
    input: str = Field(description="The structural architecture and technical dependencies.")
    output: str = Field(description="The exact provided code bundle.")

# ==========================================
# ADVANCED ODOO DEPENDENCY GRAPH
# ==========================================
class OdooGraphEngine:
    def __init__(self, module_path: Path):
        self.module_path = module_path
        self.module_name = module_path.name
        self.registry: Dict[str, str] = {}
        self.model_links: Dict[str, Set[str]] = {}
        
    def index_workspace(self):
        """Scans workspace and builds a bi-directional AST mapping."""
        for root, _, files in os.walk(self.module_path):
            root_path = Path(root)
            # Skip irrelevant directories but keep 'static' for JS/CSS and 'tests' for test cases
            if any(part in root_path.parts for part in ['i18n', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith(('.py', '.xml', '.js', '.csv', '.css', '.scss')) and '__manifest__' not in file:
                    full_path = root_path / file
                    rel_path = str(full_path.relative_to(self.module_path))
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            self.registry[rel_path] = content
                            self._extract_bi_directional_links(rel_path, content)
                    except Exception as e:
                        logger.error(f"IO Error on {rel_path}: {e}")

    def _extract_bi_directional_links(self, rel_path: str, content: str):
        """Advanced Regex to map Python logic to XML UI explicitly."""
        # 1. Catch Python Backend Models
        if rel_path.endswith('.py'):
            py_models = re.findall(r"_(?:name|inherit)\s*=\s*['\"]([^'\"]+)['\"]", content)
            for m in py_models:
                self.model_links.setdefault(m, set()).add(rel_path)
                
        # 2. Catch XML Frontend Target Models (Odoo specific UI mapping)
        elif rel_path.endswith('.xml'):
            xml_models = re.findall(r"<field name=[\"']model[\"']>([^<]+)</field>", content)
            for m in xml_models:
                self.model_links.setdefault(m, set()).add(rel_path)

    def compile_safe_bundles(self) -> List[str]:
        """Compiles clustered file payloads strictly guarded by Token limits."""
        bundles = []
        processed_files: Set[str] = set()

        for model_name, linked_files in self.model_links.items():
            cluster = list(linked_files)
            bundle_lines = [f"### ARCHITECTURE TIE: MODEL '{model_name}' ###"]
            
            for file_key in cluster:
                processed_files.add(file_key)
                bundle_lines.append(f"--- START FILE: {file_key} ---")
                bundle_lines.append(self.registry[file_key])
                bundle_lines.append(f"--- END FILE: {file_key} ---")
            
            payload = "\n\n".join(bundle_lines)
            token_count = len(tokenizer.encode(payload))
            
            if 100 < token_count <= settings.max_context_tokens and len(cluster) > 1:
                bundles.append(payload)

        # Catch remaining files (isolated logic, wizards, static assets)
        orphan_lines = ["### INDEPENDENT LOGIC & STRUCTURAL FILES ###"]
        orphan_payloads = []
        for rel_path, content in self.registry.items():
            if rel_path not in processed_files:
                single_payload = f"--- START FILE: {rel_path} ---\n{content}\n--- END FILE: {rel_path} ---"
                token_count = len(tokenizer.encode(single_payload))
                if 100 < token_count <= settings.max_context_tokens:
                    orphan_payloads.append(single_payload)
                    
        bundles.extend(orphan_payloads)
        return bundles

# ==========================================
# ATOMIC DATA MANAGER (AIOSQLITE + JSONL)
# ==========================================
class AsyncDataManager:
    def __init__(self, db_path: Path, jsonl_path: Path):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.db_lock = asyncio.Lock()
        self.file_lock = asyncio.Lock()

    async def init_db(self):
        """Initializes the database schema using aiosqlite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS completed_tasks (
                    task_hash TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    async def is_processed(self, task_hash: str) -> bool:
        """Checks if a task hash exists using non-blocking DB query."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT 1 FROM completed_tasks WHERE task_hash = ?", (task_hash,)) as cursor:
                return bool(await cursor.fetchone())

    async def commit_record(self, task_hash: str, record: Dict):
        """Atomic commit using JSONL and aiosqlite."""
        # 1. Append to JSONL File (Constant Time, Zero Memory Overhead)
        json_string = json.dumps(record, ensure_ascii=False) + '\n'
        async with self.file_lock:
            async with aiofiles.open(self.jsonl_path, mode='a', encoding='utf-8') as f:
                await f.write(json_string)
        
        # 2. Commit state to DB non-blockingly
        async with self.db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("INSERT INTO completed_tasks (task_hash) VALUES (?)", (task_hash,))
                await db.commit()

# ==========================================
# INFERENCE ENGINE
# ==========================================
client = ollama.AsyncClient(host=settings.ollama_host)

def clean_json_output(raw_text: str) -> str:
    """Robust parser to strip markdown artifacts from LLM outputs."""
    raw_text = raw_text.strip()
    # If wrapped in markdown code blocks like ```json ... ```
    match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def infer_llm(payload: str) -> Optional[Dict]:
    try:
        logger.info("="*40 + " REQUEST TO MODEL " + "="*40)
        logger.info(f"System Prompt:\n{SYSTEM_PROMPT}")
        logger.info(f"User Prompt:\nAnalyze this Odoo architecture:\n\n{payload}")
        logger.info("="*98)
        
        start_t = time.time()
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=settings.model_name,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': f"Analyze this Odoo architecture:\n\n{payload}"}
                    ],
                    format=SFTOutputSchema.model_json_schema(),
                    options={'temperature': 0.1, 'num_ctx': 8192} 
                ),
                timeout=600.0
            )
        except asyncio.TimeoutError:
            logger.error("LLM request timed out after 600 seconds.")
            raise
        
        elapsed = time.time() - start_t
        raw_output = response['message']['content']
        
        logger.info("="*35 + f" RESPONSE FROM MODEL ({elapsed:.2f}s) " + "="*35)
        logger.info(raw_output)
        logger.info("="*98)
        
        cleaned_json = clean_json_output(raw_output)
        
        data = SFTOutputSchema.model_validate_json(cleaned_json)
        data.output = payload
        return data.model_dump()
    except Exception as e:
        logger.warning(f"LLM Error, retrying... {e}")
        raise

# ==========================================
# MASTER ORCHESTRATOR
# ==========================================
async def process_task(task_hash: str, payload: str, data_manager: AsyncDataManager, progress: Progress, task_id):
    if shutdown_flag:
        return
        
    if await data_manager.is_processed(task_hash):
        progress.advance(task_id)
        return
        
    try:
        result = await infer_llm(payload)
        if result:
            await data_manager.commit_record(task_hash, result)
    except Exception as e:
        logger.error(f"Failed to process task {task_hash[:8]}: {e}")
    finally:
        progress.advance(task_id)

async def main():
    console.print(Panel.fit("[bold cyan]Odoo 18 Forge: PRO Dataset Engine (Elite Edition)[/bold cyan]", border_style="cyan"))
    
    # 0. Pre-checks
    if not settings.source_directory.exists():
        logger.error(f"Source directory not found: {settings.source_directory}")
        return

    data_manager = AsyncDataManager(settings.db_file, settings.output_dataset_file)
    await data_manager.init_db()
    
    # 1. Discovery Phase
    modules = [Path(root) for root, dirs, files in os.walk(settings.source_directory) if '__manifest__.py' in files]
    console.print(f"[green]✓[/green] Detected [bold]{len(modules)}[/bold] Odoo modules.")
    
    # 2. Graph Compilation Phase
    work_queue = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task1 = progress.add_task("[cyan]Building AST Dependency Graphs...", total=len(modules))
        for mod in modules:
            graph = OdooGraphEngine(mod)
            graph.index_workspace()
            for bundle in graph.compile_safe_bundles():
                stable_hash = hashlib.sha256(bundle.encode('utf-8')).hexdigest()
                work_queue.append((stable_hash, bundle))
            progress.advance(task1)
            
    console.print(f"[green]✓[/green] Graph compiled. [bold]{len(work_queue)}[/bold] strict token-safe clusters queued.")

    # 3. Execution Phase
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    
    async def bound_process_task(*args):
        async with semaphore:
            await process_task(*args)

    start_time = time.time()
    
    # Enhanced Rich Progress Bar with Time metrics
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task2 = progress.add_task("[magenta]Running Local AI Inference...", total=len(work_queue))
        
        tasks = []
        for task_hash, payload in work_queue:
            tasks.append(asyncio.create_task(bound_process_task(task_hash, payload, data_manager, progress, task2)))
            
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    if shutdown_flag:
        console.print("\n[bold yellow]⚠️ Graceful Exit Complete. All data safely appended to JSONL.[/bold yellow]")
    else:
        # Count JSONL lines natively
        records_count = 0
        if settings.output_dataset_file.exists():
            with open(settings.output_dataset_file, 'r', encoding='utf-8') as f:
                records_count = sum(1 for line in f)
        
        console.print(f"\n[bold green]🚀 Pipeline Complete! Total expert records: {records_count}[/bold green]")
        console.print(f"[dim]Finished in {elapsed:.2f} seconds.[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
