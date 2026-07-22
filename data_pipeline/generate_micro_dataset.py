import os
import asyncio
import json
import re
import ast
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
# ENTERPRISE CONFIGURATION (V2 MICRO)
# ==========================================
class Settings(BaseSettings):
    source_directory: Path = Field(..., description="Path to your Odoo 18 addons")
    output_dataset_file: Path = Field(default=Path("odoo18_micro_sft_dataset.jsonl"))
    db_file: Path = Field(default=Path("micro_forge_tracking.db"))
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
        RotatingFileHandler("micro_forge.log", maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
    ]
)
logger = logging.getLogger("odoo_micro_forge")

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.root.handlers:
    if isinstance(handler, RotatingFileHandler):
        handler.setFormatter(file_formatter)

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
# CORE SCHEMAS & PROMPTS (V2 MICRO)
# ==========================================
SYSTEM_PROMPT = """You are an elite Odoo 18 Solutions Architect. 
Analyze the provided micro-component (e.g. a specific Python method, XML record, or template) from an Odoo module.
Determine its exact logic, edge cases, and technical implementation details.

Rules:
1. 'instruction': A detailed question or scenario asking how this specific logic works, what edge cases it handles, or why it is implemented this way.
2. 'input': A highly technical explanation of the code, detailing the logic flow, field dependencies, ORM methods used, and potential lifecycle impacts.
3. 'output': EXACT copy of the prompt's source code snippet. NO modifications."""

class SFTOutputSchema(BaseModel):
    instruction: str = Field(description="The scenario or question asking to explain this specific code logic.")
    input: str = Field(description="The deep technical explanation of the method/record logic.")
    output: str = Field(description="The exact provided code snippet.")

# ==========================================
# AST & MICRO-COMPONENT EXTRACTOR
# ==========================================
class MicroCodeExtractor:
    def __init__(self, module_path: Path):
        self.module_path = module_path
        self.snippets: List[str] = []

    def extract(self):
        """Scans workspace and extracts isolated methods and XML blocks."""
        for root, _, files in os.walk(self.module_path):
            root_path = Path(root)
            if any(part in root_path.parts for part in ['i18n', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py') and '__manifest__' not in file:
                    self._parse_python_ast(root_path / file)
                elif file.endswith('.xml'):
                    self._parse_xml_blocks(root_path / file)
                elif file.endswith('.js'):
                    self._parse_js_blocks(root_path / file)

    def _parse_python_ast(self, file_path: Path):
        """Extract individual methods from Python files using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            lines = source.splitlines(keepends=True)
            tree = ast.parse(source)
            
            rel_path = str(file_path.relative_to(self.module_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # We skip extremely short functions to ensure meaningful context
                    if node.end_lineno - node.lineno < 3:
                        continue
                        
                    # Extract the exact string from the source lines
                    snippet_lines = lines[node.lineno - 1:node.end_lineno]
                    snippet_text = "".join(snippet_lines)
                    
                    # Add context header
                    class_name = "UnknownClass"
                    # Try to find parent class name (naive upward traversal is hard in ast walk, but we can do it via parent pointers if we parsed manually, doing string context here for simplicity)
                    
                    payload = f"### FILE: {rel_path} | METHOD: {node.name} ###\n\n```python\n{snippet_text.strip()}\n```"
                    self._add_if_safe(payload)
                    
        except SyntaxError:
            pass # Ignore files with syntax errors
        except Exception as e:
            logger.error(f"Error parsing Python AST {file_path}: {e}")

    def _parse_xml_blocks(self, file_path: Path):
        """Extract standalone records and templates using Regex for exact formatting."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            rel_path = str(file_path.relative_to(self.module_path))
            
            # Find <record ...> ... </record>
            records = re.finditer(r"(<record\s+[^>]*>.*?</record>)", content, re.DOTALL | re.IGNORECASE)
            for r in records:
                snippet = r.group(1).strip()
                if snippet.count('\\n') > 2 or len(snippet) > 100: # Ensure it has some substance
                    payload = f"### FILE: {rel_path} | XML RECORD ###\n\n```xml\n{snippet}\n```"
                    self._add_if_safe(payload)
                    
            # Find <template ...> ... </template>
            templates = re.finditer(r"(<template\s+[^>]*>.*?</template>)", content, re.DOTALL | re.IGNORECASE)
            for t in templates:
                snippet = t.group(1).strip()
                if snippet.count('\\n') > 2 or len(snippet) > 100:
                    payload = f"### FILE: {rel_path} | XML TEMPLATE ###\n\n```xml\n{snippet}\n```"
                    self._add_if_safe(payload)

        except Exception as e:
            logger.error(f"Error parsing XML {file_path}: {e}")

    def _parse_js_blocks(self, file_path: Path):
        """Smart JS extraction using stack-based brace counting for OWL Components and Patches."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rel_path = str(file_path.relative_to(self.module_path))

            # Match signatures of Odoo JS components/patches
            # e.g., 'export class MyComponent', 'patch(MyComponent...', 'function init('
            pattern = re.compile(r'((?:export\s+)?class\s+\w+|patch\s*\(|function\s+\w+\s*\()')
            
            idx = 0
            while idx < len(content):
                match = pattern.search(content, idx)
                if not match:
                    break
                
                start_idx = match.start()
                # Find the first opening brace associated with this block
                brace_idx = content.find('{', start_idx)
                
                # If we hit another block definition before a brace, or no brace exists, skip
                next_match = pattern.search(content, match.end())
                if brace_idx == -1 or (next_match and next_match.start() < brace_idx):
                    idx = match.end()
                    continue
                
                open_braces = 0
                end_idx = -1
                in_string = False
                string_char = ''
                
                # Safely parse braces, ignoring braces inside strings
                for i in range(brace_idx, len(content)):
                    char = content[i]
                    
                    if not in_string and char in ('"', "'", '`'):
                        in_string = True
                        string_char = char
                    elif in_string and char == string_char and content[i-1] != '\\':
                        in_string = False
                    
                    if not in_string:
                        if char == '{':
                            open_braces += 1
                        elif char == '}':
                            open_braces -= 1
                            if open_braces == 0:
                                end_idx = i + 1
                                break
                
                if end_idx != -1:
                    # Capture the full block (e.g. full class or patch statement)
                    # If it's a patch(), we want to grab the closing `);` if it exists.
                    if content.startswith('patch', start_idx) and content.find(');', end_idx) == end_idx:
                        end_idx += 2

                    snippet = content[start_idx:end_idx].strip()
                    
                    # Ensure it has substance (more than just an empty class)
                    if snippet.count('\n') >= 2:
                        payload = f"### FILE: {rel_path} | JS COMPONENT ###\n\n```javascript\n{snippet}\n```"
                        self._add_if_safe(payload)
                        
                    idx = end_idx
                else:
                    idx = match.end()

        except Exception as e:
            logger.error(f"Error parsing JS {file_path}: {e}")

    def _add_if_safe(self, payload: str):
        token_count = len(tokenizer.encode(payload))
        if 50 < token_count <= settings.max_context_tokens:
            self.snippets.append(payload)

# ==========================================
# ATOMIC DATA MANAGER
# ==========================================
class AsyncDataManager:
    def __init__(self, db_path: Path, jsonl_path: Path):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.db_lock = asyncio.Lock()
        self.file_lock = asyncio.Lock()

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS completed_tasks (
                    task_hash TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    async def is_processed(self, task_hash: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT 1 FROM completed_tasks WHERE task_hash = ?", (task_hash,)) as cursor:
                return bool(await cursor.fetchone())

    async def commit_record(self, task_hash: str, record: Dict):
        json_string = json.dumps(record, ensure_ascii=False) + '\n'
        async with self.file_lock:
            async with aiofiles.open(self.jsonl_path, mode='a', encoding='utf-8') as f:
                await f.write(json_string)
        
        async with self.db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("INSERT INTO completed_tasks (task_hash) VALUES (?)", (task_hash,))
                await db.commit()

# ==========================================
# INFERENCE ENGINE
# ==========================================
client = ollama.AsyncClient(host=settings.ollama_host)

def clean_json_output(raw_text: str) -> str:
    raw_text = raw_text.strip()
    match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def infer_llm(payload: str) -> Optional[Dict]:
    try:
        logger.info("="*40 + " REQUEST TO MODEL " + "="*40)
        
        start_t = time.time()
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=settings.model_name,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': f"Analyze this specific Odoo 18 code component:\n\n{payload}"}
                    ],
                    format=SFTOutputSchema.model_json_schema(),
                    options={'temperature': 0.1, 'num_ctx': 8192} 
                ),
                timeout=300.0 # Shorter timeout since snippets are smaller
            )
        except asyncio.TimeoutError:
            logger.error("LLM request timed out after 300 seconds.")
            raise
        
        elapsed = time.time() - start_t
        raw_output = response['message']['content']
        
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
    console.print(Panel.fit("[bold cyan]Odoo 18 Forge: MICRO-DETAILS Dataset Engine (V2)[/bold cyan]", border_style="cyan"))
    
    if not settings.source_directory.exists():
        logger.error(f"Source directory not found: {settings.source_directory}")
        return

    data_manager = AsyncDataManager(settings.db_file, settings.output_dataset_file)
    await data_manager.init_db()
    
    modules = [Path(root) for root, dirs, files in os.walk(settings.source_directory) if '__manifest__.py' in files]
    console.print(f"[green]✔[/green] Detected [bold]{len(modules)}[/bold] Odoo modules.")
    
    work_queue = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task1 = progress.add_task("[cyan]Extracting AST Methods & XML Blocks...", total=len(modules))
        for mod in modules:
            extractor = MicroCodeExtractor(mod)
            extractor.extract()
            for snippet in extractor.snippets:
                stable_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()
                work_queue.append((stable_hash, snippet))
            progress.advance(task1)
            
    console.print(f"[green]✔[/green] Extraction complete. [bold]{len(work_queue)}[/bold] micro-components queued.")

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    
    async def bound_process_task(*args):
        async with semaphore:
            await process_task(*args)

    start_time = time.time()
    
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
        records_count = 0
        if settings.output_dataset_file.exists():
            with open(settings.output_dataset_file, 'r', encoding='utf-8') as f:
                records_count = sum(1 for line in f)
        
        console.print(f"\n[bold green]✅ Pipeline Complete! Total expert micro-records: {records_count}[/bold green]")
        console.print(f"[dim]Finished in {elapsed:.2f} seconds.[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
