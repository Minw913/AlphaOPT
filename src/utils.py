import os
import re
import time
import json
from typing import Any, Callable
from dotenv import load_dotenv

#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

def _build_client(model: str, service: str):
    """
    Return (vendor, client) according to the model name
    """
    load_dotenv()
    
    # Get API keys from config or fallback to environment variables
    openrouter_key = config.api_keys.OPEN_ROUTER_KEY or os.getenv("OPEN_ROUTER_KEY")
    gemini_key = config.api_keys.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    openai_key = config.api_keys.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    
    if service and service.lower() != "null":
        # OpenAI-compatible (includes vLLM)
        from openai import OpenAI
        base_url = "https://openrouter.ai/api/v1"   # "http://127.0.0.1:8014/v1"
        # api_key = os.getenv("OPEN_ROUTER_KEY")    # "dummy-key"
        # client = OpenAI(api_key=api_key, base_url=base_url)

        if not openrouter_key:
            raise RuntimeError("OPEN_ROUTER_KEY is not set in config or environment")
        client = OpenAI(api_key=openrouter_key, base_url=base_url)
        return "openai", client

    # Gemini family
    if "gemini" in model.lower():
        # api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set in config or environment")
        from google import genai
        client = genai.Client(api_key=gemini_key)
        return "gemini", client

    if "gpt" in model.lower():
        # OpenAI-compatible (includes vLLM)
        from openai import OpenAI
        # api_key = os.getenv("OPENAI_API_KEY")
        # if not api_key:
        #     raise RuntimeError("OPENAI_API_KEY is not set")
        # client = OpenAI(api_key=api_key)
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set in config or environment")
        client = OpenAI(api_key=openai_key)
        return "openai", client
    
    # Default case - should not reach here
    raise RuntimeError(f"Unsupported model: {model}")


def call_llm_and_parse_with_retry(
    model: str,
    service: str | None = None,
    prompt: str | None = None,
    parse_fn: Callable[..., Any] = None,
    temperature: float = 0.7,        # sampling temperature
    max_retry: int = 3,
    sleep_sec: float = 2,
    verbose: bool = True,
    log_header: str | None = None,
    error_message: str | None = None,
) -> Any:
    """
    Send a chat prompt to OpenAI or Gemini automatically detected by `model`,
    retry on failure, and parse the raw text with `parse_fn`.
    """
    vendor, client = _build_client(model, service)

    def _send_request() -> str:
        """
        Dispatch the request to the proper SDK and return raw text.
        """
        # OpenAI call
        if vendor == "openai":
            msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature
            )
            return completion.choices[0].message.content

        # Gemini call
        if vendor == "gemini":
            from google.genai import types  
            # Disable thinking for gemini-2.5-flash
            if model == "gemini-2.5-flash":               
                completion = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                    ),
                )
            # gemini-2.5-pro cannot disable thinking
            else:                                               
                completion = client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
            return completion.text

        raise RuntimeError("Unsupported vendor")

    # Retry loop
    for attempt in range(1, max_retry + 1):

        if log_header is not None:
            if verbose: print(log_header)
        try:
            t0 = time.time()
            if verbose: print(f"[Attempt {attempt}/{max_retry}]\n")

            raw_text = _send_request()
            if verbose: 
                print(raw_text)
            resp_time = time.time() - t0

            if verbose: print(f"Done in {resp_time:.2f}s")

            result = parse_fn(raw_text)
            return result

        except Exception as err:
            # final attempt → raise
            if attempt == max_retry:
                raise RuntimeError(error_message or
                                f"\nFailed to parse results from LLM after {max_retry} attempts") from err
            # exponential back-off
            backoff = sleep_sec * (2 ** (attempt - 1))
            print(f"\nFailed to parse results on attempt {attempt}. \nError: {err}. \nRetrying in {backoff:.1f}s …")
            time.sleep(backoff)

def save_log_data(data, data_path):
    # Save and run corrected code
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    _, ext = os.path.splitext(data_path)
    if data:
        if ext == ".json":
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        if ext == ".txt":
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(data)

        if ext == ".py":
            with open(data_path, "w") as f:
                f.write(data)
    

def cal_time_cost(start_time, phase_name):
    """
    Calculate the the duration of a phase in minutes
    """
    total_minutes = (time.time() - start_time) / 60.0
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n[{phase_name}] took {hours}h {minutes}min")
    return round(total_minutes, 3)


def extract_json_object(text: str):
    """
    Extract the first JSON *object* from an LLM output and return it as a Python dict
    """
    candidate = None
    try:
        # Keep original for debugging
        raw = text

        # Locate the outermost JSON object
        start = raw.find('{')
        end   = raw.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in the text.")
        candidate = raw[start:end+1]

        cand = candidate.strip()

        # Remove trailing commas before ']' or '}'
        cand = re.sub(r",\s*(\]|\})", r"\1", cand)

        # 使用与extract_json_array相同的sanitize_json_like方法
        # 这样可以保持一致性，并且更简洁
        cand = sanitize_json_like(cand)

        # Parse JSON
        result = json.loads(cand)
        if not isinstance(result, dict):
            raise ValueError(f"The parsed JSON is not an object (dict); got {type(result).__name__}")
        return result

    except Exception as e:
        print("LLM raw text:\n", text)
        print("Extracted JSON candidate:\n", candidate if candidate is not None else '<no candidate>')
        print("Error during extracting json object:", repr(e))
        raise


def sanitize_json_like(text: str) -> str:
    # Escape backslashes that are not followed by a valid escape char: " \ / b f n r t u
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Remove trailing commas like ", ]" or ", }"
    text = re.sub(r',\s*([\]\}])', r'\1', text)
    return text


def _extract_json_fence_scope(text: str):
    """
    Return the inner content of a ```json fenced block, but DO NOT close the fence
    if the closing ``` appears inside a JSON string. This prevents early cutoff
    by lines like ```latex that live inside a JSON string value.
    Returns the 'scope' string or None if no json fence exists.
    """
    # Find the opening fence line: ^```[ \t]*json...
    open_pat = re.compile(r'^```[ \t]*json[^\n]*\n', re.IGNORECASE | re.MULTILINE)
    m = open_pat.search(text)
    if not m:
        return None

    i = m.end()                   # start scanning after opening fence newline
    n = len(text)
    in_str = False               # inside a JSON string? (double quotes only)
    escape = False               # previous char was a backslash
    line_start = i               # index of current line start

    while i < n:
        ch = text[i]

        if ch == '\n':
            # track line start
            line_start = i + 1

        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
        else:
            # not in a JSON string
            if ch == '"':
                in_str = True
            else:
                # Only consider a closing fence if it is at the start of a line
                # AND we're not in a JSON string.
                if i == line_start and text.startswith('```', i):
                    # Found the real closing fence
                    return text[m.end():i]

        i += 1

    # No closing fence found; treat until EOF as scope
    return text[m.end():]


def _find_array_slice_bracket_scan(s: str, start_idx: int = None):
    """
    Bracket-aware scan for a top-level JSON array slice in string s.
    Ignores brackets inside JSON strings (double-quoted) and handles escapes.
    Returns (start, end) indices or (None, None).
    """
    n = len(s)
    i = 0 if start_idx is None else max(0, start_idx)
    start = s.find('[', i)
    if start == -1:
        return (None, None)

    in_str = False
    escape = False
    depth = 0
    for j in range(start, n):
        ch = s[j]
        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            continue
        # not in string
        if ch == '"':
            in_str = True
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return (start, j)
    return (None, None)


def extract_json_array(text: str):
    """
    Extract the first JSON array from the LLM output and return it as a Python list of dicts.
    - If a ```json fence exists: use a fence-aware scope (won't close on ``` inside JSON strings),
    then bracket-scan to cut the top-level [ ... ].
    - Else: bracket-scan the whole text.
    """
    try:
        # 1) Fence-aware scope extraction
        scope = _extract_json_fence_scope(text)
        if scope is None:
            scope = text  # no json fence; fall back to whole text

        # 2) Bracket-aware slice to get the array block
        start, end = _find_array_slice_bracket_scan(scope)
        if start is None or end is None or end <= start:
            # Coarse fallback (very rare)
            start = scope.find('[')
            end   = scope.rfind(']')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON array found in the text.")
        block = scope[start:end+1].strip()

        # 3) Try direct JSON parsing (raw first)
        try:
            result = json.loads(block)
            if isinstance(result, list) and all(isinstance(x, dict) for x in result):
                return result
        except json.JSONDecodeError:
            pass

        # 4) Try after gentle sanitization
        block2 = sanitize_json_like(block)
        try:
            result = json.loads(block2)
            if isinstance(result, list) and all(isinstance(x, dict) for x in result):
                return result
        except json.JSONDecodeError:
            pass

        # 5) Last resort: scan subsequent top-level arrays (avoid naive regex that hits `[t]`)
        idx = end + 1
        last_fragment = None
        while True:
            s2, e2 = _find_array_slice_bracket_scan(scope, start_idx=idx)
            if s2 is None or e2 is None:
                break
            frag = scope[s2:e2+1].strip()
            last_fragment = frag
            for cand in (frag, sanitize_json_like(frag)):
                try:
                    res = json.loads(cand)
                    if isinstance(res, list) and all(isinstance(x, dict) for x in res):
                        return res
                except Exception:
                    continue
            idx = e2 + 1

        raise ValueError("No valid JSON array of objects found.")

    except Exception as e:
        # Debug output
        print("LLM raw text:\n", text)
        if 'block' in locals():
            print("Extracted block (raw):\n", block[:1000])
        if 'block2' in locals():
            print("Extracted block (sanitized):\n", block2[:1000])
        if 'last_fragment' in locals() and last_fragment is not None:
            print("Last scanned fragment (prefix):\n", last_fragment[:1000])
        print("Error:", repr(e))
        raise

