#!/usr/bin/env python3
import re
import sys

if len(sys.argv) < 2:
    sys.exit(1)

content = sys.stdin.read()

# Заменить секрет на использование переменной окружения
pattern = r'OPENAI_API_KEY = "sk-proj-[^"]*"'
replacement = 'OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")\nif not OPENAI_API_KEY:\n    raise ValueError("OPENAI_API_KEY environment variable is required")'

content = re.sub(pattern, replacement, content)

# Убедиться что import os присутствует
if 'import os' not in content:
    content = content.replace('import asyncio', 'import asyncio\nimport os', 1)

sys.stdout.write(content)

