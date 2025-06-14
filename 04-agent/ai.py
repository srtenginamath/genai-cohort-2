import os
import openai
import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import requests


load_dotenv()

client = OpenAI()

# Tool implementations
def run_command(command):
    print(f"Running: {command}")
    result = os.system(command)
    return f"Command executed: {command} | Exit code: {result}"

def create_file(input):
    path = os.path.join("workspace", input["path"])
    content = input["content"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ File '{path}' created."

def read_file(path):
    full_path = os.path.join("workspace", path)
    if not os.path.exists(full_path):
        return f"❌ File '{path}' does not exist."
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

# Available tools
available_tools = {
    "run_command": run_command,
    "create_file": create_file,
    "read_file": read_file,
}

SYSTEM_PROMPT = """
You are a helpful AI coding agent that follows four structured stages: plan, action, observe, and output.

🎯 Your job is to help users build coding projects or automate dev tasks based on prompts like:
- "Make a simple calculator website"
- "Create a to-do app in React"
- "Create a Next.js app with TypeScript and App Router"
- "Read a file and update its content"

🧠 Workflow:
You complete each request using a step-by-step loop:
1. Step 1 (plan): Analyze the user request and describe what needs to be done.
2. Step 2 (plan): Decide which tool(s) are needed.
3. Step 3 (action): Call ONE tool with appropriate input.
4. Step 4 (observe): Wait for the tool output before continuing.
5. Step 5 (output): Give the final result or summary.

🧾 Response Format:
Always return valid JSON in this exact format:
{
  "step": "plan" | "action" | "observe" | "output",
  "content": "detailed explanation or output",
  "function": "function_name (only for 'action' step)",
  "input": "tool input (string or object depending on tool)"
}

🚫 Rules:
- Only do ONE action per step.
- Do NOT skip observation steps.
- Be accurate and detailed when creating or editing files.
- Use forward slashes (`/`) in file paths for compatibility across Windows, macOS, and Linux.

🛠️ Available Tools:
- **run_command** `(string)` → Run terminal commands (e.g. mkdir, cd, git init, npm/yarn commands, scaffolding tools like create-next-app).
- **create_file** `(dict)` → Create or overwrite a file. Input: `{ "path": "file path", "content": "file content" }`
- **append_to_file** `(dict)` → Append content to a file. Input: `{ "path": "file path", "content": "content to append" }`
- **read_file** `(string)` → Read file content. Input: `"file path"`
- **delete_file** `(string)` → Delete a file. Input: `"file path"`

📦 Supported Tasks:
- Static frontend websites (HTML/CSS/JS)
- React and Vite-based projects
- Next.js 14+ apps using App Router and TypeScript
- Cross-platform file and project automation
- Editing, creating, and managing code files

💡 Example 1: Create a static calculator website

1. { "step": "plan", "content": "User wants a frontend calculator. We'll create index.html, style.css, and script.js." }
2. { "step": "plan", "content": "We'll use create_file to generate those files." }
3. { "step": "action", "function": "create_file", "input": { "path": "index.html", "content": "<!DOCTYPE html>..." } }
4. { "step": "observe", "content": "index.html created." }
5. ... repeat for style.css and script.js ...
6. { "step": "output", "content": "Calculator website created with index.html, style.css, and script.js." }

💡 Example 2: Create a Next.js app with App Router and TypeScript

1. { "step": "plan", "content": "User wants a modern Next.js app with App Router and TypeScript." }
2. { "step": "plan", "content": "We’ll run a command to scaffold the app and then add a custom page." }
3. { "step": "action", "function": "run_command", "input": "npx create-next-app@latest my-app --typescript --app" }
4. { "step": "observe", "content": "Next.js app scaffolded in 'my-app'." }
5. { "step": "action", "function": "run_command", "input": "cd my-app && npm install" }
6. { "step": "observe", "content": "Dependencies installed." }
7. { "step": "action", "function": "create_file", "input": { "path": "my-app/app/page.tsx", "content": "export default function Page() { return <main>Hello, world!</main>; }" } }
8. { "step": "observe", "content": "page.tsx created successfully." }
9. { "step": "output", "content": "Next.js app with App Router and TypeScript has been created and customized." }

Available tools: run_command, create_file, read_file. File path root is workspace/.

Respond step-by-step. Use forward slashes in file paths for cross-platform support.
"""

SYSTEM_PROMPT1 = """You are a helpful AI coding agent that completes programming tasks using tools.
Follow these steps:
1. plan
2. action (use one tool)
3. observe
4. output

Always return JSON in this format:
{
  "step": "plan" | "action" | "observe" | "output",
  "content": "explanation or result text",
  "function": "function_name (only if step is action)",
  "input": "function input"
}

Available tools: run_command, create_file, read_file. File path root is workspace/.

Respond step-by-step. Use forward slashes in file paths for cross-platform support.
"""

# Initial user goal
goal = "Create a todo app using Next.js with TypeScript and App Router. Include app/page.tsx and one reusable component."

# Agent loop
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    { "role": "user", "content": goal }
]

while True:
    response = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=messages
        )
    
    reply = response.choices[0].message
    try:
        content_json = json.loads(reply.content)
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON response:\n", reply.content)
        break

    # print(f"\n🤖 {content_json['step'].upper()}: {content_json['content']}") // when add not working
    
    if content_json["step"] == "output":
        break
    elif content_json["step"] == "action":
        tool_name = content_json["function"]
        tool_input = content_json["input"]
        if tool_name not in available_tools:
            print(f"❌ Unknown tool: {tool_name}")
            break
        result = available_tools[tool_name](tool_input)
        messages.append(reply)
        messages.append({ "role": "function", "name": tool_name, "content": result })
    else:
        messages.append(reply)
