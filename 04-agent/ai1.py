import os
import openai
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ✅ TOOL FUNCTIONS
def run_command(command):
    print(f"Running: {command}")
    result = os.system(command)
    return f"✅ Command executed: {command} | Exit code: {result}"

def create_file(input):
    path = input["path"]
    content = input["content"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ File '{path}' created."

def read_file(path):
    if not os.path.exists(path):
        return f"❌ File '{path}' does not exist."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ✅ TOOL MAP
available_tools = {
    "run_command": run_command,
    "create_file": create_file,
    "read_file": read_file,
}

# ✅ SYSTEM PROMPT
SYSTEM_PROMPT = """
You are a helpful AI coding agent that follows four structured stages: plan, action, observe, and output.

🎯 You help users build coding projects based on prompts like:
- "Create a to-do app in React"
- "Make a Next.js app with App Router and TypeScript"
- "Generate reusable components"
- "Read/update a file"

🧠 Follow the loop:
1. plan → explain what needs to be done.
2. action → call ONE tool with input.
3. observe → wait for tool output.
4. output → final result or summary.

🛠️ Tools available:
- run_command (string)
- create_file (dict: { "path": "file path", "content": "text" })
- read_file (string)

✅ Output format (strict JSON):
{
  "step": "plan" | "action" | "observe" | "output",
  "content": "explanation or result",
  "function": "only if step is 'action'",
  "input": "input for the function"
}

📦 Notes:
- Always use forward slashes (`/`) in file paths.
- Do not use `workspace/` – create files in the real file structure.
- Create files inside the project directory if using frameworks like Next.js.

💡 Example:
- Project: Todo app using Next.js 14+, App Router, and TypeScript
- Use: `npx create-next-app@latest my-todo --typescript --app`
- Then: add `app/page.tsx` and reusable components

"""

# ✅ USER GOAL
goal = "Create a todo app using Next.js with TypeScript and App Router. Include app/page.tsx and one reusable component."

# ✅ AGENT LOOP
messages = [
    { "role": "system", "content": SYSTEM_PROMPT },
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
        print("❌ Invalid JSON response:\n", reply.content)
        break

    print(f"\n🧠 {content_json['step'].upper()}: {content_json['content']}")

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
        messages.append({
            "role": "function",
            "name": tool_name,
            "content": result
        })
    else:
        messages.append(reply)
