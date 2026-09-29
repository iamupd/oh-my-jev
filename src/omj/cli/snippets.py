"""Connection snippet generators printed by `omj init` on success (REQ-010).

Both snippets target the *origin* (scheme+host+port, no path): the official
Python SDK appends its own `/v1/systemone` path internally
(typesafe_sdk._core.constants.SYSTEM_ONE_PATH), and the TypeScript snippet
follows the same convention so both examples stay symmetric.
"""

from __future__ import annotations


def python_snippet(base_url: str) -> str:
    """Return a copy-pasteable Python example using the official typesafe_sdk client."""
    return (
        "from typesafe_sdk import TypeSafeClient\n"
        "\n"
        f'client = TypeSafeClient(api_key="local", base_url="{base_url}")\n'
        "\n"
        "result = client.system_one(\n"
        '    state="2 + 2 = 4",\n'
        "    questions={\n"
        '        "check": {"type": "noul", "instructions": "Is the statement true?"},\n'
        "    },\n"
        '    model="jev-latest",\n'
        ")\n"
        "print(result)\n"
    )


def typescript_snippet(base_url: str) -> str:
    """Return a copy-pasteable TypeScript example using @ai-sdk/gateway."""
    return (
        'import { createGateway } from "@ai-sdk/gateway";\n'
        "\n"
        f'const gateway = createGateway({{ baseURL: "{base_url}" }});\n'
        "\n"
        "async function evaluate() {\n"
        f'  const response = await fetch("{base_url}/v1/systemone", {{\n'
        '    method: "POST",\n'
        "    headers: {\n"
        '      "Content-Type": "application/json",\n'
        '      Authorization: "Bearer local",\n'
        "    },\n"
        "    body: JSON.stringify({\n"
        '      model: "jev-latest",\n'
        '      state: "2 + 2 = 4",\n'
        "      questions: {\n"
        '        check: { type: "noul", instructions: "Is the statement true?" },\n'
        "      },\n"
        "    }),\n"
        "  });\n"
        "  return response.json();\n"
        "}\n"
        "\n"
        "evaluate().then((result) => console.log(result));\n"
        "// gateway is the @ai-sdk/gateway provider bound to the same origin above.\n"
        "void gateway;\n"
    )
