# asimov

A small [ampcode-style](https://ampcode.com/how-to-build-an-agent) coding agent to demonstrate naive or vulnerable
practices with agents. This is a WIP demonstration of a project I'm working on for someone else, not for public
use.

To run, install rust and do:

```bash
[you@machine]$ ANTHROPIC_API_KEY=... cargo run
```

Asimov contains the following components:
- `core/`: A small, self-rolled agent framework that provides traits for LLMs, Tools, and Agentic workflows. Agentic functions are achieved by running the LLM in a loop and encouraging it to chain tool calls.
- `anthropic/`: A small anthropic-based implementation of `core`'s Agent framework.
- `main.rs`: The user input and model setup workflows.
