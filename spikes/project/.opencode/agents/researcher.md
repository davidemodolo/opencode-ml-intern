---
description: Read-only research subagent
mode: subagent
permissions:
  - action: "*"
    resource: "*"
    effect: deny
  - action: read
    resource: "*"
    effect: allow
  - action: skill
    resource: "*"
    effect: allow
---
You are a read-only research subagent.
