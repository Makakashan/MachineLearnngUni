# MachineLearnngUni

This repository contains my university assignments for the **Machine Lerning** course.

## Project structure

- `python_tasks_1/` - Python assignment scripts
- `python_tasks_1/output/trajectories/` - generated trajectory images

## Scripts

- `python_tasks_1/trebuchet.py`
- `python_tasks_1/biorhythms.py`

## Editor setup

For correct import checks in text editors (Zed, VS Code, etc.), you may need a local Pyright config that points to the project virtual environment.

Example `pyrightconfig.json`:

```json
{
  "venvPath": ".",
  "venv": "venv",
  "include": ["python_tasks_1"]
}
```

This file can stay local and be ignored by git.

## Note

All code in this repository is made for university learning purposes.
