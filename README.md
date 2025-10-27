[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)


# The F1 GPT

Getting information about Formula 1 history with a simple chatbot. Built with LangChain for the backend and React with RSbuild for the frontend.

## Getting started

You will need to install [Node.js](https://nodejs.org/en/download) on your device.

To install react with RS build, you can follow this [tutorial](https://rsbuild.rs/guide/start/quick-start), from which I outline below the main steps.
In your default shell, run:
```powershell
npm create rsbuild@latest
```
After that, run:
```powershell
npm install
npm run dev
```
This will install all dependencies needed and open the React App in your local browser.

## Contributing to the repo

First fork the repo.
You can then clone this repo running:
```bash
git clone https://github.com/julescrevola/f1_info.git
```

### Set up coding environment

To use this repo, first run:
```bash
source cli-aliases.sh
```
This will make sure that the aliases are loaded in your bash terminal.

You can then install the environment with:
```bash
envc
```
And you can update it with:
```bash
envu
```

To install pre-commit hooks, run:
```bash
pre-commit install
```
