# 🚀 Streamlined Development Workflow for DSPy Code

## Overview

DSPy Code now includes a **streamlined development workflow** that makes it super easy for human users to publish packages, push to GitHub, and update version numbers. No more complex commands or manual steps!

## 🎯 **Super Easy Commands**

### **Quick Development Cycle**
```bash
# One command to rule them all - format, test, commit, push
./scripts/dev.sh quick "Fix bug in CLI"

# Or from within the agent
dspy-code> dev quick "Fix bug in CLI"
```

### **Full Release Workflow**
```bash
# Complete release: test, lint, version bump, build, commit, push, GitHub release, PyPI publish
./scripts/dev.sh release patch    # 0.1.0 → 0.1.1
./scripts/dev.sh release minor    # 0.1.0 → 0.2.0
./scripts/dev.sh release major    # 0.1.0 → 1.0.0

# Or from within the agent
dspy-code> release patch
```

### **Individual Commands**
```bash
# Build and test
./scripts/dev.sh build
./scripts/dev.sh test

# Lint and format
./scripts/dev.sh lint
./scripts/dev.sh format

# Version management
./scripts/dev.sh version          # Show current version
./scripts/dev.sh bump patch       # Bump version

# Publishing
./scripts/dev.sh publish          # Publish to PyPI
./scripts/dev.sh publish-test     # Publish to Test PyPI

# Git operations
./scripts/dev.sh status           # Show git status
./scripts/dev.sh commit "message" # Commit and push
```

## 🤖 **From Within the Agent**

When using the enhanced coding mode, you can run all these commands directly:

```bash
# Start the agent in coding mode
uv run dspy-code --coding-mode

# Then use the dev commands
dspy-code> dev quick "Add new feature"
dspy-code> dev test
dspy-code> dev build
dspy-code> release minor
dspy-code> publish
```

## 📋 **Complete Workflow Examples**

### **Daily Development**
```bash
# 1. Make your changes
# 2. Quick dev cycle (format, test, commit, push)
./scripts/dev.sh quick "Add new CLI command"

# 3. That's it! Your changes are committed and pushed
```

### **Feature Release**
```bash
# 1. Complete your feature
# 2. Run full release workflow
./scripts/dev.sh release minor

# 3. Everything is automated:
#    - Tests run
#    - Code is linted
#    - Version is bumped
#    - Package is built
#    - Changes are committed and pushed
#    - GitHub release is created
#    - Package is published to PyPI
```

### **Hotfix Release**
```bash
# 1. Fix the bug
# 2. Quick patch release
./scripts/dev.sh release patch

# 3. Bug fix is released and published
```

## 🔧 **What Each Command Does**

### **`dev quick [message]`**
1. Formats code with `ruff format`
2. Runs tests with `pytest`
3. Commits changes with your message
4. Pushes to GitHub

### **`release [type]`**
1. Runs tests and linting
2. Bumps version (patch/minor/major)
3. Builds the package
4. Commits version change
5. Pushes to GitHub
6. Creates GitHub release
7. Publishes to PyPI

### **`publish [--test]`**
1. Publishes to PyPI (or Test PyPI if `--test`)

## 🎨 **Beautiful Output**

All commands provide clear, colored output:
- ✅ **Green**: Success
- ❌ **Red**: Error
- ⚠️ **Yellow**: Warning
- 🔄 **Blue**: In progress

## 🛡️ **Safety Features**

- **Confirmation prompts** for destructive operations (release, publish)
- **Automatic rollback** if any step fails
- **Git status checks** before releases
- **Test validation** before publishing

## 📊 **Learning Integration**

The agent learns from your development patterns:
- **Successful releases** are recorded as learning patterns
- **Failed operations** help improve future suggestions
- **Feedback scores** improve the agent's performance

## 🚀 **Getting Started**

1. **Make sure you're in the project root**
2. **Run your first quick dev cycle:**
   ```bash
   ./scripts/dev.sh quick "Initial setup"
   ```
3. **Try a full release:**
   ```bash
   ./scripts/dev.sh release patch
   ```

## 💡 **Pro Tips**

### **Use the Agent for Everything**
```bash
# Start the agent
uv run dspy-code --coding-mode

# Do all your development through the agent
dspy-code> dev quick "Fix bug"
dspy-code> dev test
dspy-code> release patch
dspy-code> learn "successful patch release workflow"
dspy-code> feedback 9
```

### **Automate Everything**
- The agent learns from your successful patterns
- It can suggest optimal release strategies
- It remembers your preferred workflows

### **Stay Safe**
- Always test before releasing
- Use `publish-test` for testing PyPI publishing
- Check git status before major releases

## 🔄 **Integration with Existing Systems**

The streamlined workflow integrates seamlessly with:
- **GitHub**: Automatic releases and tags
- **PyPI**: Automatic publishing
- **CI/CD**: Can be integrated into GitHub Actions
- **Docker**: Works with the existing Docker stack
- **Streaming**: All actions are logged and learned from

## 📈 **Benefits**

1. **⚡ Speed**: One command does everything
2. **🛡️ Safety**: Built-in checks and confirmations
3. **🧠 Learning**: Agent learns from your patterns
4. **🎯 Consistency**: Standardized workflow across team
5. **📊 Visibility**: Clear feedback on every step
6. **🔄 Automation**: No manual steps required

## 🎉 **Ready to Use!**

Your development workflow is now **super streamlined**! Just run:

```bash
./scripts/dev.sh quick "Your changes"
```

And watch the magic happen! 🚀
