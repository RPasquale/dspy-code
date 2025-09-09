import dspy

print('dspy version:', getattr(dspy, '__version__', None))
print('has configure:', hasattr(dspy, 'configure'))
attrs = dir(dspy)
print('top-level attrs (sample):', attrs[:50])
for name in attrs:
    if name.lower() in ("lm", "language_model", "openai") or "lm" in name.lower():
        print('candidate:', name, getattr(dspy, name))
