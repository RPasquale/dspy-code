import inspect
import dspy

print('LM class:', dspy.LM)
print('LM __init__:', inspect.signature(dspy.LM.__init__))
print('LM callable signature (if any):', inspect.signature(dspy.LM) if hasattr(dspy, 'LM') else 'n/a')
