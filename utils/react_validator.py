import re

def validate_trace_output(output: str) -> dict:
    issues = []
    passed = True

    # ✅ Check for mandatory REACT components
    if "Final Answer:" not in output:
        passed = False
        issues.append("❌ Missing 'Final Answer:' block.")

    if "Thought:" not in output:
        passed = False
        issues.append("❌ Missing 'Thought:' block.")

    if "Action:" not in output:
        passed = False
        issues.append("❌ Missing 'Action:' block.")

    if "Observation:" not in output:
        passed = False
        issues.append("❌ Missing 'Observation:' block.")

    # ✅ Check for lifted/generated quote distinction
    lifted = "🔹" in output
    generated = "🧠" in output
    if not lifted:
        issues.append("⚠️ No 🔹 lifted quote — nothing traceable to context.")
    if not generated:
        issues.append("⚠️ No 🧠 generated insight — lacks original reasoning.")

    # ✅ Check for valid REACT step sequencing
    react_blocks = re.findall(r"(Thought:.*?)(?=Thought:|Final Answer:|$)", output, re.DOTALL)
    for block in react_blocks:
        if "Action:" not in block or "Observation:" not in block:
            issues.append("⚠️ Incomplete REACT block — missing Action or Observation.")

    return {
        "is_valid": passed,
        "issues": issues,
        "lifted_present": lifted,
        "generated_present": generated,
        "react_steps": len(react_blocks)
    }
