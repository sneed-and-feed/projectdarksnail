import re
import sys


def audit_file(filepath: str) -> int:
    print(f"Auditing GFM KaTeX compliance (all 21 rules) for: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [raw.rstrip("\r\n") for raw in f.readlines()]

    issues = []
    in_code_block = False
    in_math_block = False
    math_expr_count = 0

    for idx, line in enumerate(lines):
        line_num = idx + 1

        if line.strip().startswith("```"):
            if line.strip().startswith("```math"):
                in_math_block = True
                math_expr_count += 1
                if len(line) - len(line.lstrip()) > 0:
                    issues.append((line_num, "Rule 8: Indented ```math block (must start at Column 0)"))
                if idx > 0 and lines[idx - 1].strip() != "":
                    issues.append((line_num, "Rule 8: Missing blank line before ```math block"))
            elif in_math_block and line.strip() == "```":
                in_math_block = False
                if len(line) - len(line.lstrip()) > 0:
                    issues.append((line_num, "Rule 8: Indented closing ``` for math block (must start at Column 0)"))
                if idx + 1 < len(lines) and lines[idx + 1].strip() != "":
                    issues.append((line_num, "Rule 8: Missing blank line after closing ``` of math block"))
            elif in_code_block and line.strip() == "```":
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block and not in_math_block:
            continue

        # Rule 12: 4+ spaces leading indentation after blank line
        if not in_math_block and idx > 0 and lines[idx - 1].strip() == "" and line.startswith("    ") and line.strip() != "":
            issues.append((line_num, "Rule 12: >=4 spaces leading indent after blank line (triggers indented code block)"))

        # Rule 8 / 14: $$ in prose
        if not in_math_block and "$$" in line:
            issues.append((line_num, "Rule 8: $$ delimiter found in prose/table (use ```math block or single $...$)"))

        # Rule 1: Backtick math
        if "$`" in line or "`$" in line:
            issues.append((line_num, "Rule 1: Backtick math found ($` or `$)"))

        # Rule 2: Markdown bold/italics containing $
        bold_spans = re.findall(r"\*\*((?:(?!\*\*).)+)\*\*", line)
        if any("$" in b for b in bold_spans) or re.search(r"(?<!\*)\*[^\s$\n*]*\$[^\s$\n*]*\*(?!\*)", line):
            issues.append((line_num, "Rule 2: Markdown bold/italics wrapping math delimiter"))

        # Rule 3: \text{--} in math
        if r"\text{--}" in line:
            issues.append((line_num, r"Rule 3: \text{--} ligature in math"))

        # Rule 5: Raw | inside table math cells
        if line.strip().startswith("|") and "$" in line:
            cells = line.split("|")[1:-1]
            for c in cells:
                if c.count("$") % 2 != 0:
                    issues.append((line_num, "Rule 5: Raw pipe '|' inside table math cell (use \\mid or \\lvert...\\rvert)"))
                    break

        # Rule 6: Unclosed $ delimiter on a line
        if not in_math_block and line.count("$") % 2 != 0:
            issues.append((line_num, "Rule 6: Unclosed/odd number of '$' delimiters on line"))

        # Rule 7: \hline in markdown table
        if r"\hline" in line:
            issues.append((line_num, r"Rule 7: \hline in markdown table"))

        # Rule 10: List item beginning with inline math before bold title
        if re.search(r"^\s*\d+\.\s*\$", line):
            issues.append((line_num, "Rule 10: Numbered list item begins with inline math"))

        # Rule 11: \operatorname
        if r"\operatorname" in line:
            issues.append((line_num, r"Rule 11: \operatorname is disallowed in GitHub KaTeX (use \mathrm)"))

        # Rule 13: \widetilde{\mathrm{SL}} or Lie group underscore subscripts
        if r"\widetilde{\mathrm" in line or re.search(r"\\mathrm\{(SL|SU|SO|Sp)\}_", line):
            issues.append((line_num, "Rule 13: Lie group underscore subscript or \\widetilde{\\mathrm{...}}"))

        # Rule 14: Parenthesized math ($...$) containing internal parentheses
        if not in_math_block:
            for m in re.finditer(r"\(\$([^$]+)\$\)", line):
                if "(" in m.group(1) or ")" in m.group(1):
                    issues.append((line_num, "Rule 14: Parenthesized math ($...$) containing internal parentheses"))

        # Rule 15: \left\{ or \right\} in math
        if r"\left\{" in line or r"\right\}" in line:
            issues.append((line_num, r"Rule 15: \left\{ or \right\} in math (use \left\lbrace and \right\rbrace)"))

        # Rule 16: Brace-preceded or punctuation-preceded subscripts
        if re.search(r"\\(mathcal|mathbb|mathbf|mathfrak|mathrm|text|vec)\{[^}]+\}_", line):
            issues.append((line_num, "Rule 16: Brace-preceded font macro subscript (e.g. \\mathcal{H}_L or \\mathrm{II}_{1,1})"))
        for m16 in re.finditer(r".{0,12}[\}\)\]]_.{0,12}", line):
            issues.append((line_num, f"Rule 16b: Closing brace/paren/bracket immediately followed by '_': '{m16.group(0)}'"))

        # Rule 17: Math inside markdown link
        if re.search(r"\[[^\]]*\$[^\]]*\]\([^\)]*\)", line):
            issues.append((line_num, "Rule 17: Math inside markdown link anchor text (or [math] followed by (...) on same line)"))

        # Rule 18: Prime-preceded subscripts
        if re.search(r"'_", line):
            issues.append((line_num, "Rule 18: Prime-preceded subscript '_"))

        # Rule 19: Hyphen-math
        if not in_math_block and re.search(r"[a-zA-Z0-9]-\$(?!\$)", line):
            issues.append((line_num, "Rule 19: Hyphen immediately preceding inline math (e.g. low-$x$)"))

        # Rule 20: Unescaped & inside \text or \mathrm
        if re.search(r"\\(text|mathrm)\{[^}]*?[^\\]&.*?\}", line):
            issues.append((line_num, "Rule 20: Unescaped literal '&' inside \\text{...} or \\mathrm{...}"))

        # Rule 21: Underscore or \textunderscore inside \text or \mathrm
        if re.search(r"\\(text|mathrm)\{[^}]*?(_|\\textunderscore).*?\}", line):
            issues.append((line_num, "Rule 21: '_' or '\\textunderscore' inside \\text{...} or \\mathrm{...} (use code span outside math or hyphens)"))

        if in_math_block:
            if re.search(r"<[a-zA-Z]", line):
                issues.append((line_num, "Raw '<' followed by letter in display math (use \\lt)"))
            if "*" in line:
                issues.append((line_num, "Rule 4: Raw '*' inside display math (use \\ast)"))
        elif "$" in line:
            math_spans = re.findall(r"\$([^$]+)\$", line)
            math_expr_count += len(math_spans)
            underscore_spans = [s for s in math_spans if "_" in s]
            if len(underscore_spans) >= 2:
                issues.append((line_num, f"Rule 9: {len(underscore_spans)} inline math spans with '_' on same line"))
            for span in math_spans:
                if re.search(r"<[a-zA-Z]", span):
                    issues.append((line_num, "Raw '<' followed by letter in inline math (use \\lt)"))
                if "*" in span:
                    issues.append((line_num, "Rule 4: Raw '*' inside inline math (use \\ast)"))

    if math_expr_count >= 800:
        issues.append((0, f"Budget Rule: Total math expression count {math_expr_count} >= 800 budget target"))

    print(f"Total math expressions: {math_expr_count} (Target < 800, GitHub limit ~1,000)")
    if not issues:
        print("PASS: 0 KaTeX/GFM compliance issues detected across all 21 rules!")
        return 0
    else:
        print(f"FAILED: {len(issues)} compliance issues detected:")
        for l_num, desc in issues:
            print(f"  Line {l_num}: {desc}")
        return 1


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["papers/darksnail_v2_preprint.md"]
    rc = 0
    for target in targets:
        rc |= audit_file(target)
    sys.exit(rc)

