
# 📋 LATEX TABLE USAGE INSTRUCTIONS

## 🎯 **RECOMMENDED TABLE (Normal Size):**
Use the table from `proper_sized_classification_table.tex` - this has standard academic paper sizing.

## 📏 **TEXT SIZE OPTIONS:**

### Option 1: Standard Size (Recommended)
- Use the table as-is from `proper_sized_classification_table.tex`
- This is the normal size used in academic papers
- Text is readable and professional

### Option 2: Large Text
- Use the table from `large_text_classification_table.tex`
- Adds `\large` command to make entire table bigger
- Good if you want more emphasis

### Option 3: Custom Sizing
- Use the table from `custom_sized_classification_table.tex`
- You can manually adjust font sizes in Overleaf

## 🔧 **HOW TO MAKE TEXT BIGGER IN OVERLEAF:**

### Method 1: Use \large command
Add `\large` before `\begin{tabular}` to make the entire table larger.

### Method 2: Adjust specific elements
You can modify individual parts:
```latex
\caption{\Large Classification Performance Metrics}  % Larger caption
\textbf{\Large Class}  % Larger column headers
```

### Method 3: Use font size commands
```latex
\begin{table}[h]
\centering
\Large  % Makes everything in table larger
\caption{Classification Performance Metrics}
...
\end{table}
```

## 📱 **FOR MOBILE/PHONE VIEWING:**
- Use the `large_text_classification_table.tex` version
- This ensures readability on smaller screens
- The `\large` command makes text bigger for mobile viewing

## 💡 **PRO TIP:**
If you want even bigger text, you can use:
- `\Large` for very large text
- `\LARGE` for extra large text
- `\huge` for huge text

Just add these commands before `\begin{tabular}` in your chosen table.
