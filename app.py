import streamlit as st
import google.generativeai as genai
import time
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import math
import io
import contextlib

# Configure page
st.set_page_config(
    page_title="DSA Mastery", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemma-3-27b-it')

# Data structures and topics
DS_TOPICS = {
    "Arrays": ["Basic Operations", "Two Pointers", "Sliding Window", "Prefix Sum"],
    "Strings": ["Pattern Matching", "String Manipulation", "Palindromes", "Anagrams"],
    "Linked Lists": ["Singly Linked List", "Doubly Linked List", "Circular Lists", "Merging"],
    "Stacks": ["Basic Operations", "Expression Evaluation", "Monotonic Stack", "Next Greater Element"],
    "Queues": ["Basic Operations", "Circular Queue", "Priority Queue", "Deque"],
    "Recursion": ["Base Cases", "Recursive Relations", "Tree Recursion", "Tail Recursion"],
    "Backtracking": ["N-Queens", "Sudoku", "Permutations", "Combinations", "Maze Solving"],
    "Binary Trees": ["Tree Traversal", "Tree Construction", "Tree Properties", "Path Problems"],
    "Binary Search Trees": ["BST Operations", "BST Validation", "BST to Array", "Balanced BST"],
    "Heaps": ["Min Heap", "Max Heap", "Heap Sort", "Priority Queue", "K-way Merge"],
    "Hashing": ["Hash Tables", "Hash Functions", "Collision Resolution", "Two Sum Problems"],
    "Dynamic Programming": ["1D DP", "2D DP", "Memoization", "Tabulation", "Classic Problems"],
    "Bit Manipulation": ["Basic Operations", "Bit Tricks", "XOR Properties", "Counting Bits"],
    "Graphs": ["Graph Representation", "BFS", "DFS", "Shortest Path"],
    "Greedy": ["Activity Selection", "Huffman Coding", "Fractional Knapsack", "MST"],
    "Divide & Conquer": ["Merge Sort", "Quick Sort", "Binary Search", "Closest Pair"],
    "Trie": ["Prefix Tree", "Auto-complete", "Word Search", "Longest Prefix"],
    "Segment Trees": ["Range Query", "Range Update", "Lazy Propagation", "Applications"]
}

DIFFICULTY_LEVELS = ["Newbie", "Easy", "Medium", "Hard", "Nightmare"]

# Initialize session state
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'user_solution' not in st.session_state:
    st.session_state.user_solution = ""
if 'hint_count' not in st.session_state:
    st.session_state.hint_count = 0
if 'solved_problems' not in st.session_state:
    st.session_state.solved_problems = []

def generate_problem(topic, difficulty, language, model):
    """Generate a coding problem using Gemini"""
    prompt = f"""
    Generate a {difficulty} level coding problem for {topic} in {language}.
    
    Return the response in this exact JSON format:
    {{
        "title": "Problem Title",
        "description": "Detailed problem description with examples",
        "input_format": "Input format description",
        "output_format": "Output format description",
        "constraints": "Problem constraints",
        "example_input": "Sample input",
        "example_output": "Sample output",
        "hints": ["Hint 1", "Hint 2", "Hint 3"],
        "solution_approach": "High-level approach to solve",
        "time_complexity": "Expected time complexity",
        "space_complexity": "Expected space complexity",
        "starter_code": "Starter code template in {language}",
        "solution_code": "Complete solution in {language}"
    }}
    
    Make sure the problem is educational and focuses on understanding {topic} concepts.
    Provide language-specific starter code and solution.
    """
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Failed to parse response"}
    except Exception as e:
        return {"error": f"Error generating problem: {str(e)}"}

def get_code_editor_theme(language):
    """Get appropriate syntax highlighting for code editor"""
    theme_map = {
        "python": "python",
        "java": "java",
        "cpp": "cpp",
        "javascript": "javascript", 
        "c": "c",
        "go": "go",
        "rust": "rust",
        "csharp": "csharp"
    }
    return theme_map.get(language, "python")

def get_starter_template(language):
    """Get language-specific starter templates"""
    templates = {
        "python": "# Write your solution here\ndef solve():\n    pass\n\n# Test your solution\nif __name__ == '__main__':\n    pass",
        "java": "public class Solution {\n    public static void main(String[] args) {\n        // Write your solution here\n    }\n}",
        "cpp": "#include <iostream>\n#include <vector>\nusing namespace std;\n\nint main() {\n    // Write your solution here\n    return 0;\n}",
        "javascript": "// Write your solution here\nfunction solve() {\n    \n}\n\n// Test your solution\nconsole.log(solve());",
        "c": "#include <stdio.h>\n#include <stdlib.h>\n\nint main() {\n    // Write your solution here\n    return 0;\n}",
        "go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    // Write your solution here\n}",
        "rust": "fn main() {\n    // Write your solution here\n}\n\nfn solve() {\n    \n}",
        "csharp": "using System;\n\nclass Program {\n    static void Main() {\n        // Write your solution here\n    }\n}"
    }
    return templates.get(language, templates["python"])
def visualize_array(arr, title="Array Visualization"):
    """Create interactive array visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(arr))),
        y=arr,
        text=arr,
        textposition='auto',
        marker_color='lightblue',
        marker_line_color='navy',
        marker_line_width=2
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        height=400
    )
    
    return fig

def visualize_sorting_steps(arr, algorithm="bubble_sort"):
    """Visualize sorting algorithm steps"""
    steps = []
    arr_copy = arr.copy()
    
    if algorithm == "bubble_sort":
        n = len(arr_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr_copy[j] > arr_copy[j + 1]:
                    arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                    steps.append(arr_copy.copy())
    
    return steps

def visualize_binary_tree(nodes, title="Binary Tree"):
    """Create binary tree visualization"""
    if not nodes:
        return go.Figure()
    
    # Calculate positions for binary tree layout
    fig = go.Figure()
    
    # Simple binary tree visualization (can be expanded)
    levels = {}
    for i, node in enumerate(nodes):
        level = int(math.log2(i + 1)) if i > 0 else 0
        if level not in levels:
            levels[level] = []
        levels[level].append((i, node))
    
    # Add nodes
    for level, level_nodes in levels.items():
        y_pos = -level
        for i, (idx, value) in enumerate(level_nodes):
            x_pos = (i - len(level_nodes)/2) * (4 - level)
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[y_pos],
                mode='markers+text',
                text=[str(value)],
                textposition='middle center',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_heap(arr, heap_type="min"):
    """Visualize heap structure"""
    fig = go.Figure()
    
    # Create heap visualization
    for i in range(len(arr)):
        level = int(math.log2(i + 1)) if i > 0 else 0
        y_pos = -level
        x_pos = (i - (2**level - 1)) * (8 / (2**level))
        
        color = 'lightgreen' if heap_type == 'min' else 'lightcoral'
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos],
            mode='markers+text',
            text=[str(arr[i])],
            textposition='middle center',
            marker=dict(size=40, color=color, line=dict(width=2, color='darkgreen' if heap_type == 'min' else 'darkred')),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{heap_type.capitalize()} Heap Visualization",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_recursion_tree(func_name, depth=4):
    """Create recursion tree visualization"""
    fig = go.Figure()
    
    # Example for fibonacci recursion
    if func_name == "fibonacci":
        def add_fib_nodes(n, x=0, y=0, level=0, parent_x=None, parent_y=None):
            if level > depth or n <= 1:
                return
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[f'fib({n})'],
                textposition='middle center',
                marker=dict(size=30, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
            
            # Add edge from parent
            if parent_x is not None:
                fig.add_trace(go.Scatter(
                    x=[parent_x, x], y=[parent_y, y],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))
            
            # Add child nodes
            offset = 2 ** (depth - level - 1)
            add_fib_nodes(n-1, x-offset, y-1, level+1, x, y)
            add_fib_nodes(n-2, x+offset, y-1, level+1, x, y)
        
        add_fib_nodes(5)
    
    fig.update_layout(
        title=f"Recursion Tree - {func_name}",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_bit_operations(num1, num2, operation):
    """Visualize bit manipulation operations"""
    fig = go.Figure()
    
    # Convert to binary representation
    bin1 = format(num1, '08b')
    bin2 = format(num2, '08b')
    
    if operation == "AND":
        result = num1 & num2
    elif operation == "OR":
        result = num1 | num2
    elif operation == "XOR":
        result = num1 ^ num2
    else:
        result = num1
    
    result_bin = format(result, '08b')
    
    # Create visualization
    positions = list(range(8))
    
    fig.add_trace(go.Bar(
        x=positions, y=[1]*8,
        text=list(bin1),
        textposition='auto',
        name=f'Number 1 ({num1})',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[2]*8,
        text=list(bin2),
        textposition='auto',
        name=f'Number 2 ({num2})',
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[3]*8,
        text=list(result_bin),
        textposition='auto',
        name=f'Result ({result})',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f"Bit {operation} Operation",
        xaxis_title="Bit Position",
        yaxis=dict(showticklabels=False),
        height=400
    )
    
    return fig

def visualize_dp_table(dp_table, problem_name):
    """Visualize DP table"""
    fig = go.Figure(data=go.Heatmap(
        z=dp_table,
        colorscale='Viridis',
        showscale=True,
        text=dp_table,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f"DP Table - {problem_name}",
        height=400
    )
    
    return fig

# --- BADGE SYSTEM ---
BADGES = [
    {"name": "First Solve", "desc": "Solved your first problem!", "criteria": lambda s: s >= 1, "emoji": "🥇"},
    {"name": "Speedster", "desc": "Solved a problem in under 60s!", "criteria": lambda s: any(p.get('time_taken', 9999) < 60 for p in st.session_state.solved_problems), "emoji": "⚡"},
    {"name": "Hintless Hero", "desc": "Solved a problem without hints!", "criteria": lambda s: any(p.get('hints_used', 1e9) == 0 for p in st.session_state.solved_problems), "emoji": "🦸"},
    {"name": "DSA Explorer", "desc": "Solved problems in 3+ topics!", "criteria": lambda s: len(set(p['title'].split('-')[0].strip() for p in st.session_state.solved_problems)) >= 3, "emoji": "🧭"},
    {"name": "Nightmare Conqueror", "desc": "Solved a Nightmare problem!", "criteria": lambda s: any(p.get('difficulty', '') == 'Nightmare' for p in st.session_state.solved_problems), "emoji": "👹"},
    {"name": "Persistence", "desc": "Solved 10+ problems!", "criteria": lambda s: s >= 10, "emoji": "🏆"},
]

def get_earned_badges():
    solved = len(st.session_state.solved_problems)
    badges = []
    for badge in BADGES:
        # Fix: Always pass the correct type to the criteria lambda
        # If the lambda expects a count, pass solved; if it expects a list, pass the list
        import inspect
        try:
            # Check if lambda expects a list (by checking if it uses 'for' or 'any'/'len')
            # We'll use the function's code object to check argument names
            params = inspect.signature(badge["criteria"]).parameters
            # If the lambda expects a list, pass the list; else, pass the count
            # We'll use a heuristic: if the lambda's code contains 'for' or 'any', pass the list
            src = badge["criteria"].__code__.co_code
            # Actually, let's use the description as a hint (as in the original code)
            if "problems" in badge["desc"].lower():
                arg = st.session_state.solved_problems
            else:
                arg = solved
            if badge["criteria"](arg):
                badges.append(badge)
        except Exception:
            # fallback: try both
            try:
                if badge["criteria"](solved):
                    badges.append(badge)
            except Exception:
                try:
                    if badge["criteria"](st.session_state.solved_problems):
                        badges.append(badge)
                except Exception:
                    pass
    return badges

# --- CODE EXECUTION (PYTHON ONLY, SAFE) ---
def safe_run_python(user_code, input_str):
    """Safely execute user Python code and capture output."""
    try:
        # Prepare a local namespace
        local_ns = {}
        # Redirect stdout
        with contextlib.redirect_stdout(io.StringIO()) as f:
            # Prepare input() mocking
            input_lines = input_str.strip().split('\n')
            input_iter = iter(input_lines)
            def input_mock(prompt=''):
                return next(input_iter)
            # Patch builtins
            import builtins
            real_input = builtins.input
            builtins.input = input_mock
            try:
                exec(user_code, {}, local_ns)
            finally:
                builtins.input = real_input
        output = f.getvalue().strip()
        return output, None
    except Exception as e:
        return "", str(e)

def compare_outputs(user_out, expected_out):
    """Compare outputs, ignoring whitespace differences."""
    return user_out.strip() == expected_out.strip()

def main():
    st.title("🧠 DSA Mastery - AI-Powered Learning")
    st.markdown("*Master Data Structures and Algorithms with AI guidance*")
    st.info("Welcome! Ready to level up your DSA skills? 🚀")

    # Initialize Gemini
    try:
        model = init_gemini()
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        st.stop()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🎯 Problem Selection")
        
        # Language selection
        languages = {
            "Python": "python",
            "Java": "java", 
            "C++": "cpp",
            "JavaScript": "javascript",
            "C": "c",
            "Go": "go",
            "Rust": "rust",
            "C#": "csharp"
        }
        
        selected_language = st.selectbox(
            "Choose Programming Language:",
            list(languages.keys()),
            index=0
        )
        
        # Topic selection
        selected_topic = st.selectbox(
            "Choose Data Structure/Algorithm:",
            list(DS_TOPICS.keys()),
            index=0
        )
        
        # Subtopic selection
        if selected_topic:
            subtopic = st.selectbox(
                "Choose Subtopic:",
                DS_TOPICS[selected_topic],
                index=0
            )
        
        # Difficulty selection
        difficulty = st.selectbox(
            "Select Difficulty:",
            DIFFICULTY_LEVELS,
            index=1
        )
        
        # Generate problem button
        if st.button("🎲 Generate New Problem", type="primary"):
            with st.spinner("Generating problem..."):
                problem = generate_problem(f"{selected_topic} - {subtopic}", difficulty, selected_language, model)
                if "error" not in problem:
                    st.session_state.current_problem = problem
                    st.session_state.current_language = selected_language
                    st.session_state.start_time = time.time()
                    st.session_state.hint_count = 0
                    st.session_state.user_solution = get_starter_template(languages[selected_language])
                    st.rerun()
                else:
                    st.error(problem["error"])
        
        # Stats section
        st.header("📊 Your Stats")
        st.metric("Problems Solved", len(st.session_state.solved_problems))
        if st.session_state.solved_problems:
            avg_time = sum(p.get('time_taken', 0) for p in st.session_state.solved_problems) / len(st.session_state.solved_problems)
            st.metric("Avg Time", f"{avg_time:.1f}s")
        
        # --- BADGES DISPLAY ---
        st.header("🏅 Badges")
        badges = get_earned_badges()
        if badges:
            for badge in badges:
                st.markdown(f"{badge['emoji']} **{badge['name']}**: {badge['desc']}")
        else:
            st.caption("No badges yet. Start solving to earn some! 🌟")

    # Main content area
    if st.session_state.current_problem:
        problem = st.session_state.current_problem

        # Timer
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.metric("⏱️ Time Elapsed", f"{elapsed:.1f}s")
        
        # Problem display
        st.header(f"📝 {problem['title']}")
        st.success("Good luck! You can do it! 💪")

        # Problem tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Problem", "Solution", "Hints", "Visualization"])
        
        with tab1:
            st.markdown("### Problem Description")
            st.write(problem['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Format:**")
                st.code(problem['input_format'])
                st.markdown("**Example Input:**")
                st.code(problem['example_input'])
            
            with col2:
                st.markdown("**Output Format:**")
                st.code(problem['output_format'])
                st.markdown("**Example Output:**")
                st.code(problem['example_output'])
            
            st.markdown("**Constraints:**")
            st.write(problem['constraints'])
            
            # Code editor
            st.markdown("### 💻 Your Solution")
            language_key = languages[st.session_state.get('current_language', 'Python')]
            
            # Display starter code if available
            if 'starter_code' in problem:
                st.markdown("**Starter Code:**")
                st.code(problem['starter_code'], language=language_key)
            
            user_code = st.text_area(
                "Write your solution here:",
                value=st.session_state.user_solution,
                height=300,
                placeholder=get_starter_template(language_key),
                help=f"Write your solution in {st.session_state.get('current_language', 'Python')}"
            )
            st.session_state.user_solution = user_code
            
            # --- OUTPUT DISPLAY ---
            st.markdown("### 🖨️ Output Checker")
            output_placeholder = st.empty()
            feedback_placeholder = st.empty()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏃‍♂️ Run Code"):
                    if language_key == "python":
                        user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                        if err:
                            output_placeholder.error(f"Error: {err}")
                        else:
                            output_placeholder.code(user_out, language="text")
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")
                    else:
                        st.info("For non-Python languages, please run your code in your local environment or an online compiler.")
                        user_out = st.text_area(
                            "Paste your program's output here for checking:",
                            value="",
                            key="manual_output_check",
                            height=100,
                            help="Copy the output from your compiler/interpreter and paste here."
                        )
                        if st.button("Check Output", key="check_output_btn"):
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")

            with col2:
                if st.button("✅ Submit"):
                    if user_code.strip():
                        correct = False
                        if language_key == "python":
                            user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                            expected = problem.get('example_output', '').strip()
                            if not err and compare_outputs(user_out, expected):
                                correct = True
                        else:
                            # For non-Python, check if user pasted correct output
                            user_out = st.session_state.get("manual_output_check", "")
                            expected = problem.get('example_output', '').strip()
                            if user_out and compare_outputs(user_out, expected):
                                correct = True
                            elif user_out:
                                correct = False
                            else:
                                correct = None  # Not checked

                        time_taken = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                        st.session_state.solved_problems.append({
                            'title': problem['title'],
                            'difficulty': st.session_state.get('current_difficulty', 'Unknown'),
                            'time_taken': time_taken,
                            'hints_used': st.session_state.hint_count
                        })
                        if correct is True:
                            st.balloons()
                            st.success(f"🎉 Correct! Problem submitted! Time: {time_taken:.1f}s")
                            st.info("You've earned a badge? Check the sidebar! 🏅")
                        elif correct is False:
                            st.warning("❌ Output is incorrect. Try again or use a hint!")
                        else:
                            st.info("Submission recorded! (Output not auto-checked for this language)")
                    else:
                        st.warning("Please write a solution first")

            with col3:
                if st.button("🔄 Reset"):
                    st.session_state.user_solution = ""
                    st.session_state.hint_count = 0
                    st.rerun()
        
        with tab2:
            st.markdown("### 🎯 Solution Approach")
            st.write(problem['solution_approach'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Time Complexity:**")
                st.code(problem['time_complexity'])
            with col2:
                st.markdown("**Space Complexity:**")
                st.code(problem['space_complexity'])
            
            # Display solution code if available
            if 'solution_code' in problem:
                st.markdown("**Complete Solution:**")
                language_key = languages[st.session_state.get('current_language', 'Python')]
                st.code(problem['solution_code'], language=language_key)
            
            # Generate detailed solution
            if st.button("🔍 Get Detailed Solution"):
                with st.spinner("Generating detailed solution..."):
                    solution_prompt = f"""
                    Provide a detailed solution for this problem in {st.session_state.get('current_language', 'Python')}:
                    {problem['title']}
                    {problem['description']}
                    
                    Include:
                    1. Step-by-step explanation
                    2. Complete code solution
                    3. Explanation of the approach
                    """
                    try:
                        response = model.generate_content(solution_prompt)
                        st.markdown("### 📚 Detailed Solution")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating solution: {str(e)}")
        
        with tab3:
            st.markdown("### 💡 Hints")
            
            for i, hint in enumerate(problem['hints']):
                if st.button(f"Reveal Hint {i+1}", key=f"hint_{i}"):
                    st.info(f"**Hint {i+1}:** {hint}")
                    st.session_state.hint_count = max(st.session_state.hint_count, i+1)
            
            # AI-powered personalized hint
            if st.button("🤖 Get AI Hint"):
                hint_prompt = f"""
                User is solving: {problem['title']}
                Their current code: {st.session_state.user_solution}
                
                Provide a helpful hint without giving away the complete solution.
                Focus on the next logical step they should take.
                """
                try:
                    response = model.generate_content(hint_prompt)
                    st.success(f"**AI Hint:** {response.text}")
                    st.session_state.hint_count += 1
                except Exception as e:
                    st.error(f"Error generating hint: {str(e)}")
        
        with tab4:
            st.markdown("### 📊 Algorithm Visualization")
            
            # Topic-specific visualizations
            if "array" in selected_topic.lower():
                sample_array = [64, 34, 25, 12, 22, 11, 90]
                fig = visualize_array(sample_array, "Sample Array")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🎬 Visualize Sorting"):
                    steps = visualize_sorting_steps(sample_array)
                    progress_bar = st.progress(0)
                    chart = st.empty()
                    
                    for i, step in enumerate(steps):
                        fig = visualize_array(step, f"Sorting Step {i+1}")
                        chart.plotly_chart(fig, use_container_width=True)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
            
            elif "binary tree" in selected_topic.lower():
                sample_tree = [1, 2, 3, 4, 5, 6, 7]
                fig = visualize_binary_tree(sample_tree, "Binary Tree Structure")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🌳 Show Tree Traversals"):
                    st.code("Inorder: 4 2 5 1 6 3 7\nPreorder: 1 2 4 5 3 6 7\nPostorder: 4 5 2 6 7 3 1")
            
            elif "bst" in selected_topic.lower() or "binary search tree" in selected_topic.lower():
                sample_bst = [4, 2, 6, 1, 3, 5, 7]
                fig = visualize_binary_tree(sample_bst, "Binary Search Tree")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("BST Property: Left < Root < Right")
            
            elif "heap" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    min_heap = [1, 3, 6, 5, 9, 8]
                    fig = visualize_heap(min_heap, "min")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    max_heap = [9, 8, 6, 5, 3, 1]
                    fig = visualize_heap(max_heap, "max")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif "recursion" in selected_topic.lower():
                fig = visualize_recursion_tree("fibonacci")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🔄 Show Call Stack"):
                    st.code("""
Call Stack Visualization:
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2) → 1
│   │   └── fib(1) → 1
│   └── fib(2) → 1
└── fib(3) → 2
                    """)
            
            elif "bit manipulation" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    num1 = st.number_input("Number 1", value=12, min_value=0, max_value=255)
                with col2:
                    num2 = st.number_input("Number 2", value=10, min_value=0, max_value=255)
                
                operation = st.selectbox("Operation", ["AND", "OR", "XOR"])
                fig = visualize_bit_operations(num1, num2, operation)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "dynamic programming" in selected_topic.lower() or "dp" in selected_topic.lower():
                # Example DP table for fibonacci
                dp_table = [[0, 1, 1, 2, 3, 5, 8, 13]]
                fig = visualize_dp_table(dp_table, "Fibonacci DP")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("DP Table shows memoized results to avoid recomputation")
            
            elif "hashing" in selected_topic.lower():
                # Hash table visualization
                hash_data = {
                    'Key': ['apple', 'banana', 'cherry', 'date'],
                    'Hash': [hash('apple') % 10, hash('banana') % 10, hash('cherry') % 10, hash('date') % 10],
                    'Value': [5, 7, 3, 9]
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=hash_data['Key'], y=hash_data['Hash'], name='Hash Values'),
                ])
                fig.update_layout(title="Hash Function Visualization", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "backtracking" in selected_topic.lower():
                st.markdown("### 🔄 Backtracking Visualization")
                st.code("""
N-Queens Solution Tree:
Try Queen at (0,0) ✓
├── Try Queen at (1,2) ✓
│   ├── Try Queen at (2,1) ✗ (Conflict!)
│   └── Backtrack...
└── Try Queen at (1,3) ✓
    └── Continue...
                """)
            
            elif "graph" in selected_topic.lower():
                st.info("Graph visualization - BFS/DFS traversal animations")
                # Add graph visualization here
            
            # Interactive complexity analysis
            st.markdown("### 📈 Complexity Analysis")
            
            n_values = list(range(1, 1001, 50))
            complexities = {
                "O(1)": [1] * len(n_values),
                "O(log n)": [math.log2(n) for n in n_values],
                "O(n)": n_values,
                "O(n log n)": [n * math.log2(n) for n in n_values],
                "O(n²)": [n**2 for n in n_values],
            }
            
            complexity_df = pd.DataFrame(complexities, index=n_values)
            st.line_chart(complexity_df)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to DSA Mastery!
        
        Your AI-powered companion for mastering Data Structures and Algorithms.
        
        ### Features:
        - 🎯 **AI-Generated Problems**: Personalized problems based on your skill level
        - ⏱️ **Timed Practice**: Contest-style environment with timers
        - 💡 **Smart Hints**: AI-powered hints that guide without spoiling
        - 📊 **Interactive Visualizations**: See algorithms in action
        - 📈 **Progress Tracking**: Monitor your improvement over time
        
        ### How to Start:
        1. Select a topic from the sidebar
        2. Choose your difficulty level
        3. Click "Generate New Problem"
        4. Start coding and learning!
        
        **Ready to begin your DSA journey?** 👈 Use the sidebar to get started!
        """)
        
        # Display sample visualization
        st.markdown("### 🎨 Sample Visualization")
        sample_data = [3, 7, 1, 9, 4, 6, 8, 2, 5]
        fig = visualize_array(sample_data, "Sample Array - Click Generate Problem to start!")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()import streamlit as st
import google.generativeai as genai
import time
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import math
import io
import contextlib

# Configure page
st.set_page_config(
    page_title="DSA Mastery", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemma-3-27b-it')

# Data structures and topics
DS_TOPICS = {
    "Arrays": ["Basic Operations", "Two Pointers", "Sliding Window", "Prefix Sum"],
    "Strings": ["Pattern Matching", "String Manipulation", "Palindromes", "Anagrams"],
    "Linked Lists": ["Singly Linked List", "Doubly Linked List", "Circular Lists", "Merging"],
    "Stacks": ["Basic Operations", "Expression Evaluation", "Monotonic Stack", "Next Greater Element"],
    "Queues": ["Basic Operations", "Circular Queue", "Priority Queue", "Deque"],
    "Recursion": ["Base Cases", "Recursive Relations", "Tree Recursion", "Tail Recursion"],
    "Backtracking": ["N-Queens", "Sudoku", "Permutations", "Combinations", "Maze Solving"],
    "Binary Trees": ["Tree Traversal", "Tree Construction", "Tree Properties", "Path Problems"],
    "Binary Search Trees": ["BST Operations", "BST Validation", "BST to Array", "Balanced BST"],
    "Heaps": ["Min Heap", "Max Heap", "Heap Sort", "Priority Queue", "K-way Merge"],
    "Hashing": ["Hash Tables", "Hash Functions", "Collision Resolution", "Two Sum Problems"],
    "Dynamic Programming": ["1D DP", "2D DP", "Memoization", "Tabulation", "Classic Problems"],
    "Bit Manipulation": ["Basic Operations", "Bit Tricks", "XOR Properties", "Counting Bits"],
    "Graphs": ["Graph Representation", "BFS", "DFS", "Shortest Path"],
    "Greedy": ["Activity Selection", "Huffman Coding", "Fractional Knapsack", "MST"],
    "Divide & Conquer": ["Merge Sort", "Quick Sort", "Binary Search", "Closest Pair"],
    "Trie": ["Prefix Tree", "Auto-complete", "Word Search", "Longest Prefix"],
    "Segment Trees": ["Range Query", "Range Update", "Lazy Propagation", "Applications"]
}

DIFFICULTY_LEVELS = ["Newbie", "Easy", "Medium", "Hard", "Nightmare"]

# Initialize session state
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'user_solution' not in st.session_state:
    st.session_state.user_solution = ""
if 'hint_count' not in st.session_state:
    st.session_state.hint_count = 0
if 'solved_problems' not in st.session_state:
    st.session_state.solved_problems = []

# --- NEW FEATURE: Streak Tracking ---
from datetime import date

if 'last_solved_date' not in st.session_state:
    st.session_state.last_solved_date = None
if 'streak' not in st.session_state:
    st.session_state.streak = 0

def update_streak():
    today = date.today()
    last = st.session_state.last_solved_date
    if last is None or (today - last).days > 1:
        st.session_state.streak = 1
    elif (today - last).days == 1:
        st.session_state.streak += 1
    # else: same day, don't increment
    st.session_state.last_solved_date = today

# --- NEW FEATURE: Leaderboard (local session only) ---
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = []

def update_leaderboard(username, problems_solved):
    found = False
    for entry in st.session_state.leaderboard:
        if entry['username'] == username:
            entry['problems_solved'] = problems_solved
            found = True
            break
    if not found:
        st.session_state.leaderboard.append({'username': username, 'problems_solved': problems_solved})
    st.session_state.leaderboard.sort(key=lambda x: x['problems_solved'], reverse=True)

# --- NEW FEATURE: Problem Bookmarking ---
if 'bookmarked_problems' not in st.session_state:
    st.session_state.bookmarked_problems = []

def bookmark_problem(problem):
    if problem and problem not in st.session_state.bookmarked_problems:
        st.session_state.bookmarked_problems.append(problem)

# --- NEW FEATURE: Problem Retry History ---
if 'retry_history' not in st.session_state:
    st.session_state.retry_history = {}

def add_retry(problem_title):
    st.session_state.retry_history[problem_title] = st.session_state.retry_history.get(problem_title, 0) + 1

# --- NEW FEATURE: Custom Input for Code Testing ---
def safe_run_python_custom(user_code, input_str):
    return safe_run_python(user_code, input_str)

# --- NEW FEATURE: Export Solved Problems as CSV ---
def export_solved_problems_csv():
    if not st.session_state.solved_problems:
        return ""
    df = pd.DataFrame(st.session_state.solved_problems)
    return df.to_csv(index=False)

# --- NEW FEATURE: Theme Toggle ---
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# --- NEW FEATURE: Daily Challenge ---
if 'daily_challenge' not in st.session_state:
    st.session_state.daily_challenge = None
if 'daily_challenge_date' not in st.session_state:
    st.session_state.daily_challenge_date = None

def set_daily_challenge(model):
    today = date.today()
    if st.session_state.daily_challenge_date != today:
        # Pick a random topic/difficulty/language
        topic = random.choice(list(DS_TOPICS.keys()))
        subtopic = random.choice(DS_TOPICS[topic])
        difficulty = random.choice(DIFFICULTY_LEVELS)
        language = "Python"
        st.session_state.daily_challenge = generate_problem(f"{topic} - {subtopic}", difficulty, language, model)
        st.session_state.daily_challenge_date = today

# --- NEW FEATURE: Problem Rating ---
if 'problem_ratings' not in st.session_state:
    st.session_state.problem_ratings = {}

def rate_problem(title, rating):
    st.session_state.problem_ratings[title] = rating

# --- NEW FEATURE: Motivational Quotes ---
MOTIVATIONAL_QUOTES = [
    "Every expert was once a beginner.",
    "Success is the sum of small efforts, repeated.",
    "Don't watch the clock; do what it does. Keep going.",
    "The harder you work for something, the greater you'll feel when you achieve it.",
    "Push yourself, because no one else is going to do it for you.",
    "Great things never come from comfort zones.",
    "Dream it. Wish it. Do it.",
    "Success doesn’t just find you. You have to go out and get it.",
    "The key to success is to focus on goals, not obstacles.",
    "Believe you can and you're halfway there."
]
def get_random_quote():
    return random.choice(MOTIVATIONAL_QUOTES)

def generate_problem(topic, difficulty, language, model):
    """Generate a coding problem using Gemini"""
    prompt = f"""
    Generate a {difficulty} level coding problem for {topic} in {language}.
    
    Return the response in this exact JSON format:
    {{
        "title": "Problem Title",
        "description": "Detailed problem description with examples",
        "input_format": "Input format description",
        "output_format": "Output format description",
        "constraints": "Problem constraints",
        "example_input": "Sample input",
        "example_output": "Sample output",
        "hints": ["Hint 1", "Hint 2", "Hint 3"],
        "solution_approach": "High-level approach to solve",
        "time_complexity": "Expected time complexity",
        "space_complexity": "Expected space complexity",
        "starter_code": "Starter code template in {language}",
        "solution_code": "Complete solution in {language}"
    }}
    
    Make sure the problem is educational and focuses on understanding {topic} concepts.
    Provide language-specific starter code and solution.
    """
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Failed to parse response"}
    except Exception as e:
        return {"error": f"Error generating problem: {str(e)}"}

def get_code_editor_theme(language):
    """Get appropriate syntax highlighting for code editor"""
    theme_map = {
        "python": "python",
        "java": "java",
        "cpp": "cpp",
        "javascript": "javascript", 
        "c": "c",
        "go": "go",
        "rust": "rust",
        "csharp": "csharp"
    }
    return theme_map.get(language, "python")

def get_starter_template(language):
    """Get language-specific starter templates"""
    templates = {
        "python": "# Write your solution here\ndef solve():\n    pass\n\n# Test your solution\nif __name__ == '__main__':\n    pass",
        "java": "public class Solution {\n    public static void main(String[] args) {\n        // Write your solution here\n    }\n}",
        "cpp": "#include <iostream>\n#include <vector>\nusing namespace std;\n\nint main() {\n    // Write your solution here\n    return 0;\n}",
        "javascript": "// Write your solution here\nfunction solve() {\n    \n}\n\n// Test your solution\nconsole.log(solve());",
        "c": "#include <stdio.h>\n#include <stdlib.h>\n\nint main() {\n    // Write your solution here\n    return 0;\n}",
        "go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    // Write your solution here\n}",
        "rust": "fn main() {\n    // Write your solution here\n}\n\nfn solve() {\n    \n}",
        "csharp": "using System;\n\nclass Program {\n    static void Main() {\n        // Write your solution here\n    }\n}"
    }
    return templates.get(language, templates["python"])
def visualize_array(arr, title="Array Visualization"):
    """Create interactive array visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(arr))),
        y=arr,
        text=arr,
        textposition='auto',
        marker_color='lightblue',
        marker_line_color='navy',
        marker_line_width=2
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        height=400
    )
    
    return fig

def visualize_sorting_steps(arr, algorithm="bubble_sort"):
    """Visualize sorting algorithm steps"""
    steps = []
    arr_copy = arr.copy()
    
    if algorithm == "bubble_sort":
        n = len(arr_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr_copy[j] > arr_copy[j + 1]:
                    arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                    steps.append(arr_copy.copy())
    
    return steps

def visualize_binary_tree(nodes, title="Binary Tree"):
    """Create binary tree visualization"""
    if not nodes:
        return go.Figure()
    
    # Calculate positions for binary tree layout
    fig = go.Figure()
    
    # Simple binary tree visualization (can be expanded)
    levels = {}
    for i, node in enumerate(nodes):
        level = int(math.log2(i + 1)) if i > 0 else 0
        if level not in levels:
            levels[level] = []
        levels[level].append((i, node))
    
    # Add nodes
    for level, level_nodes in levels.items():
        y_pos = -level
        for i, (idx, value) in enumerate(level_nodes):
            x_pos = (i - len(level_nodes)/2) * (4 - level)
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[y_pos],
                mode='markers+text',
                text=[str(value)],
                textposition='middle center',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_heap(arr, heap_type="min"):
    """Visualize heap structure"""
    fig = go.Figure()
    
    # Create heap visualization
    for i in range(len(arr)):
        level = int(math.log2(i + 1)) if i > 0 else 0
        y_pos = -level
        x_pos = (i - (2**level - 1)) * (8 / (2**level))
        
        color = 'lightgreen' if heap_type == 'min' else 'lightcoral'
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos],
            mode='markers+text',
            text=[str(arr[i])],
            textposition='middle center',
            marker=dict(size=40, color=color, line=dict(width=2, color='darkgreen' if heap_type == 'min' else 'darkred')),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{heap_type.capitalize()} Heap Visualization",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_recursion_tree(func_name, depth=4):
    """Create recursion tree visualization"""
    fig = go.Figure()
    
    # Example for fibonacci recursion
    if func_name == "fibonacci":
        def add_fib_nodes(n, x=0, y=0, level=0, parent_x=None, parent_y=None):
            if level > depth or n <= 1:
                return
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[f'fib({n})'],
                textposition='middle center',
                marker=dict(size=30, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
            
            # Add edge from parent
            if parent_x is not None:
                fig.add_trace(go.Scatter(
                    x=[parent_x, x], y=[parent_y, y],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))
            
            # Add child nodes
            offset = 2 ** (depth - level - 1)
            add_fib_nodes(n-1, x-offset, y-1, level+1, x, y)
            add_fib_nodes(n-2, x+offset, y-1, level+1, x, y)
        
        add_fib_nodes(5)
    
    fig.update_layout(
        title=f"Recursion Tree - {func_name}",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_bit_operations(num1, num2, operation):
    """Visualize bit manipulation operations"""
    fig = go.Figure()
    
    # Convert to binary representation
    bin1 = format(num1, '08b')
    bin2 = format(num2, '08b')
    
    if operation == "AND":
        result = num1 & num2
    elif operation == "OR":
        result = num1 | num2
    elif operation == "XOR":
        result = num1 ^ num2
    else:
        result = num1
    
    result_bin = format(result, '08b')
    
    # Create visualization
    positions = list(range(8))
    
    fig.add_trace(go.Bar(
        x=positions, y=[1]*8,
        text=list(bin1),
        textposition='auto',
        name=f'Number 1 ({num1})',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[2]*8,
        text=list(bin2),
        textposition='auto',
        name=f'Number 2 ({num2})',
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[3]*8,
        text=list(result_bin),
        textposition='auto',
        name=f'Result ({result})',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f"Bit {operation} Operation",
        xaxis_title="Bit Position",
        yaxis=dict(showticklabels=False),
        height=400
    )
    
    return fig

def visualize_dp_table(dp_table, problem_name):
    """Visualize DP table"""
    fig = go.Figure(data=go.Heatmap(
        z=dp_table,
        colorscale='Viridis',
        showscale=True,
        text=dp_table,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f"DP Table - {problem_name}",
        height=400
    )
    
    return fig

# --- BADGE SYSTEM ---
BADGES = [
    {"name": "First Solve", "desc": "Solved your first problem!", "criteria": lambda s: s >= 1, "emoji": "🥇"},
    {"name": "Speedster", "desc": "Solved a problem in under 60s!", "criteria": lambda s: any(p.get('time_taken', 9999) < 60 for p in st.session_state.solved_problems), "emoji": "⚡"},
    {"name": "Hintless Hero", "desc": "Solved a problem without hints!", "criteria": lambda s: any(p.get('hints_used', 1e9) == 0 for p in st.session_state.solved_problems), "emoji": "🦸"},
    {"name": "DSA Explorer", "desc": "Solved problems in 3+ topics!", "criteria": lambda s: len(set(p['title'].split('-')[0].strip() for p in st.session_state.solved_problems)) >= 3, "emoji": "🧭"},
    {"name": "Nightmare Conqueror", "desc": "Solved a Nightmare problem!", "criteria": lambda s: any(p.get('difficulty', '') == 'Nightmare' for p in st.session_state.solved_problems), "emoji": "👹"},
    {"name": "Persistence", "desc": "Solved 10+ problems!", "criteria": lambda s: s >= 10, "emoji": "🏆"},
]

def get_earned_badges():
    solved = len(st.session_state.solved_problems)
    badges = []
    for badge in BADGES:
        # Fix: Always pass the correct type to the criteria lambda
        # If the lambda expects a count, pass solved; if it expects a list, pass the list
        import inspect
        try:
            # Check if lambda expects a list (by checking if it uses 'for' or 'any'/'len')
            # We'll use the function's code object to check argument names
            params = inspect.signature(badge["criteria"]).parameters
            # If the lambda expects a list, pass the list; else, pass the count
            # We'll use a heuristic: if the lambda's code contains 'for' or 'any', pass the list
            src = badge["criteria"].__code__.co_code
            # Actually, let's use the description as a hint (as in the original code)
            if "problems" in badge["desc"].lower():
                arg = st.session_state.solved_problems
            else:
                arg = solved
            if badge["criteria"](arg):
                badges.append(badge)
        except Exception:
            # fallback: try both
            try:
                if badge["criteria"](solved):
                    badges.append(badge)
            except Exception:
                try:
                    if badge["criteria"](st.session_state.solved_problems):
                        badges.append(badge)
                except Exception:
                    pass
    return badges

# --- CODE EXECUTION (PYTHON ONLY, SAFE) ---
def safe_run_python(user_code, input_str):
    """Safely execute user Python code and capture output."""
    try:
        # Prepare a local namespace
        local_ns = {}
        # Redirect stdout
        with contextlib.redirect_stdout(io.StringIO()) as f:
            # Prepare input() mocking
            input_lines = input_str.strip().split('\n')
            input_iter = iter(input_lines)
            def input_mock(prompt=''):
                return next(input_iter)
            # Patch builtins
            import builtins
            real_input = builtins.input
            builtins.input = input_mock
            try:
                exec(user_code, {}, local_ns)
            finally:
                builtins.input = real_input
        output = f.getvalue().strip()
        return output, None
    except Exception as e:
        return "", str(e)

def compare_outputs(user_out, expected_out):
    """Compare outputs, ignoring whitespace differences."""
    return user_out.strip() == expected_out.strip()

def main():
    st.title("🧠 DSA Mastery - AI-Powered Learning")
    st.markdown("*Master Data Structures and Algorithms with AI guidance*")
    st.info("Welcome! Ready to level up your DSA skills? 🚀")

    # Initialize Gemini
    try:
        model = init_gemini()
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        st.stop()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🎯 Problem Selection")
        
        # Language selection
        languages = {
            "Python": "python",
            "Java": "java", 
            "C++": "cpp",
            "JavaScript": "javascript",
            "C": "c",
            "Go": "go",
            "Rust": "rust",
            "C#": "csharp"
        }
        
        selected_language = st.selectbox(
            "Choose Programming Language:",
            list(languages.keys()),
            index=0
        )
        
        # Topic selection
        selected_topic = st.selectbox(
            "Choose Data Structure/Algorithm:",
            list(DS_TOPICS.keys()),
            index=0
        )
        
        # Subtopic selection
        if selected_topic:
            subtopic = st.selectbox(
                "Choose Subtopic:",
                DS_TOPICS[selected_topic],
                index=0
            )
        
        # Difficulty selection
        difficulty = st.selectbox(
            "Select Difficulty:",
            DIFFICULTY_LEVELS,
            index=1
        )
        
        # Generate problem button
        if st.button("🎲 Generate New Problem", type="primary"):
            with st.spinner("Generating problem..."):
                problem = generate_problem(f"{selected_topic} - {subtopic}", difficulty, selected_language, model)
                if "error" not in problem:
                    st.session_state.current_problem = problem
                    st.session_state.current_language = selected_language
                    st.session_state.start_time = time.time()
                    st.session_state.hint_count = 0
                    st.session_state.user_solution = get_starter_template(languages[selected_language])
                    st.rerun()
                else:
                    st.error(problem["error"])
        
        # Stats section
        st.header("📊 Your Stats")
        st.metric("Problems Solved", len(st.session_state.solved_problems))
        if st.session_state.solved_problems:
            avg_time = sum(p.get('time_taken', 0) for p in st.session_state.solved_problems) / len(st.session_state.solved_problems)
            st.metric("Avg Time", f"{avg_time:.1f}s")
        
        # --- BADGES DISPLAY ---
        st.header("🏅 Badges")
        badges = get_earned_badges()
        if badges:
            for badge in badges:
                st.markdown(f"{badge['emoji']} **{badge['name']}**: {badge['desc']}")
        else:
            st.caption("No badges yet. Start solving to earn some! 🌟")

    # --- THEME TOGGLE BUTTON ---
    theme_col1, theme_col2 = st.columns([1, 8])
    with theme_col1:
        if st.button(f"Switch to {'🌙 Dark' if st.session_state.theme == 'light' else '☀️ Light'} Theme"):
            toggle_theme()
            st.rerun()
    with theme_col2:
        st.caption(f"Theme: {st.session_state.theme.capitalize()}")

    # --- MOTIVATIONAL QUOTE ---
    st.info(f"💡 Motivation: {get_random_quote()}")

    # --- DAILY CHALLENGE ---
    set_daily_challenge(init_gemini())
    if st.session_state.daily_challenge:
        with st.expander("🔥 Daily Challenge"):
            st.markdown(f"**{st.session_state.daily_challenge['title']}**")
            st.write(st.session_state.daily_challenge['description'])
            if st.button("Try Daily Challenge"):
                st.session_state.current_problem = st.session_state.daily_challenge
                st.session_state.current_language = "Python"
                st.session_state.start_time = time.time()
                st.session_state.hint_count = 0
                st.session_state.user_solution = get_starter_template("python")
                st.rerun()

    # --- LEADERBOARD ---
    with st.sidebar:
        st.header("🏆 Leaderboard")
        username = st.text_input("Your Name", value="You")
        update_leaderboard(username, len(st.session_state.solved_problems))
        for i, entry in enumerate(st.session_state.leaderboard[:5]):
            st.markdown(f"{i+1}. **{entry['username']}** - {entry['problems_solved']} solved")

    # --- STREAK DISPLAY ---
    with st.sidebar:
        st.header("🔥 Streak")
        st.metric("Current Streak", st.session_state.streak)
        if st.session_state.streak > 0:
            st.caption(f"Keep it up! {st.session_state.streak} days in a row!")

    # --- EXPORT SOLVED PROBLEMS ---
    with st.sidebar:
        st.header("⬇️ Export")
        if st.button("Export Solved Problems (CSV)"):
            csv = export_solved_problems_csv()
            st.download_button("Download CSV", csv, file_name="solved_problems.csv", mime="text/csv")

    # --- BOOKMARKED PROBLEMS ---
    with st.sidebar:
        st.header("🔖 Bookmarks")
        if st.session_state.bookmarked_problems:
            for p in st.session_state.bookmarked_problems:
                st.markdown(f"- {p['title']}")
        else:
            st.caption("No bookmarks yet.")

    # Main content area
    if st.session_state.current_problem:
        problem = st.session_state.current_problem

        # Timer
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.metric("⏱️ Time Elapsed", f"{elapsed:.1f}s")
        
        # Problem display
        st.header(f"📝 {problem['title']}")
        st.success("Good luck! You can do it! 💪")

        # Problem tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Problem", "Solution", "Hints", "Visualization"])
        
        with tab1:
            st.markdown("### Problem Description")
            st.write(problem['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Format:**")
                st.code(problem['input_format'])
                st.markdown("**Example Input:**")
                st.code(problem['example_input'])
            
            with col2:
                st.markdown("**Output Format:**")
                st.code(problem['output_format'])
                st.markdown("**Example Output:**")
                st.code(problem['example_output'])
            
            st.markdown("**Constraints:**")
            st.write(problem['constraints'])
            
            # Code editor
            st.markdown("### 💻 Your Solution")
            language_key = languages[st.session_state.get('current_language', 'Python')]
            
            # Display starter code if available
            if 'starter_code' in problem:
                st.markdown("**Starter Code:**")
                st.code(problem['starter_code'], language=language_key)
            
            user_code = st.text_area(
                "Write your solution here:",
                value=st.session_state.user_solution,
                height=300,
                placeholder=get_starter_template(language_key),
                help=f"Write your solution in {st.session_state.get('current_language', 'Python')}",
                key="user_solution_text_areaaaa2"
            )
            st.session_state.user_solution = user_code
            
            # --- OUTPUT DISPLAY ---
            st.markdown("### 🖨️ Output Checker")
            output_placeholder = st.empty()
            feedback_placeholder = st.empty()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏃‍♂️ Run Code", key="run_code_btn_tab1"):
                    if language_key == "python":
                        user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                        if err:
                            output_placeholder.error(f"Error: {err}")
                        else:
                            output_placeholder.code(user_out, language="text")
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")
                    else:
                        st.info("For non-Python languages, please run your code in your local environment or an online compiler.")
                        user_out = st.text_area(
                            "Paste your program's output here for checking:",
                            value="",
                            key="manual_output_check",
                            height=100,
                            help="Copy the output from your compiler/interpreter and paste here."
                        )
                        if st.button("Check Output", key="check_output_btn"):
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")

            with col2:
                if st.button("✅ Submit", key="submit_btn_tab1"):
                    if user_code.strip():
                        correct = False
                        if language_key == "python":
                            user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                            expected = problem.get('example_output', '').strip()
                            if not err and compare_outputs(user_out, expected):
                                correct = True
                        else:
                            # For non-Python, check if user pasted correct output
                            user_out = st.session_state.get("manual_output_check", "")
                            expected = problem.get('example_output', '').strip()
                            if user_out and compare_outputs(user_out, expected):
                                correct = True
                            elif user_out:
                                correct = False
                            else:
                                correct = None  # Not checked

                        time_taken = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                        st.session_state.solved_problems.append({
                            'title': problem['title'],
                            'difficulty': st.session_state.get('current_difficulty', 'Unknown'),
                            'time_taken': time_taken,
                            'hints_used': st.session_state.hint_count
                        })
                        if correct is True:
                            update_streak()  # <-- NEW: update streak on correct submission
                            st.balloons()
                            st.success(f"🎉 Correct! Problem submitted! Time: {time_taken:.1f}s")
                            st.info("You've earned a badge? Check the sidebar! 🏅")
                        elif correct is False:
                            st.warning("❌ Output is incorrect. Try again or use a hint!")
                        else:
                            st.info("Submission recorded! (Output not auto-checked for this language)")
                    else:
                        st.warning("Please write a solution first")

            with col3:
                if st.button("🔄 Reset", key="reset_btn_tab1"):
                    st.session_state.user_solution = ""
                    st.session_state.hint_count = 0
                    st.rerun()
        
        with tab2:
            st.markdown("### 🎯 Solution Approach")
            st.write(problem['solution_approach'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Time Complexity:**")
                st.code(problem['time_complexity'])
            with col2:
                st.markdown("**Space Complexity:**")
                st.code(problem['space_complexity'])
            
            # Display solution code if available
            if 'solution_code' in problem:
                st.markdown("**Complete Solution:**")
                language_key = languages[st.session_state.get('current_language', 'Python')]
                st.code(problem['solution_code'], language=language_key)
            
            # Generate detailed solution
            if st.button("🔍 Get Detailed Solution"):
                with st.spinner("Generating detailed solution..."):
                    solution_prompt = f"""
                    Provide a detailed solution for this problem in {st.session_state.get('current_language', 'Python')}:
                    {problem['title']}
                    {problem['description']}
                    
                    Include:
                    1. Step-by-step explanation
                    2. Complete code solution
                    3. Explanation of the approach
                    """
                    try:
                        response = model.generate_content(solution_prompt)
                        st.markdown("### 📚 Detailed Solution")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating solution: {str(e)}")
        
        with tab3:
            st.markdown("### 💡 Hints")
            
            for i, hint in enumerate(problem['hints']):
                if st.button(f"Reveal Hint {i+1}", key=f"hint_{i}"):
                    st.info(f"**Hint {i+1}:** {hint}")
                    st.session_state.hint_count = max(st.session_state.hint_count, i+1)
            
            # AI-powered personalized hint
            if st.button("🤖 Get AI Hint"):
                hint_prompt = f"""
                User is solving: {problem['title']}
                Their current code: {st.session_state.user_solution}
                
                Provide a helpful hint without giving away the complete solution.
                Focus on the next logical step they should take.
                """
                try:
                    response = model.generate_content(hint_prompt)
                    st.success(f"**AI Hint:** {response.text}")
                    st.session_state.hint_count += 1
                except Exception as e:
                    st.error(f"Error generating hint: {str(e)}")
        
        with tab4:
            st.markdown("### 📊 Algorithm Visualization")
            
            # Topic-specific visualizations
            if "array" in selected_topic.lower():
                sample_array = [64, 34, 25, 12, 22, 11, 90]
                fig = visualize_array(sample_array, "Sample Array")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🎬 Visualize Sorting"):
                    steps = visualize_sorting_steps(sample_array)
                    progress_bar = st.progress(0)
                    chart = st.empty()
                    
                    for i, step in enumerate(steps):
                        fig = visualize_array(step, f"Sorting Step {i+1}")
                        chart.plotly_chart(fig, use_container_width=True)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
            
            elif "binary tree" in selected_topic.lower():
                sample_tree = [1, 2, 3, 4, 5, 6, 7]
                fig = visualize_binary_tree(sample_tree, "Binary Tree Structure")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🌳 Show Tree Traversals"):
                    st.code("Inorder: 4 2 5 1 6 3 7\nPreorder: 1 2 4 5 3 6 7\nPostorder: 4 5 2 6 7 3 1")
            
            elif "bst" in selected_topic.lower() or "binary search tree" in selected_topic.lower():
                sample_bst = [4, 2, 6, 1, 3, 5, 7]
                fig = visualize_binary_tree(sample_bst, "Binary Search Tree")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("BST Property: Left < Root < Right")
            
            elif "heap" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    min_heap = [1, 3, 6, 5, 9, 8]
                    fig = visualize_heap(min_heap, "min")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    max_heap = [9, 8, 6, 5, 3, 1]
                    fig = visualize_heap(max_heap, "max")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif "recursion" in selected_topic.lower():
                fig = visualize_recursion_tree("fibonacci")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🔄 Show Call Stack"):
                    st.code("""
Call Stack Visualization:
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2) → 1
│   │   └── fib(1) → 1
│   └── fib(2) → 1
└── fib(3) → 2
                    """)
            
            elif "bit manipulation" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    num1 = st.number_input("Number 1", value=12, min_value=0, max_value=255)
                with col2:
                    num2 = st.number_input("Number 2", value=10, min_value=0, max_value=255)
                
                operation = st.selectbox("Operation", ["AND", "OR", "XOR"])
                fig = visualize_bit_operations(num1, num2, operation)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "dynamic programming" in selected_topic.lower() or "dp" in selected_topic.lower():
                # Example DP table for fibonacci
                dp_table = [[0, 1, 1, 2, 3, 5, 8, 13]]
                fig = visualize_dp_table(dp_table, "Fibonacci DP")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("DP Table shows memoized results to avoid recomputation")
            
            elif "hashing" in selected_topic.lower():
                # Hash table visualization
                hash_data = {
                    'Key': ['apple', 'banana', 'cherry', 'date'],
                    'Hash': [hash('apple') % 10, hash('banana') % 10, hash('cherry') % 10, hash('date') % 10],
                    'Value': [5, 7, 3, 9]
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=hash_data['Key'], y=hash_data['Hash'], name='Hash Values'),
                ])
                fig.update_layout(title="Hash Function Visualization", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "backtracking" in selected_topic.lower():
                st.markdown("### 🔄 Backtracking Visualization")
                st.code("""
N-Queens Solution Tree:
Try Queen at (0,0) ✓
├── Try Queen at (1,2) ✓
│   ├── Try Queen at (2,1) ✗ (Conflict!)
│   └── Backtrack...
└── Try Queen at (1,3) ✓
    └── Continue...
                """)
            
            elif "graph" in selected_topic.lower():
                st.info("Graph visualization - BFS/DFS traversal animations")
                # Add graph visualization here
            
            # Interactive complexity analysis
            st.markdown("### 📈 Complexity Analysis")
            
            n_values = list(range(1, 1001, 50))
            complexities = {
                "O(1)": [1] * len(n_values),
                "O(log n)": [math.log2(n) for n in n_values],
                "O(n)": n_values,
                "O(n log n)": [n * math.log2(n) for n in n_values],
                "O(n²)": [n**2 for n in n_values],
            }
            
            complexity_df = pd.DataFrame(complexities, index=n_values)
            st.line_chart(complexity_df)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to DSA Mastery!
        
        Your AI-powered companion for mastering Data Structures and Algorithms.
        
        ### Features:
        - 🎯 **AI-Generated Problems**: Personalized problems based on your skill level
        - ⏱️ **Timed Practice**: Contest-style environment with timers
        - 💡 **Smart Hints**: AI-powered hints that guide without spoiling
        - 📊 **Interactive Visualizations**: See algorithms in action
        - 📈 **Progress Tracking**: Monitor your improvement over time
        
        ### How to Start:
        1. Select a topic from the sidebar
        2. Choose your difficulty level
        3. Click "Generate New Problem"
        4. Start coding and learning!
        
        **Ready to begin your DSA journey?** 👈 Use the sidebar to get started!
        """)
        
        # Display sample visualization
        st.markdown("### 🎨 Sample Visualization")
        sample_data = [3, 7, 1, 9, 4, 6, 8, 2, 5]
        fig = visualize_array(sample_data, "Sample Array - Click Generate Problem to start!")
        st.plotly_chart(fig, use_container_width=True)

    # --- BOOKMARKED PROBLEMS ---
    with st.sidebar:
        st.header("🔖 Bookmarks")
        if st.session_state.bookmarked_problems:
            for p in st.session_state.bookmarked_problems:
                st.markdown(f"- {p['title']}")
        else:
            st.caption("No bookmarks yet.")

    if st.session_state.current_problem:
        problem = st.session_state.current_problem

        # Timer
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.metric("⏱️ Time Elapsed", f"{elapsed:.1f}s")
        
        # Problem display
        st.header(f"📝 {problem['title']}")
        st.success("Good luck! You can do it! 💪")

        # Problem tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Problem", "Solution", "Hints", "Visualization"])
        
        with tab1:
            st.markdown("### Problem Description")
            st.write(problem['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Format:**")
                st.code(problem['input_format'])
                st.markdown("**Example Input:**")
                st.code(problem['example_input'])
            
            with col2:
                st.markdown("**Output Format:**")
                st.code(problem['output_format'])
                st.markdown("**Example Output:**")
                st.code(problem['example_output'])
            
            st.markdown("**Constraints:**")
            st.write(problem['constraints'])
            
            # Code editor
            st.markdown("### 💻 Your Solution")
            language_key = languages[st.session_state.get('current_language', 'Python')]
            
            # Display starter code if available
            if 'starter_code' in problem:
                st.markdown("**Starter Code:**")
                st.code(problem['starter_code'], language=language_key)
            
            user_code = st.text_area(
                "Write your solution here:",
                value=st.session_state.user_solution,
                height=300,
                placeholder=get_starter_template(language_key),
                help=f"Write your solution in {st.session_state.get('current_language', 'Python')}",
                key="user_solution_text_area"
            )
            st.session_state.user_solution = user_code
            
            # --- OUTPUT DISPLAY ---
            st.markdown("### 🖨️ Output Checker")
            output_placeholder = st.empty()
            feedback_placeholder = st.empty()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏃‍♂️ Run Code", key="run_code_btn_tab1e48jjj"):
                    if language_key == "python":
                        user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                        if err:
                            output_placeholder.error(f"Error: {err}")
                        else:
                            output_placeholder.code(user_out, language="text")
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")
                    else:
                        st.info("For non-Python languages, please run your code in your local environment or an online compiler.")
                        user_out = st.text_area(
                            "Paste your program's output here for checking:",
                            value="",
                            key="manual_output_check",
                            height=100,
                            help="Copy the output from your compiler/interpreter and paste here."
                        )
                        if st.button("Check Output", key="check_output_btn"):
                            expected = problem.get('example_output', '').strip()
                            if compare_outputs(user_out, expected):
                                feedback_placeholder.success("✅ Output matches the example! Well done!")
                            else:
                                feedback_placeholder.warning("⚠️ Output does not match the example. Check your logic.")

            with col2:
                if st.button("✅ Submit", key="submit_btn_tab1"):
                    if user_code.strip():
                        correct = False
                        if language_key == "python":
                            user_out, err = safe_run_python(user_code, problem.get('example_input', ''))
                            expected = problem.get('example_output', '').strip()
                            if not err and compare_outputs(user_out, expected):
                                correct = True
                        else:
                            # For non-Python, check if user pasted correct output
                            user_out = st.session_state.get("manual_output_check", "")
                            expected = problem.get('example_output', '').strip()
                            if user_out and compare_outputs(user_out, expected):
                                correct = True
                            elif user_out:
                                correct = False
                            else:
                                correct = None  # Not checked

                        time_taken = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                        st.session_state.solved_problems.append({
                            'title': problem['title'],
                            'difficulty': st.session_state.get('current_difficulty', 'Unknown'),
                            'time_taken': time_taken,
                            'hints_used': st.session_state.hint_count
                        })
                        if correct is True:
                            update_streak()  # <-- NEW: update streak on correct submission
                            st.balloons()
                            st.success(f"🎉 Correct! Problem submitted! Time: {time_taken:.1f}s")
                            st.info("You've earned a badge? Check the sidebar! 🏅")
                        elif correct is False:
                            st.warning("❌ Output is incorrect. Try again or use a hint!")
                        else:
                            st.info("Submission recorded! (Output not auto-checked for this language)")
                    else:
                        st.warning("Please write a solution first")

            with col3:
                if st.button("🔄 Reset", key="reset_btn_tab1"):
                    st.session_state.user_solution = ""
                    st.session_state.hint_count = 0
                    st.rerun()
        
        with tab2:
            st.markdown("### 🎯 Solution Approach")
            st.write(problem['solution_approach'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Time Complexity:**")
                st.code(problem['time_complexity'])
            with col2:
                st.markdown("**Space Complexity:**")
                st.code(problem['space_complexity'])
            
            # Display solution code if available
            if 'solution_code' in problem:
                st.markdown("**Complete Solution:**")
                language_key = languages[st.session_state.get('current_language', 'Python')]
                st.code(problem['solution_code'], language=language_key)
            
            # Generate detailed solution
            if st.button("🔍 Get Detailed Solution"):
                with st.spinner("Generating detailed solution..."):
                    solution_prompt = f"""
                    Provide a detailed solution for this problem in {st.session_state.get('current_language', 'Python')}:
                    {problem['title']}
                    {problem['description']}
                    
                    Include:
                    1. Step-by-step explanation
                    2. Complete code solution
                    3. Explanation of the approach
                    """
                    try:
                        response = model.generate_content(solution_prompt)
                        st.markdown("### 📚 Detailed Solution")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating solution: {str(e)}")
        
        with tab3:
            st.markdown("### 💡 Hints")
            
            for i, hint in enumerate(problem['hints']):
                if st.button(f"Reveal Hint {i+1}", key=f"hint_{i}"):
                    st.info(f"**Hint {i+1}:** {hint}")
                    st.session_state.hint_count = max(st.session_state.hint_count, i+1)
            
            # AI-powered personalized hint
            if st.button("🤖 Get AI Hint"):
                hint_prompt = f"""
                User is solving: {problem['title']}
                Their current code: {st.session_state.user_solution}
                
                Provide a helpful hint without giving away the complete solution.
                Focus on the next logical step they should take.
                """
                try:
                    response = model.generate_content(hint_prompt)
                    st.success(f"**AI Hint:** {response.text}")
                    st.session_state.hint_count += 1
                except Exception as e:
                    st.error(f"Error generating hint: {str(e)}")
        
        with tab4:
            st.markdown("### 📊 Algorithm Visualization")
            
            # Topic-specific visualizations
            if "array" in selected_topic.lower():
                sample_array = [64, 34, 25, 12, 22, 11, 90]
                fig = visualize_array(sample_array, "Sample Array")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🎬 Visualize Sorting"):
                    steps = visualize_sorting_steps(sample_array)
                    progress_bar = st.progress(0)
                    chart = st.empty()
                    
                    for i, step in enumerate(steps):
                        fig = visualize_array(step, f"Sorting Step {i+1}")
                        chart.plotly_chart(fig, use_container_width=True)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
            
            elif "binary tree" in selected_topic.lower():
                sample_tree = [1, 2, 3, 4, 5, 6, 7]
                fig = visualize_binary_tree(sample_tree, "Binary Tree Structure")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🌳 Show Tree Traversals"):
                    st.code("Inorder: 4 2 5 1 6 3 7\nPreorder: 1 2 4 5 3 6 7\nPostorder: 4 5 2 6 7 3 1")
            
            elif "bst" in selected_topic.lower() or "binary search tree" in selected_topic.lower():
                sample_bst = [4, 2, 6, 1, 3, 5, 7]
                fig = visualize_binary_tree(sample_bst, "Binary Search Tree")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("BST Property: Left < Root < Right")
            
            elif "heap" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    min_heap = [1, 3, 6, 5, 9, 8]
                    fig = visualize_heap(min_heap, "min")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    max_heap = [9, 8, 6, 5, 3, 1]
                    fig = visualize_heap(max_heap, "max")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif "recursion" in selected_topic.lower():
                fig = visualize_recursion_tree("fibonacci")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🔄 Show Call Stack"):
                    st.code("""
Call Stack Visualization:
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2) → 1
│   │   └── fib(1) → 1
│   └── fib(2) → 1
└── fib(3) → 2
                    """)
            
            elif "bit manipulation" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    num1 = st.number_input("Number 1", value=12, min_value=0, max_value=255)
                with col2:
                    num2 = st.number_input("Number 2", value=10, min_value=0, max_value=255)
                
                operation = st.selectbox("Operation", ["AND", "OR", "XOR"])
                fig = visualize_bit_operations(num1, num2, operation)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "dynamic programming" in selected_topic.lower() or "dp" in selected_topic.lower():
                # Example DP table for fibonacci
                dp_table = [[0, 1, 1, 2, 3, 5, 8, 13]]
                fig = visualize_dp_table(dp_table, "Fibonacci DP")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("DP Table shows memoized results to avoid recomputation")
            
            elif "hashing" in selected_topic.lower():
                # Hash table visualization
                hash_data = {
                    'Key': ['apple', 'banana', 'cherry', 'date'],
                    'Hash': [hash('apple') % 10, hash('banana') % 10, hash('cherry') % 10, hash('date') % 10],
                    'Value': [5, 7, 3, 9]
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=hash_data['Key'], y=hash_data['Hash'], name='Hash Values'),
                ])
                fig.update_layout(title="Hash Function Visualization", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "backtracking" in selected_topic.lower():
                st.markdown("### 🔄 Backtracking Visualization")
                st.code("""
N-Queens Solution Tree:
Try Queen at (0,0) ✓
├── Try Queen at (1,2) ✓
│   ├── Try Queen at (2,1) ✗ (Conflict!)
│   └── Backtrack...
└── Try Queen at (1,3) ✓
    └── Continue...
                """)
            
            elif "graph" in selected_topic.lower():
                st.info("Graph visualization - BFS/DFS traversal animations")
                # Add graph visualization here
            
            # Interactive complexity analysis
            st.markdown("### 📈 Complexity Analysis")
            
            n_values = list(range(1, 1001, 50))
            complexities = {
                "O(1)": [1] * len(n_values),
                "O(log n)": [math.log2(n) for n in n_values],
                "O(n)": n_values,
                "O(n log n)": [n * math.log2(n) for n in n_values],
                "O(n²)": [n**2 for n in n_values],
            }
            
            complexity_df = pd.DataFrame(complexities, index=n_values)
            st.line_chart(complexity_df)
    
    else:
        st.markdown("### 🎯 Try the Daily Challenge above or generate a new problem from the sidebar!")

if __name__ == "__main__":
    main()
